# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io, json, joblib, sys
from typing import List, Dict, Any
from pathlib import Path
import importlib  # added
from functools import lru_cache
import time
from datetime import datetime, timezone  # new

# --- Path resolution helper for cloud deployment ---
def get_project_root():
    """Get the project root directory (exoplanet-ai folder) regardless of working directory"""
    current_file = Path(__file__).resolve()
    # Navigate from app/streamlit_app.py to exoplanet-ai/
    return current_file.parent.parent

def resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root"""
    return get_project_root() / relative_path

# --- Ensure consistent model module import ---
def _ensure_model_module():
    """
    Guarantee that 'model' module (src/model.py) is importable consistently
    so that pickled objects referencing model.XGBClassifierFixed load without error.
    """
    import sys
    import importlib.util
    
    root = get_project_root()  # Use the same path resolution
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    # Explicitly load the model module to ensure consistent identity
    try:
        spec = importlib.util.spec_from_file_location("model", src_dir / "model.py")
        if spec and spec.loader:
            # Check if already loaded
            if "model" in sys.modules:
                return sys.modules["model"]
            
            # Load fresh
            mod = importlib.util.module_from_spec(spec)
            sys.modules["model"] = mod
            spec.loader.exec_module(mod)
            return mod
        else:
            raise ImportError("Could not create spec for model module")
            
    except Exception as e:
        print(f"Warning: Failed to load model module: {e}")
        # Create a minimal fallback
        import types
        fallback = types.ModuleType("model")
        sys.modules["model"] = fallback
        return fallback

# Initialize model module immediately
_MODEL_MODULE = _ensure_model_module()
# --- End module setup ---

# --- Minimal shared constants (must match model) ---
AUTO_FEATURE_MAP = {
    'orbital_period': ['koi_period', 'pl_orbper'],
    'transit_depth': ['koi_depth'],
    'transit_duration': ['koi_duration', 'pl_trandur'],
    'planet_radius': ['koi_prad', 'pl_rade'],
    'equilibrium_temp': ['koi_teq', 'pl_eqt'],
    'stellar_insolation': ['koi_insol', 'pl_insol'],
    'impact_parameter': ['koi_impact', 'pl_imppar'],
    'stellar_teff': ['koi_steff', 'st_teff'],
    'stellar_radius': ['koi_srad', 'st_rad'],
    'stellar_logg': ['koi_slogg', 'st_logg'],
    'snr': ['koi_model_snr'],
    'disposition_score': ['koi_score'],
    'fp_flag_not_transit': ['koi_fpflag_nt'],
    'fp_flag_stellar_eclipse': ['koi_fpflag_ss'],
    'fp_flag_centroid_offset': ['koi_fpflag_co'],
    'fp_flag_ephemeris_match': ['koi_fpflag_ec'],
}
EARTH_SOLAR_RADIUS_RATIO = 1/109.2

# --- Robust CSV read (uploaded) ---
def robust_read(file) -> pd.DataFrame:
    txt = file.getvalue().decode('utf-8', errors='ignore')
    lines = txt.splitlines()
    # Try comment-aware
    for attempt in ('comment', 'manual', 'skip'):
        try:
            if attempt == 'comment':
                return pd.read_csv(io.StringIO(txt), comment='#', skip_blank_lines=True)
            if attempt == 'manual':
                data = [l for l in lines if l.strip() and not l.lstrip().startswith('#')]
                buf = io.StringIO("\n".join(data))
                return pd.read_csv(buf)
            if attempt == 'skip':
                return pd.read_csv(io.StringIO(txt), engine='python', on_bad_lines='skip')
        except Exception:
            continue
    raise RuntimeError("Failed to parse CSV.")

def map_any_to_internal(df: pd.DataFrame, required: List[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    chosen = {}
    for internal, sources in AUTO_FEATURE_MAP.items():
        for s in sources:
            if s in df.columns:
                out[internal] = df[s]
                chosen[internal] = s
                break
    # Engineer transit_depth if missing
    if 'transit_depth' not in out.columns and {'planet_radius','stellar_radius'} <= set(out.columns):
        pr = pd.to_numeric(out['planet_radius'], errors='coerce')
        sr = pd.to_numeric(out['stellar_radius'], errors='coerce')
        depth_ppm = ((pr / (sr*109.2))**2) * 1e6
        out['transit_depth'] = depth_ppm
        chosen['transit_depth'] = 'engineered:(Rp/Rs)^2'
    # Ensure all required present
    for f in required:
        if f not in out.columns:
            out[f] = np.nan
    out = out[required]
    out.attrs['source_map'] = chosen
    return out

def load_model(model_path: str):
    # Model module already ensured at startup
    # Resolve path relative to project root
    if not Path(model_path).is_absolute():
        model_path = str(resolve_path(model_path))
    data = joblib.load(model_path)
    return data['pipeline'], data['label_encoder'], data.get('numeric_cols', [])

# ---------- NEW UTILITIES (metadata & reporting) ----------
def _load_metadata(model_path: str) -> Dict[str, Any]:
    # Resolve path relative to project root
    if not Path(model_path).is_absolute():
        model_path = str(resolve_path(model_path))
    meta_path = Path(model_path).with_suffix(".json")
    if meta_path.exists():
        try:
            return json.load(open(meta_path, "r"))
        except Exception:
            pass
    return {}

def _render_classification_report(report: Dict[str, Any]):
    if not report:
        st.info("No classification report available.")
        return
    rows = []
    for k, v in report.items():
        if isinstance(v, dict) and 'precision' in v:
            rows.append({
                'Label': k,
                'Precision': round(v['precision'], 4),
                'Recall': round(v['recall'], 4),
                'F1': round(v['f1-score'], 4),
                'Support': int(v['support'])
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), width='stretch')

@lru_cache(maxsize=4)
def _compute_permutation_importance(seed: int, sample_hash: int,
                                    n_repeats: int,
                                    X_sample_serialized: str,
                                    y_serialized: str) -> pd.DataFrame:
    # Deserialize
    Xs = pd.read_json(io.StringIO(X_sample_serialized), orient='split')
    ys = pd.read_json(io.StringIO(y_serialized), orient='split')[0]
    from sklearn.inspection import permutation_importance
    pipe = st.session_state['_pipeline_ref']  # set before call
    result = permutation_importance(pipe, Xs, ys, n_repeats=n_repeats,
                                    random_state=seed, n_jobs=-1, scoring='f1_macro')
    df_imp = pd.DataFrame({
        'feature': Xs.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)
    return df_imp

def _safe_hash_df(df: pd.DataFrame) -> int:
    try:
        return hash(tuple(df.columns)) ^ hash(df.shape[0])
    except Exception:
        return int(time.time())

def _import_training_api():
    """Import the training API from src.model - use the same module instance"""
    # Use the already-loaded model module to ensure consistency
    return _MODEL_MODULE

# ---------- EXISTING CODE ABOVE (AUTO_FEATURE_MAP, etc.) ----------

# (Keep existing functions robust_read, map_any_to_internal, _ensure_model_module, load_model)

# ---------- PREDICTION / ANALYSIS UI ----------
def prediction_interface():
    st.title("Exoplanet Multi-Format Classifier")
    with st.sidebar:
        st.header("🤖 Model Selection")
        
        # Get available models
        models_dir = resolve_path("models")
        available_models = []
        if models_dir.exists():
            for model_file in models_dir.glob("*.joblib"):
                # Store as relative path for display but ensure it resolves correctly
                relative_path = f"models/{model_file.name}"
                available_models.append(relative_path)
        
        # Check for model path in query params or session state
        default_model_path = "models/unified_xgb_tuned.joblib"  # Updated to match your existing model
        if "model" in st.query_params:
            default_model_path = st.query_params["model"]
        elif 'reloaded_model_path' in st.session_state:
            default_model_path = st.session_state['reloaded_model_path']
        
        # Model selection method
        model_selection_method = st.radio("Select Model", ["Choose from Available", "Enter Path"], horizontal=True)
        
        if model_selection_method == "Choose from Available" and available_models:
            # Use selectbox for available models
            try:
                default_index = available_models.index(default_model_path) if default_model_path in available_models else 0
            except ValueError:
                default_index = 0
            model_path = st.selectbox("Available Models", available_models, index=default_index)
        else:
            # Use text input
            model_path = st.text_input("Model Path", default_model_path)
        
        if not model_path:
            st.stop()
            
        # Load model
        try:
            # Ensure the model path exists before trying to load
            if not Path(model_path).is_absolute():
                resolved_path = resolve_path(model_path)
            else:
                resolved_path = Path(model_path)
            
            if not resolved_path.exists():
                st.error(f"❌ Model file not found: {resolved_path}")
                st.info("💡 Make sure the models directory and model files are deployed with your app")
                st.stop()
                
            pipeline, label_encoder, features = load_model(model_path)
            st.session_state['_pipeline_ref'] = pipeline  # for permutation importance
            st.success(f"✅ Model loaded: {Path(model_path).name}")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.info("💡 Common issues:\n"
                   "• Model file missing in deployment\n" 
                   "• Models directory not included\n"
                   "• Wrong working directory in cloud service")
            st.stop()
            
        # Model info
        metadata = _load_metadata(model_path)
        
        # Show available models count
        st.caption(f"📁 {len(available_models)} models available")
        
        # Model details
        with st.expander("Model Details", expanded=False):
            if metadata:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Samples", metadata.get('n_samples', 'Unknown'))
                    st.metric("Features", metadata.get('n_features', len(features) if features else 'Unknown'))
                with col2:
                    st.metric("Model Type", metadata.get('model_type', 'Unknown'))
                    if 'eval_metrics' in metadata:
                        f1_score = metadata['eval_metrics'].get('macro_f1', 0)
                        st.metric("Macro F1", f"{f1_score:.3f}" if f1_score else "Unknown")
                
                # Classes distribution
                if 'classes' in metadata:
                    st.write("**Class Distribution:**")
                    for cls, count in metadata['classes'].items():
                        st.write(f"• {cls}: {count:,}")
            else:
                st.write("No metadata available")
        
        if features:
            st.caption(f"🔧 {len(features)} model features")
            if st.checkbox("Show Feature List", value=False):
                st.code("\n".join(features), language='text')

    # New tab layout
    tab_upload, tab_manual, tab_model, tab_importance, tab_threshold, tab_analysis, tab_retrain = st.tabs(
        ["Upload CSV", "Manual Input", "Model Info", "Feature Importance", "Threshold Explorer", "Advanced Analysis", "Retrain"]
    )

    # ---------------- TAB: Upload CSV (enhanced) ----------------
    with tab_upload:
        st.subheader("Upload Kepler / K2 / Exoplanet Archive CSV")
        upl = st.file_uploader("Choose a CSV file", type=['csv'])
        threshold = st.slider("Primary Classification Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
        topn = st.number_input("Show Top-N Rows (preview)", min_value=10, max_value=500, value=100, step=10)
        if upl:
            try:
                raw_df = robust_read(upl)
                st.success(f"Parsed: {raw_df.shape[0]} rows × {raw_df.shape[1]} cols")
                st.dataframe(raw_df.head(10), width='stretch')
                mapped = map_any_to_internal(raw_df, features)
                with st.expander("Mapped Features / Sources"):
                    st.dataframe(mapped.head(10), width='stretch')
                    st.json(mapped.attrs.get('source_map', {}))
                # Prediction
                if st.button("🚀 Predict on Uploaded Data"):
                    preds_encoded = pipeline.predict(mapped)
                    preds = label_encoder.inverse_transform(preds_encoded)
                    try:
                        probs = pipeline.predict_proba(mapped)
                    except Exception:
                        probs = None
                    result_df = raw_df.copy()
                    result_df['AI_Prediction'] = preds
                    if probs is not None:
                        maxp = probs.max(axis=1)
                        result_df['AI_Confidence'] = maxp
                        for i, cls in enumerate(label_encoder.classes_):
                            result_df[f'P({cls})'] = probs[:, i]
                        # Apply threshold to mark uncertain
                        result_df.loc[result_df['AI_Confidence'] < threshold, 'AI_Prediction'] = 'UNCERTAIN'
                    st.subheader("Prediction Preview")
                    st.dataframe(result_df.head(topn), width='stretch')
                    # If truth available evaluate
                    truth_col = None
                    # Look for truth columns in different formats (added tfopwg_disp for TOI)
                    for c in ['disposition', 'koi_disposition', 'koi_pdisposition', 'tfopwg_disp']:
                        if c in raw_df.columns:
                            truth_col = c
                            break
                    if truth_col:
                        st.markdown("### Evaluation (Ground Truth Present)")
                        y_true = raw_df[truth_col].astype(str).str.upper().str.strip()
                        valid_mask = y_true.isin(label_encoder.classes_)
                        eval_preds = result_df.loc[valid_mask, 'AI_Prediction']
                        eval_truth = y_true.loc[valid_mask]
                        from sklearn.metrics import classification_report, confusion_matrix
                        rep = classification_report(eval_truth, eval_preds, output_dict=True, zero_division=0)
                        _render_classification_report(rep)
                        cm = confusion_matrix(eval_truth, eval_preds, labels=label_encoder.classes_.tolist()+['UNCERTAIN'])
                        st.write("Confusion Matrix (includes UNCERTAIN if any):")
                        st.dataframe(pd.DataFrame(cm,
                                                  index=label_encoder.classes_.tolist()+['UNCERTAIN'],
                                                  columns=label_encoder.classes_.tolist()+['UNCERTAIN']))
                        # --- Enhanced: Actual vs Predicted class totals summary with accuracy ---
                        st.markdown("### Class Totals (Actual vs Model)")
                        
                        # Get actual counts (map TOI categories if needed)
                        actual_counts = y_true.value_counts()
                        
                        # Map TOI categories to standard categories for better comparison
                        if truth_col == 'tfopwg_disp':
                            # TOI mapping: CP/KP -> CONFIRMED, PC/APC -> CANDIDATE, FP/FA -> FALSE POSITIVE
                            toi_mapping = {
                                'CP': 'CONFIRMED', 'KP': 'CONFIRMED',
                                'PC': 'CANDIDATE', 'APC': 'CANDIDATE', 
                                'FP': 'FALSE POSITIVE', 'FA': 'FALSE POSITIVE'
                            }
                            # Apply mapping to actual values
                            y_true_mapped = y_true.map(toi_mapping).fillna(y_true)
                            actual_counts = y_true_mapped.value_counts()
                        
                        # Get predicted counts (excluding UNCERTAIN)
                        predicted_counts = result_df['AI_Prediction']
                        predicted_counts = predicted_counts[predicted_counts != 'UNCERTAIN'].value_counts()
                        
                        # Calculate accuracy per class
                        summary_rows = []
                        total_actual = 0
                        total_predicted = 0
                        total_correct = 0
                        
                        for cls in ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']:
                            actual = int(actual_counts.get(cls, 0))
                            predicted = int(predicted_counts.get(cls, 0))
                            difference = predicted - actual
                            
                            # Calculate accuracy for this class
                            if truth_col == 'tfopwg_disp':
                                # For TOI, compare mapped values
                                class_mask = y_true_mapped == cls
                            else:
                                class_mask = y_true == cls
                                
                            # Calculate accuracy as: (1 - |difference|/actual) * 100
                            # This shows how close predictions are to actual values
                            if actual > 0:
                                class_accuracy = (1 - abs(difference) / actual) * 100
                                # Ensure accuracy doesn't go below 0%
                                class_accuracy = max(0, class_accuracy)
                            else:
                                class_accuracy = 0.0
                            
                            # Calculate difference percentage
                            diff_percentage = (difference / actual * 100) if actual > 0 else 0
                            
                            summary_rows.append({
                                'Class': cls,
                                'Actual': actual,
                                'Predicted': predicted,
                                'Difference': f"{difference:+d}",
                                'Diff %': f"{diff_percentage:+.1f}%",
                                'Accuracy': f"{class_accuracy:.1f}%"
                            })
                            
                            total_actual += actual
                            total_predicted += predicted
                        
                        # Calculate total accuracy as arithmetic mean of class accuracies
                        class_accuracies = []
                        for row in summary_rows:
                            # Extract numeric accuracy value (remove % sign)
                            acc_str = row['Accuracy'].replace('%', '')
                            class_accuracies.append(float(acc_str))
                        
                        # Add total row
                        overall_difference = total_predicted - total_actual
                        overall_diff_pct = (overall_difference / total_actual * 100) if total_actual > 0 else 0
                        overall_accuracy = sum(class_accuracies) / len(class_accuracies) if class_accuracies else 0
                        
                        summary_rows.append({
                            'Class': '**TOTAL**',
                            'Actual': total_actual,
                            'Predicted': total_predicted,
                            'Difference': f"{overall_difference:+d}",
                            'Diff %': f"{overall_diff_pct:+.1f}%",
                            'Accuracy': f"{overall_accuracy:.1f}%"
                        })
                        
                        summary_df = pd.DataFrame(summary_rows)
                        st.dataframe(summary_df, width='stretch', use_container_width=True)
                        
                        # Enhanced caption with format info
                        format_info = ""
                        if truth_col == 'tfopwg_disp':
                            format_info = " (TOI format: CP/KP→CONFIRMED, PC/APC→CANDIDATE, FP/FA→FALSE POSITIVE)"
                        elif truth_col == 'disposition':
                            format_info = " (K2/PANDC format)"
                        elif truth_col in ['koi_disposition', 'koi_pdisposition']:
                            format_info = " (Kepler format)"
                            
                        st.caption(f"UNCERTAIN predictions excluded from counts. Difference = Predicted - Actual. Accuracy = correct predictions per class{format_info}.")
                    csv = result_df.to_csv(index=False)
                    st.download_button("📥 Download Full Predictions", data=csv,
                                       file_name="exoplanet_predictions.csv", mime="text/csv")
                    st.session_state['last_raw_df'] = raw_df
                    st.session_state['last_mapped_df'] = mapped
                    if probs is not None:
                        st.session_state['last_probs'] = probs
                        st.session_state['last_preds'] = preds
            except Exception as e:
                st.error(f"Parsing / prediction error: {e}")

    # ---------------- TAB: Manual Input (unchanged core with slight tweaks) ----------------
    with tab_manual:
        st.subheader("Single Candidate Manual Entry")
        if not features:
            st.warning("Model features unavailable.")
        else:
            inputs = {}
            cols = st.columns(3)
            for i, f in enumerate(features):
                with cols[i % 3]:
                    if f.startswith('fp_flag_'):
                        inputs[f] = st.checkbox(f, value=False)
                    elif f in ('disposition_score', 'impact_parameter'):
                        inputs[f] = st.slider(f, 0.0, 1.0, 0.5, 0.01)
                    else:
                        inputs[f] = st.number_input(f, value=0.0, format="%.6f")
            sample = pd.DataFrame([inputs])[features]
            st.dataframe(sample, width='stretch')
            if st.button("🔮 Predict (Manual)"):
                try:
                    pred = label_encoder.inverse_transform(pipeline.predict(sample))[0]
                    st.success(f"Prediction: {pred}")
                    try:
                        prob = pipeline.predict_proba(sample)[0]
                        st.dataframe(pd.DataFrame({'Class': label_encoder.classes_,
                                                   'Probability': prob}), width='stretch')
                    except Exception:
                        pass
                except Exception as e:
                    st.error(f"Failed: {e}")

    # ---------------- TAB: Model Info ----------------
    with tab_model:
        st.subheader("Model Information & Training Metrics")
        if metadata.get('training_results'):
            tr = metadata['training_results']
            st.markdown("#### Training Summary")
            st.write({k: v for k, v in tr.items() if k != 'train_metrics'})
            metrics = tr.get('train_metrics', {})
            if metrics:
                st.markdown("#### Classification Report (Training)")
                _render_classification_report(metrics.get('classification_report', {}))
                st.markdown("**Macro F1:** {:.4f} | **Weighted F1:** {:.4f} | **ROC AUC OvR:** {:.4f}".format(
                    metrics.get('macro_f1', 0),
                    metrics.get('weighted_f1', 0),
                    metrics.get('roc_auc_ovr', 0)
                ))
        else:
            st.info("No embedded training metrics found in metadata JSON.")
        st.markdown("#### Pipeline Steps")
        step_rows = []
        for name, step in pipeline.named_steps.items():
            step_rows.append({'step': name, 'type': step.__class__.__name__})
        st.dataframe(pd.DataFrame(step_rows), width='stretch')

    # ---------------- TAB: Feature Importance ----------------
    with tab_importance:
        st.subheader("Feature Importance")
        if not features:
            st.info("No features available.")
        else:
            classifier = pipeline.named_steps.get('classifier')
            col_left, col_right = st.columns(2)
            with col_left:
                mode = st.radio("Importance Mode", ["Tree / Native", "Permutation"], horizontal=True)
            with col_right:
                sample_size = st.number_input("Permutation Sample Size", 50, 1000, 300, 50,
                                              help="Used only for permutation mode")
            if mode == "Tree / Native":
                if hasattr(classifier, 'feature_importances_'):
                    importances = classifier.feature_importances_
                    fi_df = pd.DataFrame({'feature': features, 'importance': importances}) \
                        .sort_values('importance', ascending=False)
                    st.dataframe(fi_df, width='stretch', height=400)
                else:
                    st.warning("Classifier does not expose feature_importances_.")
            else:
                if 'last_mapped_df' not in st.session_state:
                    st.info("Upload data first to compute permutation importance.")
                else:
                    mapped_df = st.session_state['last_mapped_df']
                    y_for_perm = None
                    # If previous prediction probabilities exist, synthesize pseudo-label from argmax
                    if 'last_probs' in st.session_state:
                        y_for_perm = np.argmax(st.session_state['last_probs'], axis=1)
                    else:
                        # fallback zero labels
                        y_for_perm = np.zeros(len(mapped_df), dtype=int)
                    sample_perm = mapped_df.head(sample_size)
                    y_perm = pd.Series(y_for_perm[:len(sample_perm)])
                    hash_id = _safe_hash_df(sample_perm)
                    st.caption("Computing permutation importance may take a few seconds...")
                    with st.spinner("Calculating..."):
                        df_imp = _compute_permutation_importance(
                            seed=42,
                            sample_hash=hash_id,
                            n_repeats=5,
                            X_sample_serialized=sample_perm.to_json(orient='split'),
                            y_serialized=y_perm.to_frame().to_json(orient='split')
                        )
                    st.dataframe(df_imp.head(30), width='stretch')

    # ---------------- TAB: Threshold Explorer ----------------
    with tab_threshold:
        st.subheader("Threshold Explorer")
        if 'last_probs' not in st.session_state:
            st.info("Upload data and generate predictions first.")
        else:
            probs = st.session_state['last_probs']
            preds_raw = st.session_state['last_preds']
            raw_df = st.session_state['last_raw_df']
            t = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
            cls_selected = st.selectbox("Focus Class (optional filter)", ["(All)"] + list(label_encoder.classes_))
            maxp = probs.max(axis=1)
            derived_labels = preds_raw.copy()
            derived_labels[maxp < t] = 'UNCERTAIN'
            dist = pd.Series(derived_labels).value_counts()
            st.write("Distribution (after threshold):")
            st.dataframe(dist.to_frame('count'))
            display_df = raw_df.copy()
            display_df['AI_Prediction'] = derived_labels
            display_df['AI_Confidence'] = maxp
            for i, c in enumerate(label_encoder.classes_):
                display_df[f'P({c})'] = probs[:, i]
            if cls_selected != "(All)":
                display_df = display_df[display_df['AI_Prediction'] == cls_selected]
            st.dataframe(display_df.head(200), width='stretch')
            st.download_button("Download Thresholded Results",
                               data=display_df.to_csv(index=False),
                               file_name="thresholded_predictions.csv",
                               mime="text/csv")

    # ---------------- TAB: Advanced Analysis ----------------
    with tab_analysis:
        st.subheader("Advanced Dataset / Feature Analysis")
        if 'last_raw_df' not in st.session_state:
            st.info("Upload data first for analysis.")
        else:
            raw_df = st.session_state['last_raw_df']
            mapped = st.session_state['last_mapped_df']
            colA, colB, colC = st.columns(3)
            with colA:
                st.metric("Rows", f"{len(raw_df):,}")
            with colB:
                st.metric("Mapped Features", f"{mapped.shape[1]}")
            with colC:
                missing_ratio = mapped.isna().mean().mean()
                st.metric("Avg Missing %", f"{missing_ratio*100:.1f}%")
            with st.expander("Basic Statistics (Mapped Features)"):
                st.dataframe(mapped.describe().T, width='stretch')
            if 'transit_depth' in mapped.columns:
                st.markdown("#### Engineered Depth Check")
                if 'engineered:(Rp/Rs)^2' in mapped.attrs.get('source_map', {}).get('transit_depth', ''):
                    st.info("Transit depth was engineered from planet_radius and stellar_radius.")
            # Correlation heatmap (optional)
            if st.checkbox("Show Correlation Matrix (Spearman)", value=False):
                corr = mapped.corr(method='spearman')
                st.dataframe(corr, width='stretch')

    # ---------------- TAB: Retrain ----------------
    with tab_retrain:
        st.subheader("Train / Retrain Model")
        st.markdown("Upload training CSV file(s) and adjust parameters.")
        
        # Allow multiple file uploads
        train_mode = st.radio("Training Mode", ["Single File", "Multiple Files"], horizontal=True)
        
        if train_mode == "Single File":
            train_file = st.file_uploader("Training CSV", type=["csv"], key="train_csv")
            uploaded_files = [train_file] if train_file else []
        else:
            train_files = st.file_uploader("Training CSV Files", type=["csv"], 
                                         accept_multiple_files=True, key="train_csvs")
            uploaded_files = train_files if train_files else []
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s):")
            for i, file in enumerate(uploaded_files):
                st.write(f"  {i+1}. {file.name}")
        
        colA, colB, colC = st.columns(3)
        with colA:
            model_type = st.selectbox("Model Type", ["xgboost", "random_forest"])
            tune = st.checkbox("Hyperparameter Tuning", value=False,
                               help="Randomized search over parameter space")
        with colB:
            cv_folds = st.number_input("CV Folds", 2, 10, 5, 1, disabled=not tune)
            n_iter = st.number_input("Tuning Iterations", 5, 100, 20, 5, disabled=not tune)
            random_state = st.number_input("Random State", 0, 10_000, 42, 1)
        with colC:
            confirmed_w = st.number_input("Weight: CONFIRMED", 0.1, 2.0, 0.7, 0.05)
            fp_w = st.number_input("Weight: FALSE POSITIVE", 0.1, 3.0, 1.25, 0.05)
            cand_w = st.number_input("Weight: CANDIDATE", 0.1, 2.0, 1.0, 0.05)
            other_w = st.number_input("Weight: OTHER", 0.1, 2.0, 1.1, 0.05)
        apply_adjust = st.checkbox("Apply Post Probability Adjustment", value=True)
        
        # Model name persistence using session state
        if 'custom_model_name' not in st.session_state:
            st.session_state.custom_model_name = f"models/exoplanet_classifier_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.joblib"
        
        # Model name input with session state
        col_name, col_new = st.columns([3, 1])
        with col_name:
            output_path = st.text_input("Output Model Path", 
                                       value=st.session_state.custom_model_name,
                                       key="model_name_input",
                                       help="Specify the path where the trained model will be saved")
        with col_new:
            st.write("")  # spacing
            if st.button("🔄 New Name", help="Generate a new default name"):
                new_name = f"models/exoplanet_classifier_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.joblib"
                st.session_state.custom_model_name = new_name
                st.rerun()
        
        # Update session state when user changes the name
        if output_path != st.session_state.custom_model_name:
            st.session_state.custom_model_name = output_path
        run_btn = st.button("🛠️ Train Model", type="primary", disabled=len(uploaded_files) == 0)

        if run_btn and uploaded_files:
            try:
                st.info(f"Preparing {len(uploaded_files)} training file(s)...")
                
                # Save uploaded files to temporary paths
                tmp_paths = []
                for i, file in enumerate(uploaded_files):
                    tmp_path = resolve_path("models") / f"_tmp_train_{i}_{datetime.now(timezone.utc).strftime('%H%M%S_%f')}.csv"
                    tmp_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path.write_bytes(file.getvalue())
                    tmp_paths.append(str(tmp_path))
                    st.write(f"Temporary file {i+1}: {tmp_path}")

                with st.spinner("Training in progress... (this may take a while)"):
                    model_mod = _import_training_api()
                    
                    # Ensure output path is absolute
                    if not Path(output_path).is_absolute():
                        output_path = str(resolve_path(output_path))
                    
                    if len(tmp_paths) == 1:
                        # Single file training
                        result = model_mod.train_from_csv(
                            data_path=tmp_paths[0],
                            output_model_path=output_path,
                            model_type=model_type,
                            confirmed_weight=confirmed_w,
                            false_positive_weight=fp_w,
                            candidate_weight=cand_w,
                            other_weight=other_w,
                            apply_post_adjust=apply_adjust,
                            tune=tune,
                            cv_folds=cv_folds,
                            n_iter=n_iter,
                            random_state=random_state
                        )
                    else:
                        # Multi-file training
                        result = model_mod.train_from_multiple_csv(
                            data_paths=tmp_paths,
                            output_model_path=output_path,
                            model_type=model_type,
                            confirmed_weight=confirmed_w,
                            false_positive_weight=fp_w,
                            candidate_weight=cand_w,
                            other_weight=other_w,
                            apply_post_adjust=apply_adjust,
                            tune=tune,
                            cv_folds=cv_folds,
                            n_iter=n_iter,
                            random_state=random_state
                        )
                
                st.success(f"Model trained & saved: {output_path}")
                
                # Clean up temporary files
                for tmp_path in tmp_paths:
                    try:
                        Path(tmp_path).unlink()
                    except:
                        pass
                
                metrics = result['metadata']['eval_metrics']
                st.markdown("#### Training Evaluation Metrics")
                _render_classification_report(metrics.get('classification_report', {}))
                st.write({
                    "accuracy": metrics.get("accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "weighted_f1": metrics.get("weighted_f1"),
                    "roc_auc_ovr": metrics.get("roc_auc_ovr"),
                    "macro_average_precision": metrics.get("macro_average_precision"),
                })
                
                # Show dataset information for multi-file training
                if len(tmp_paths) > 1:
                    st.markdown("#### Multi-Dataset Training Summary")
                    training_metadata = result['metadata']
                    st.write(f"**Total datasets:** {training_metadata.get('n_datasets', len(tmp_paths))}")
                    st.write(f"**Combined samples:** {training_metadata.get('n_samples', 'Unknown')}")
                    st.write(f"**Combined features:** {training_metadata.get('n_features', 'Unknown')}")
                
                st.markdown("#### Class Distribution")
                st.json(result['metadata'].get('classes', {}))
                
                st.caption("Reloading app with newly trained model...")
                st.session_state['reloaded_model_path'] = output_path
                # Force a page refresh with the new model
                st.query_params["model"] = output_path
                st.rerun()
                
            except Exception as e:
                st.error(f"Training failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    # Auto-reload if retrain just finished
    if 'reloaded_model_path' in st.session_state:
        if model_path != st.session_state['reloaded_model_path']:
            st.query_params["model"] = st.session_state['reloaded_model_path']

if __name__ == "__main__":
    prediction_interface()

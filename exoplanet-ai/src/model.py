#!/usr/bin/env python3
"""
Machine Learning Pipeline for Exoplanet Classification

This module provides a complete ML pipeline for classifying exoplanet candidates
into three categories: CONFIRMED, CANDIDATE, and FALSE POSITIVE.
"""

import json
import joblib
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import randint, uniform
import io, csv
from pandas.errors import ParserError

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True

    class XGBClassifierFixed(XGBClassifier):
        """
        Wrapper to:
        - Auto-set objective/num_class in fit based on y.
        - Guarantee predict returns 1D class indices (not probability matrix).
        """
        def __init__(self, **kwargs):
            # Pass all parameters to parent constructor with explicit defaults
            super().__init__(**kwargs)
            # Ensure consistent module reference for pickling
            self.__class__.__module__ = 'model'
            
        def fit(self, X, y, **kwargs):
            y_arr = np.asarray(y)
            classes = np.unique(y_arr)
            n_classes = len(classes)
            if n_classes <= 2:
                # Binary case
                if getattr(self, "objective", None) != "binary:logistic":
                    self.set_params(objective="binary:logistic")
                # Remove stale num_class if previously set
                if hasattr(self, "num_class"):
                    try:
                        delattr(self, "num_class")
                    except Exception:
                        pass
            else:
                self.set_params(objective="multi:softprob", num_class=n_classes)
            return super().fit(X, y, **kwargs)

        def predict(self, X, **kwargs):
            preds = super().predict(X, **kwargs)
            # If probabilities (multi:softprob) -> argmax
            if isinstance(preds, np.ndarray) and preds.ndim == 2:
                return np.argmax(preds, axis=1)
            return preds
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available, using RandomForest as fallback")

try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Warning: imbalanced-learn not available, skipping SMOTE")

# --- Comprehensive feature mapping from Kepler KOI columns ---
# Original KEPLER_FEATURE_MAP retained for backward compatibility
KEPLER_FEATURE_MAP = {
    # Core transit features (KOI)
    'koi_period': 'orbital_period',
    'koi_depth': 'transit_depth',
    'koi_duration': 'transit_duration',
    'koi_prad': 'planet_radius',
    'koi_teq': 'equilibrium_temp',
    'koi_insol': 'stellar_insolation',
    'koi_impact': 'impact_parameter',
    # Stellar properties
    'koi_steff': 'stellar_teff',
    'koi_srad': 'stellar_radius',
    'koi_slogg': 'stellar_logg',
    # Signal quality
    'koi_model_snr': 'snr',
    'koi_score': 'disposition_score',
    # False positive flags
    'koi_fpflag_nt': 'fp_flag_not_transit',
    'koi_fpflag_ss': 'fp_flag_stellar_eclipse',
    'koi_fpflag_co': 'fp_flag_centroid_offset',
    'koi_fpflag_ec': 'fp_flag_ephemeris_match',
}

# NEW: Multi-format mapping priorities per internal feature
# First present source column in list is used.
AUTO_FEATURE_MAP = {
    'orbital_period': ['koi_period', 'pl_orbper'],
    'transit_depth': ['koi_depth'],  # may be engineered later
    'transit_duration': ['koi_duration', 'pl_trandur'],
    'planet_radius': ['koi_prad', 'pl_rade'],  # pl_rade in Earth radii
    'equilibrium_temp': ['koi_teq', 'pl_eqt'],
    'stellar_insolation': ['koi_insol', 'pl_insol'],
    'impact_parameter': ['koi_impact', 'pl_imppar'],
    'stellar_teff': ['koi_steff', 'st_teff'],
    'stellar_radius': ['koi_srad', 'st_rad'],
    'stellar_logg': ['koi_slogg', 'st_logg'],
    'snr': ['koi_model_snr'],  # rarely in pl_* export
    'disposition_score': ['koi_score'],
    # No direct analog FP flags in pl_* dataset
    'fp_flag_not_transit': ['koi_fpflag_nt'],
    'fp_flag_stellar_eclipse': ['koi_fpflag_ss'],
    'fp_flag_centroid_offset': ['koi_fpflag_co'],
    'fp_flag_ephemeris_match': ['koi_fpflag_ec'],
}

EARTH_SOLAR_RADIUS_RATIO = 1.0 / 109.2  # Re / Rsun for depth approximation

def _extract_multi_format_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature frame from any of:
      - Kepler KOI (koi_*)
      - Exoplanet Archive (pl_*, st_*)
      - Mixed K2 EPIC exports
    Falls back to engineering transit_depth if missing.
    """
    feat_df = pd.DataFrame(index=df.index)
    chosen_sources = {}
    # Pass 1: direct extraction
    for internal, candidates in AUTO_FEATURE_MAP.items():
        for src in candidates:
            if src in df.columns:
                feat_df[internal] = df[src]
                chosen_sources[internal] = src
                break
    # Pass 2: legacy explicit KEPLER_FEATURE_MAP if something missing
    for src, internal in KEPLER_FEATURE_MAP.items():
        if internal not in feat_df.columns and src in df.columns:
            feat_df[internal] = df[src]
            chosen_sources.setdefault(internal, src)
    # Pass 3: approximate transit depth if absent
    if 'transit_depth' not in feat_df.columns:
        if 'planet_radius' in feat_df.columns and 'stellar_radius' in feat_df.columns:
            pr = feat_df['planet_radius'].astype(float)
            sr = feat_df['stellar_radius'].astype(float)
            with np.errstate(divide='ignore', invalid='ignore'):
                depth_ppm = ((pr / (sr * 109.2)) ** 2) * 1e6
            feat_df['transit_depth'] = depth_ppm.replace([np.inf, -np.inf], np.nan)
            chosen_sources['transit_depth'] = 'engineered:(Rp/Rs)^2'
    feat_df.attrs['source_columns'] = chosen_sources
    return feat_df

# Core features that should always be present
CORE_FEATURES = [
    'orbital_period',
    'transit_depth',
    'transit_duration',
    'planet_radius',
    'stellar_teff',
    'stellar_radius',
    'snr'
]

# Optional features that improve model if present
OPTIONAL_FEATURES = [
    'impact_parameter',
    'equilibrium_temp',
    'stellar_insolation',
    'stellar_logg',
    'disposition_score',
    'fp_flag_not_transit',
    'fp_flag_stellar_eclipse',
    'fp_flag_centroid_offset',
    'fp_flag_ephemeris_match'
]

# Target columns to check (in priority order)
TARGET_COLUMNS = ['koi_disposition', 'koi_pdisposition', 'disposition', 'tfopwg_disp']

# Disposition normalization
DISPOSITION_MAP = {
    'CONFIRMED': 'CONFIRMED',
    'CANDIDATE': 'CANDIDATE', 
    'FALSE POSITIVE': 'FALSE POSITIVE',
    'FALSE_POSITIVE': 'FALSE POSITIVE',
    'FP': 'FALSE POSITIVE',
    # TOI/TESS dispositions
    'CP': 'CONFIRMED',    # Confirmed Planet
    'PC': 'CANDIDATE',    # Planet Candidate  
    'APC': 'CANDIDATE',   # Ambiguous Planet Candidate
    'KP': 'CANDIDATE',    # Known Planet
    'FA': 'FALSE POSITIVE' # False Alarm
}

def normalize_disposition(val):
    if pd.isna(val):
        return np.nan
    val_upper = str(val).strip().upper()
    return DISPOSITION_MAP.get(val_upper, val_upper)

def _robust_read_csv(filepath: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Robust CSV reader for Kepler/NASA archive dumps containing leading
    comment lines starting with '#'. Tries:
      1. comment-aware pandas read (comment='#')
      2. manual comment stripping + column count normalization
      3. legacy python engine with on_bad_lines='skip'
      4. manual line count filter (existing fallback)
    Returns (DataFrame, stats).
    """
    stats: Dict[str, Any] = {
        'method': '',
        'skipped_lines': 0,
        'removed_lines': 0,
        'original_line_count': 0
    }

    # Read all lines upfront (file sizes are manageable for catalog extracts)
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        raw_lines = f.readlines()
    stats['original_line_count'] = len(raw_lines)

    # Helper: detect if frame is degenerate single-column despite commas in data
    def _degenerate(df: pd.DataFrame) -> bool:
        if df.shape[1] == 1:
            # If any non-comment raw line (after stripping) contains a comma, it's degenerate
            return any(',' in ln and not ln.lstrip().startswith('#') for ln in raw_lines)
        return False

    # Pass 1: comment-aware direct read
    try:
        df1 = pd.read_csv(filepath, comment='#', skip_blank_lines=True)
        if not _degenerate(df1):
            stats['method'] = 'comment_read_csv'
            return df1, stats
    except Exception:
        pass  # proceed to next strategy

    # Pass 2: manual strip & rebuild
    try:
        data_lines = [ln.rstrip('\n') for ln in raw_lines
                      if ln.strip() and not ln.lstrip().startswith('#')]
        if not data_lines:
            raise ValueError("No data lines after removing comments.")
        header = data_lines[0]
        expected_cols = header.count(',') + 1
        cleaned = [header]
        removed = 0
        for line in data_lines[1:]:
            # Keep only rows with matching column count (basic integrity filter)
            if line.count(',') + 1 == expected_cols:
                cleaned.append(line)
            else:
                removed += 1
        buffer = io.StringIO("\n".join(cleaned))
        df2 = pd.read_csv(buffer)
        if not _degenerate(df2):
            stats['method'] = 'manual_comment_filter'
            stats['removed_lines'] = removed
            return df2, stats
    except Exception:
        pass  # continue

    # Pass 3: python engine with skip (legacy approach)
    try:
        df3 = pd.read_csv(filepath, engine='python', on_bad_lines='skip')
        if not _degenerate(df3):
            stats['method'] = 'python_on_bad_lines_skip'
            return df3, stats
    except Exception:
        pass

    # Pass 4: existing manual line filtering (tolerant but strict on column count)
    try:
        # Reuse manual rebuild but do not discard mismatched lines silently besides count
        data_lines = [ln.rstrip('\n') for ln in raw_lines
                      if ln.strip() and not ln.lstrip().startswith('#')]
        if not data_lines:
            raise ValueError("No data lines available for manual filtering.")
        header = data_lines[0]
        expected_cols = header.count(',') + 1
        cleaned = [header]
        removed = 0
        for line in data_lines[1:]:
            if line.count(',') + 1 == expected_cols:
                cleaned.append(line)
            else:
                removed += 1
        stats['removed_lines'] = removed
        buffer2 = io.StringIO("\n".join(cleaned))
        df4 = pd.read_csv(buffer2)
        stats['method'] = 'manual_line_filter'
        return df4, stats
    except Exception as e:
        raise RuntimeError(f"Failed all parsing strategies for {filepath}") from e

def load_kepler_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load and preprocess data (Kepler / K2 / Exoplanet Archive) with robust parsing."""
    print(f"Loading data from {filepath}...")
    try:
        df, parse_stats = _robust_read_csv(filepath)
        print(f"Parse method: {parse_stats['method']}; "
              f"removed_lines={parse_stats.get('removed_lines',0)} / "
              f"original={parse_stats.get('original_line_count',len(df))}")
    except Exception as e:
        raise RuntimeError(f"Robust CSV parsing failed: {e}")

    print(f"Raw data shape: {df.shape}")

    # Build unified feature frame
    feature_df = _extract_multi_format_features(df)

    # Extend target column search (already includes 'disposition')
    target = None
    for col in TARGET_COLUMNS:
        if col in df.columns:
            target = df[col].map(normalize_disposition)
            print(f"Using target column: {col}")
            break
    if target is None:
        raise ValueError(f"No target column found. Looking for: {TARGET_COLUMNS}")

    # Identify available features (union with engineered)
    available_features = [c for c in feature_df.columns if feature_df[c].notna().any()]

    if not available_features:
        raise ValueError("No usable features found after multi-format extraction.")

    # Filter rows with target
    valid_mask = target.notna()
    X = feature_df.loc[valid_mask, available_features].copy()
    y = target.loc[valid_mask].copy()

    # Drop rows that are entirely NaN
    row_valid = X.notna().any(axis=1)
    X = X.loc[row_valid]
    y = y.loc[row_valid]

    print(f"Available (raw+engineered) features ({len(available_features)}): {available_features}")
    print(f"Final dataset shape: {X.shape}")
    print("Target distribution:")
    print(y.value_counts())

    return X, y, available_features

def load_dataset(filepath: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Wrapper used by main(): returns X, y, numeric_cols, categorical_cols
    (categorical currently empty).
    """
    X, y, feats = load_kepler_data(filepath)
    return X, y, feats, []  # no categorical columns extracted presently

# --- Simple classifier wrapper (concise) ---
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    _XGB = True
except ImportError:
    _XGB = False

class ExoplanetClassifier:
    """
    Complete exoplanet classification pipeline.
    
    Handles data preprocessing, model training, evaluation, and prediction
    for the 3-class exoplanet classification problem.
    """
    
    def __init__(self, random_state: int = 42,
                 confirmed_weight: float = 0.7,
                 false_positive_weight: float = 1.25,
                 candidate_weight: float = 1.0,
                 other_weight: float = 1.1,
                 apply_post_adjust: bool = True,
                 model_type: str = 'xgboost'):
        """Initialize the classifier."""
        self.random_state = random_state
        self.pipeline = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.numeric_cols = None
        self.categorical_cols = None
        self.classes_ = None
        self.is_fitted = False
        self.model_type = model_type  # (was hardcoded before)
    
        # --- New bias adjustment parameters ---
        self.confirmed_weight = confirmed_weight
        self.false_positive_weight = false_positive_weight
        self.candidate_weight = candidate_weight
        self.other_weight = other_weight
        self.apply_post_adjust = apply_post_adjust
        self.class_bias_factors = {}  # filled after label encoder fit

    def build_pipeline(self, numeric_cols: List[str], categorical_cols: List[str] = None):
        """
        Build the preprocessing and modeling pipeline.
        
        Args:
            numeric_cols: List of numeric column names
            categorical_cols: List of categorical column names (optional)
        """
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols or []
        
        # Preprocessing for numeric features
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        transformers = [('num', numeric_transformer, numeric_cols)]
        
        # Add categorical preprocessing if we have categorical columns
        if self.categorical_cols:
            from sklearn.preprocessing import OneHotEncoder
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_transformer, self.categorical_cols))
        
        preprocessor = ColumnTransformer(transformers=transformers)
        
        # Choose classifier based on availability
        if XGBOOST_AVAILABLE:
            classifier = XGBClassifierFixed(
                eval_metric='mlogloss',
                tree_method='hist',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.random_state,
                verbosity=0
            )
        else:
            classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        # Build pipeline
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
        
        return self.pipeline
    
    def _create_pipeline(self):
        """Create the machine learning pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from sklearn.svm import SVC
        
        # Create preprocessing steps
        scaler = StandardScaler()
        
        # Create classifier based on model type
        if self.model_type == 'xgboost':
            classifier = XGBClassifierFixed(
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
        elif self.model_type == 'random_forest':
            classifier = RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            classifier = SVC(
                random_state=42,
                probability=True
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', classifier)
        ])
        
        return pipeline

    def _get_param_distributions(self):
        """Get parameter distributions for hyperparameter tuning."""
        # Get the actual number of classes from the label encoder
        n_classes = len(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else 2
        
        if self.model_type == 'xgboost':
            # Removed objective / num_class (handled dynamically in wrapper)
            base_params = {
                'classifier__n_estimators': randint(50, 300),
                'classifier__max_depth': randint(3, 10),
                'classifier__learning_rate': uniform(0.01, 0.3),
                'classifier__subsample': uniform(0.6, 0.4),
                'classifier__colsample_bytree': uniform(0.6, 0.4),
                'classifier__min_child_weight': randint(1, 10),
                'classifier__gamma': uniform(0, 0.5),
                'classifier__reg_alpha': uniform(0, 1),
                'classifier__reg_lambda': uniform(1, 2),
            }
        else:
            base_params = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [5, 10, 15, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            }
        
        return base_params

    def train(self, X: pd.DataFrame, y: pd.Series, tune_params: bool = True, 
              cv_folds: int = 3, n_iter: int = 10) -> Dict[str, Any]:
        """
        Train the model with optional hyperparameter tuning.
        
        Args:
            X: Feature DataFrame
            y: Target Series  
            tune_params: Whether to perform hyperparameter tuning
            cv_folds: Number of CV folds
            n_iter: Number of random search iterations
            
        Returns:
            Dictionary with training metrics
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not built. Call build_pipeline() first.")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        # Initialize bias factors after classes known
        self._init_bias_factors()
        # Prepare sample weights (original label form needed)
        sample_weights = self._build_sample_weights(y)
        
        if tune_params and len(X) > 50:  # Only tune if we have enough data
            # Pass sample weights into tuning fit via fit_params
            results = self._train_with_tuning(X, y_encoded, cv_folds, n_iter,
                                              sample_weights=sample_weights)
        else:
            results = self._train_simple(X, y_encoded, sample_weights=sample_weights)
        # Add bias info to results
        results['bias_factors'] = {
            cls: self.class_bias_factors[i] for i, cls in enumerate(self.classes_)
        }
        return results
    
    def _train_simple(self, X: pd.DataFrame, y: np.ndarray, sample_weights=None) -> Dict[str, Any]:
        """Train without hyperparameter tuning."""
        if sample_weights is not None:
            self.pipeline.fit(X, y, classifier__sample_weight=sample_weights)
        else:
            self.pipeline.fit(X, y)
        self.is_fitted = True
        
        # Convert encoded y back to original labels for evaluation
        y_orig = self.label_encoder.inverse_transform(y)
        train_metrics = self.evaluate(X, pd.Series(y_orig))
        
        return {
            'method': 'simple_training',
            'train_metrics': train_metrics
        }
    
    def _train_with_tuning(self, X: pd.DataFrame, y: np.ndarray, 
                          cv_folds: int, n_iter: int, sample_weights=None) -> Dict[str, Any]:
        """Train with hyperparameter tuning."""
        # Ensure label encoder is fitted and has classes
        if not hasattr(self.label_encoder, 'classes_'):
            raise ValueError("Label encoder must be fitted before training")
        
        n_classes = len(self.label_encoder.classes_)
        print(f"Training with {n_classes} classes: {self.label_encoder.classes_}")
        
        # Create the pipeline
        pipeline = self._create_pipeline()
        
        # Get parameter distributions
        param_distributions = self._get_param_distributions()
        
        # Stratified K-Fold cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Random search with cross-validation
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring='f1_macro',  # Good for multiclass problems
            cv=cv,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )
        
        if sample_weights is not None:
            search.fit(X, y, classifier__sample_weight=sample_weights)
        else:
            search.fit(X, y)
        self.pipeline = search.best_estimator_
        self.is_fitted = True
        y_orig = self.label_encoder.inverse_transform(y)
        train_metrics = self.evaluate(X, pd.Series(y_orig))
        
        return {
            'method': 'tuned_training',
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'train_metrics': train_metrics
        }
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        proba = None
        if hasattr(self.pipeline, 'predict_proba'):
            proba = self.pipeline.predict_proba(X)
        else:
            # Try fallback
            raw = self.pipeline.predict(X)
            if isinstance(raw, np.ndarray) and raw.ndim == 2:
                return raw
            raise ValueError("predict_proba not available for this model.")
        if proba.ndim == 1:
            # Binary probability -> expand to two-column (P(class0), P(class1))
            proba = np.vstack([1 - proba, proba]).T
        base = proba if proba.ndim == 2 else np.vstack([1 - proba, proba]).T
        return self._adjust_proba(base)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using adjusted probabilities (bias-aware argmax).
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        proba = self.predict_proba(X)
        preds_idx = np.argmax(proba, axis=1)
        return self.label_encoder.inverse_transform(preds_idx)

    def _build_sample_weights(self, y: pd.Series) -> np.ndarray:
        """
        Build per-sample weights to down-weight CONFIRMED and up-weight FALSE POSITIVE.
        """
        # Default factor map (will only affect known classes)
        factor_map = {
            'CONFIRMED': self.confirmed_weight,
            'FALSE POSITIVE': self.false_positive_weight,
            'CANDIDATE': self.candidate_weight
        }
        # Any other class gets other_weight
        weights = y.map(lambda v: factor_map.get(v, self.other_weight)).astype(float)
        return weights.values

    def _init_bias_factors(self):
        """
        Create multiplicative probability adjustment factors aligning with sample weighting.
        """
        self.class_bias_factors = {}
        for i, cls in enumerate(self.classes_):
            if cls == 'CONFIRMED':
                self.class_bias_factors[i] = self.confirmed_weight
            elif cls == 'FALSE POSITIVE':
                self.class_bias_factors[i] = self.false_positive_weight
            elif cls == 'CANDIDATE':
                self.class_bias_factors[i] = self.candidate_weight
            else:
                self.class_bias_factors[i] = self.other_weight

    def _adjust_proba(self, proba: np.ndarray) -> np.ndarray:
        """
        Apply post-hoc probability scaling (soft cost-sensitive decision).
        """
        if not self.apply_post_adjust or not self.class_bias_factors:
            return proba
        scale = np.array([self.class_bias_factors[i] for i in range(proba.shape[1])], dtype=float)
        adj = proba * scale
        row_sums = adj.sum(axis=1, keepdims=True)
        # Protect against division by zero
        row_sums[row_sums == 0] = 1.0
        return adj / row_sums

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Evaluation uses adjusted probabilities to reflect actual decision policy.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        y_true_enc = self.label_encoder.transform(y)
        proba = self.predict_proba(X)  # already adjusted
        preds_idx = np.argmax(proba, axis=1)
        y_pred = self.label_encoder.inverse_transform(preds_idx)
        
        # Classification report
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true_enc, preds_idx)
        
        # Calculate ROC-AUC and AP based on number of classes
        n_classes = len(self.label_encoder.classes_)
        
        if n_classes == 2:
            # Binary classification
            try:
                roc_auc = roc_auc_score(y_true_enc, proba[:, 1])
                ap = average_precision_score(y_true_enc, proba[:, 1])
            except Exception as e:
                print(f"Warning: Could not calculate binary metrics: {e}")
                roc_auc = float('nan')
                ap = float('nan')
        else:
            # Multi-class
            try:
                roc_auc = roc_auc_score(
                    y_true_enc, proba,
                    multi_class='ovr',
                    average='macro'
                )
            except Exception as e:
                print(f"Warning: Could not calculate ROC-AUC: {e}")
                roc_auc = float('nan')
            
            try:
                # Average precision per class
                ap_scores = []
                for i in range(n_classes):
                    y_bin = (y_true_enc == i).astype(int)
                    ap_scores.append(average_precision_score(y_bin, proba[:, i]))
                ap = np.mean(ap_scores)
            except Exception as e:
                print(f"Warning: Could not calculate AP: {e}")
                ap = float('nan')
        
        return {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'classes': self.label_encoder.classes_.tolist(),
            'accuracy': report.get('accuracy', 0),
            'macro_f1': report.get('macro avg', {}).get('f1-score', 0),
            'weighted_f1': report.get('weighted avg', {}).get('f1-score', 0),
            'roc_auc_ovr': roc_auc,
            'macro_average_precision': ap
        }

    def _print_evaluation(self, metrics: Dict[str, Any]):
        """Print evaluation results in a readable format."""
        print("\n" + "="*60)
        print(" MODEL EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nOverall Performance:")
        print(f"  Macro F1-Score:      {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1-Score:   {metrics['weighted_f1']:.4f}")
        print(f"  ROC AUC (OvR):       {metrics['roc_auc_ovr']:.4f}")
        print(f"  Avg Precision:       {metrics['macro_average_precision']:.4f}")
        
        print(f"\nPer-Class Performance:")
        report = metrics['classification_report']
        for class_name in self.classes_:
            if class_name in report:
                precision = report[class_name]['precision']
                recall = report[class_name]['recall']
                f1 = report[class_name]['f1-score']
                support = report[class_name]['support']
                print(f"  {class_name:15} - P: {precision:.3f}, R: {recall:.3f}, "
                      f"F1: {f1:.3f}, N: {support}")
        
        print(f"\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print("    Predicted:")
        print("    " + "  ".join([f"{cls[:8]:>8}" for cls in self.classes_]))
        for i, true_class in enumerate(self.classes_):
            row = "  ".join([f"{cm[i,j]:8d}" for j in range(len(self.classes_))])
            print(f"{true_class[:8]:8} {row}")
    
    def save_model(self, filepath: str, metadata: Dict[str, Any] = None):
        """Save the trained model and metadata."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_data = {
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'classes_': self.classes_,
            'numeric_cols': self.numeric_cols,
            'categorical_cols': self.categorical_cols,
            'random_state': self.random_state,
            'bias_factors': self.class_bias_factors,
            'apply_post_adjust': self.apply_post_adjust
        }
        
        joblib.dump(model_data, filepath)
        
        # Save metadata
        if metadata:
            metadata_path = str(Path(filepath).with_suffix('.json'))
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model."""
        model_data = joblib.load(filepath)
        
        self.pipeline = model_data['pipeline']
        self.label_encoder = model_data['label_encoder']
        self.classes_ = model_data['classes_']
        self.numeric_cols = model_data['numeric_cols']
        self.categorical_cols = model_data['categorical_cols']
        self.random_state = model_data.get('random_state', 42)
        self.is_fitted = True
        
        print(f"✅ Model loaded from {filepath}")

    @classmethod
    def load(cls, model_path: str, metadata_path: Optional[str] = None):
        """
        Classmethod loader to match external calls:
        ExoplanetClassifier.load(model_path, metadata_path)
        metadata_path is accepted for interface compatibility but not used here.
        """
        model_data = joblib.load(model_path)
        obj = cls(random_state=model_data.get('random_state', 42))
        obj.pipeline = model_data['pipeline']
        obj.label_encoder = model_data['label_encoder']
        obj.classes_ = model_data['classes_']
        obj.numeric_cols = model_data['numeric_cols']
        obj.categorical_cols = model_data['categorical_cols']
        obj.is_fitted = True
        # Restore bias factors
        obj.class_bias_factors = model_data.get('bias_factors', {})
        obj.apply_post_adjust = model_data.get('apply_post_adjust', True)
        return obj

def load_multiple_datasets(file_paths: List[str]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Load and merge multiple CSV files with different formats.
    
    Args:
        file_paths: List of paths to CSV files
        
    Returns:
        Tuple of (X, y, available_features) combining all datasets
    """
    print(f"Loading {len(file_paths)} datasets for multi-file training...")
    
    all_X_list = []
    all_y_list = []
    all_features = set()
    dataset_info = []
    
    for i, filepath in enumerate(file_paths):
        print(f"\nProcessing dataset {i+1}/{len(file_paths)}: {filepath}")
        
        try:
            X, y, features = load_kepler_data(filepath)
            
            # Track dataset information
            dataset_info.append({
                'file': filepath,
                'samples': len(X),
                'features': len(features),
                'classes': y.value_counts().to_dict()
            })
            
            # Add source identifier to track which dataset each sample came from
            X['_source_dataset'] = i
            
            all_X_list.append(X)
            all_y_list.append(y)
            all_features.update(features)
            
            print(f"  - Loaded {len(X)} samples with {len(features)} features")
            print(f"  - Classes: {y.value_counts().to_dict()}")
            
        except Exception as e:
            print(f"  - Warning: Failed to load {filepath}: {e}")
            continue
    
    if not all_X_list:
        raise ValueError("No datasets were successfully loaded")
    
    print(f"\nCombining {len(all_X_list)} datasets...")
    
    # Get common features across all datasets (excluding source identifier)
    common_features = list(all_features - {'_source_dataset'})
    
    # Combine datasets
    combined_X_list = []
    combined_y_list = []
    
    for X, y in zip(all_X_list, all_y_list):
        # Ensure all datasets have the same features (fill missing with NaN)
        X_aligned = X.reindex(columns=common_features + ['_source_dataset'], fill_value=np.nan)
        combined_X_list.append(X_aligned)
        combined_y_list.append(y)
    
    # Concatenate all datasets
    X_combined = pd.concat(combined_X_list, ignore_index=True)
    y_combined = pd.concat(combined_y_list, ignore_index=True)
    
    print(f"\nCombined dataset summary:")
    print(f"  - Total samples: {len(X_combined)}")
    print(f"  - Total features: {len(common_features)}")
    print(f"  - Classes: {y_combined.value_counts().to_dict()}")
    
    # Print per-dataset contribution
    print(f"\nPer-dataset contribution:")
    for i, info in enumerate(dataset_info):
        source_mask = X_combined['_source_dataset'] == i
        print(f"  Dataset {i+1}: {info['samples']} samples ({source_mask.sum()} in combined)")
    
    # Remove source identifier from features for training
    X_final = X_combined.drop(columns=['_source_dataset'])
    
    return X_final, y_combined, common_features

def train_from_multiple_csv(data_paths: List[str],
                           output_model_path: str,
                           model_type: str = 'xgboost',
                           confirmed_weight: float = 0.7,
                           false_positive_weight: float = 1.25,
                           candidate_weight: float = 1.0,
                           other_weight: float = 1.1,
                           apply_post_adjust: bool = True,
                           tune: bool = False,
                           cv_folds: int = 5,
                           n_iter: int = 20,
                           random_state: int = 42) -> Dict[str, Any]:
    """
    Train exoplanet classifier on multiple CSV files with different formats.
    
    Args:
        data_paths: List of paths to CSV files (can be different formats)
        output_model_path: Path to save the trained model
        model_type: Type of model ('xgboost', 'random_forest')
        confirmed_weight: Sample weight for CONFIRMED class
        false_positive_weight: Sample weight for FALSE POSITIVE class  
        candidate_weight: Sample weight for CANDIDATE class
        other_weight: Sample weight for other classes
        apply_post_adjust: Whether to apply post-hoc probability adjustment
        tune: Whether to perform hyperparameter tuning
        cv_folds: Number of CV folds for tuning
        n_iter: Number of random search iterations
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with training results and metadata
    """
    print("🔭 Multi-Dataset Exoplanet Classification Training")
    print("=" * 60)
    
    # Load and combine multiple datasets
    X, y, features = load_multiple_datasets(data_paths)
    
    # Initialize classifier
    clf = ExoplanetClassifier(
        random_state=random_state,
        confirmed_weight=confirmed_weight,
        false_positive_weight=false_positive_weight,
        candidate_weight=candidate_weight,
        other_weight=other_weight,
        apply_post_adjust=apply_post_adjust,
        model_type=model_type
    )
    
    # Build pipeline
    clf.build_pipeline(features, [])  # No categorical features for now
    
    # Train model
    print(f"\nTraining {model_type} model on combined dataset...")
    training_results = clf.train(X, y,
                                 tune_params=tune,
                                 cv_folds=cv_folds,
                                 n_iter=n_iter)
    
    # Evaluate model
    print("\nEvaluating model on combined dataset...")
    metrics = clf.evaluate(X, y)
    clf._print_evaluation(metrics)
    
    # Create metadata
    metadata = {
        'training_results': training_results,
        'eval_metrics': metrics,
        'data_paths': data_paths,
        'n_datasets': len(data_paths),
        'n_samples': len(X),
        'n_features': len(features),
        'features': features,
        'classes': y.value_counts().to_dict(),
        'model_type': model_type,
        'weights': {
            'confirmed': confirmed_weight,
            'false_positive': false_positive_weight,
            'candidate': candidate_weight,
            'other': other_weight
        },
        'training_config': {
            'tune_params': tune,
            'cv_folds': cv_folds,
            'n_iter': n_iter,
            'random_state': random_state
        }
    }
    
    # Save model
    print(f"\nSaving model to {output_model_path}...")
    clf.save_model(output_model_path, metadata)
    
    print("\n🎉 Multi-dataset training completed successfully!")
    
    return {
        'model_path': output_model_path,
        'metadata': metadata,
        'features': features,
        'classifier': clf
    }

def train_from_csv(data_path: str,
                   output_model_path: str,
                   model_type: str = 'xgboost',
                   confirmed_weight: float = 0.7,
                   false_positive_weight: float = 1.25,
                   candidate_weight: float = 1.0,
                   other_weight: float = 1.1,
                   apply_post_adjust: bool = True,
                   tune: bool = False,
                   cv_folds: int = 5,
                   n_iter: int = 20,
                   random_state: int = 42) -> Dict[str, Any]:
    """
    Convenience training wrapper for single CSV file training.
    Returns training metadata dict.
    """
    X, y, numeric_cols, categorical_cols = load_dataset(data_path)
    clf = ExoplanetClassifier(
        random_state=random_state,
        confirmed_weight=confirmed_weight,
        false_positive_weight=false_positive_weight,
        candidate_weight=candidate_weight,
        other_weight=other_weight,
        apply_post_adjust=apply_post_adjust,
        model_type=model_type
    )
    clf.build_pipeline(numeric_cols, categorical_cols)
    training_results = clf.train(X, y,
                                 tune_params=tune,
                                 cv_folds=cv_folds,
                                 n_iter=n_iter)
    metrics = clf.evaluate(X, y)
    metadata = {
        'training_results': training_results,
        'eval_metrics': metrics,
        'data_path': data_path,
        'n_samples': len(X),
        'n_features': len(X.columns),
        'classes': y.value_counts().to_dict(),
        'model_type': model_type,
        'bias_factors': clf.class_bias_factors if hasattr(clf, 'class_bias_factors') else {}
    }
    clf.save_model(output_model_path, metadata)
    return {
        'model_path': output_model_path,
        'metadata': metadata,
        'features': numeric_cols,
        'classifier': clf
    }

def main():
    """Command-line interface for model training."""
    parser = argparse.ArgumentParser(description='Train Exoplanet Classifier')
    parser.add_argument('--data', type=str, nargs='+', required=True,
                       help='Path(s) to training data CSV file(s). Can specify multiple files.')
    parser.add_argument('--output', type=str, default='models/exoplanet_classifier.joblib',
                       help='Output path for saved model')
    parser.add_argument('--tune', action='store_true',
                       help='Enable hyperparameter tuning')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of CV folds for tuning')
    parser.add_argument('--n-iter', type=int, default=20,
                       help='Number of random search iterations')
    parser.add_argument('--model-type', type=str, default='xgboost',
                       choices=['xgboost', 'random_forest'],
                       help='Type of model to train')
    
    args = parser.parse_args()
    
    print("🔭 Exoplanet Classification Model Training")
    print("=" * 50)
    
    # Determine if single or multi-file training
    if len(args.data) == 1:
        print(f"Single dataset training mode")
        data_path = args.data[0]
        
        # Load data
        print(f"Loading data from {data_path}...")
        try:
            X, y, numeric_cols, categorical_cols = load_dataset(data_path)
            print(f"✅ Loaded {len(X)} samples with {len(X.columns)} features")
            print(f"Classes: {y.value_counts().to_dict()}")
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return 1
        
        # Initialize and train classifier
        print("\nInitializing classifier...")
        classifier = ExoplanetClassifier(model_type=args.model_type)
        classifier.build_pipeline(numeric_cols, categorical_cols)
        
        print("Starting training...")
        training_results = classifier.train(
            X, y, 
            tune_params=args.tune,
            cv_folds=args.cv_folds,
            n_iter=args.n_iter
        )
        
        # Evaluate model
        print("\nEvaluating model...")
        metrics = classifier.evaluate(X, y)
        
        # Save model
        print(f"\nSaving model to {args.output}...")
        metadata = {
            'training_results': training_results,
            'data_path': data_path,
            'n_samples': len(X),
            'n_features': len(X.columns),
            'classes': y.value_counts().to_dict(),
            'model_type': args.model_type
        }
        classifier.save_model(args.output, metadata)
        
    else:
        print(f"Multi-dataset training mode with {len(args.data)} files")
        
        # Use multi-file training
        try:
            result = train_from_multiple_csv(
                data_paths=args.data,
                output_model_path=args.output,
                model_type=args.model_type,
                tune=args.tune,
                cv_folds=args.cv_folds,
                n_iter=args.n_iter
            )
            print(f"✅ Multi-dataset training completed")
            
        except Exception as e:
            print(f"❌ Multi-dataset training failed: {e}")
            return 1
    
    print("\n🎉 Training completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())
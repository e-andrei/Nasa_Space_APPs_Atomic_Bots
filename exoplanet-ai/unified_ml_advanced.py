#!/usr/bin/env python3
"""
Model unificat îmbunătățit pentru analiza și tuning pe toate cele 3 seturi de date simultan
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model import train_from_multiple_csv, ExoplanetClassifier, load_multiple_datasets

def unified_ml_analysis_improved():
    print("🌌 ANALIZĂ UNIFICATĂ ML - Model care înțelege toate formatele")
    print("=" * 80)
    
    # Toate seturile de date
    datasets = {
        'Kepler': "date/cumulative_2025.10.03_23.13.19.csv",
        'K2': "date/k2pandc_2025.10.03_23.45.46.csv", 
        'TESS_TOI': "date/TOI_2025.10.04_00.04.19.csv"
    }
    
    # Verifică ce seturi sunt disponibile
    available_datasets = {}
    for name, path in datasets.items():
        if Path(path).exists():
            available_datasets[name] = path
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} - nu există")
    
    if len(available_datasets) < 2:
        print("❌ Sunt necesare cel puțin 2 seturi de date!")
        return
    
    print(f"\n🚀 Antrenarea modelului unificat pe {len(available_datasets)} seturi...")
    
    # Antrenează modelul pe toate datele simultan
    all_paths = list(available_datasets.values())
    
    # Prima variantă: Random Forest cu tuning
    print("\n📊 Testez Random Forest cu tuning...")
    rf_result = train_from_multiple_csv(
        data_paths=all_paths,
        output_model_path="models/unified_rf_tuned.joblib",
        model_type='random_forest',
        tune=True,
        cv_folds=3,
        n_iter=20,
        random_state=42
    )
    
    print(f"✅ Random Forest Results:")
    print(f"   Accuracy: {rf_result['metadata']['eval_metrics']['accuracy']:.3f}")
    print(f"   Macro F1: {rf_result['metadata']['eval_metrics']['macro_f1']:.3f}")
    
    # A doua variantă: XGBoost cu tuning
    print("\n📊 Testez XGBoost cu tuning...")
    try:
        xgb_result = train_from_multiple_csv(
            data_paths=all_paths,
            output_model_path="models/unified_xgb_tuned.joblib",
            model_type='xgboost',
            tune=True,
            cv_folds=3,
            n_iter=20,
            random_state=42
        )
        
        print(f"✅ XGBoost Results:")
        print(f"   Accuracy: {xgb_result['metadata']['eval_metrics']['accuracy']:.3f}")
        print(f"   Macro F1: {xgb_result['metadata']['eval_metrics']['macro_f1']:.3f}")
        
        # Compară rezultatele
        if xgb_result['metadata']['eval_metrics']['macro_f1'] > rf_result['metadata']['eval_metrics']['macro_f1']:
            best_model_path = "models/unified_xgb_tuned.joblib"
            best_result = xgb_result
            best_name = "XGBoost"
        else:
            best_model_path = "models/unified_rf_tuned.joblib"
            best_result = rf_result
            best_name = "Random Forest"
            
    except Exception as e:
        print(f"   ⚠️  XGBoost nu a funcționat: {str(e)}")
        best_model_path = "models/unified_rf_tuned.joblib"
        best_result = rf_result
        best_name = "Random Forest"
    
    print(f"\n🏆 Cel mai bun model: {best_name}")
    print(f"   Path: {best_model_path}")
    
    # Încarcă cel mai bun model
    clf = ExoplanetClassifier.load(best_model_path)
    
    # PARTEA 2: Analiză folosind datele combinate originale
    print(f"\n" + "=" * 80)
    print("ANALIZĂ DETALIATĂ FOLOSIND STRUCTURA UNIFICATĂ")
    print("=" * 80)
    
    # Folosește funcția care combină datele corect
    X_combined, y_combined, feature_names = load_multiple_datasets(all_paths)
    
    print(f"📊 Date combinate încărcate cu succes:")
    print(f"   Total obiecte: {len(X_combined):,}")
    print(f"   Caracteristici comune: {len(feature_names)}")
    print(f"   Caracteristici: {feature_names}")
    
    # Fă predicții pe datele combinate
    predictions = clf.predict(X_combined)
    probabilities = clf.predict_proba(X_combined)
    
    # Analizează rezultatele
    real_counts = y_combined.value_counts().to_dict()
    pred_counts = pd.Series(predictions).value_counts().to_dict()
    
    print(f"\n🔍 COMPARAȚIA FINALĂ: REAL vs PREZIS")
    print(f"{'Categorie':<15} {'Real':<10} {'Prezis':<10} {'Diferența':<10} {'Diferența %':<12} {'Accuracy':<10}")
    print("-" * 75)
    
    total_objects = len(y_combined)
    overall_correct = 0
    
    for category in sorted(real_counts.keys()):
        real = real_counts.get(category, 0)
        pred = pred_counts.get(category, 0)
        diff = pred - real
        diff_pct = (diff / real * 100) if real > 0 else 0
        
        # Calculează accuracy pentru această categorie
        category_correct = ((y_combined == category) & (predictions == category)).sum()
        category_total = (y_combined == category).sum()
        category_accuracy = category_correct / category_total if category_total > 0 else 0
        overall_correct += category_correct
        
        print(f"{category:<15} {real:<10,} {pred:<10,} {diff:<+10,} {diff_pct:<+11.1f}% {category_accuracy:<9.3f}")
    
    overall_accuracy = overall_correct / total_objects
    print(f"\n✅ ACCURACY GLOBALĂ: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
    
    # Analiză detaliată per dataset
    print(f"\n" + "=" * 80)
    print("ANALIZĂ PER DATASET INDIVIDUAL")
    print("=" * 80)
    
    # Separă rezultatele per dataset
    dataset_info = separate_results_by_dataset(y_combined, predictions, all_paths)
    
    for dataset_name, info in dataset_info.items():
        print(f"\n📊 {dataset_name}:")
        print(f"   Obiecte: {info['total']:,}")
        print(f"   Accuracy: {info['accuracy']:.3f} ({info['accuracy']*100:.1f}%)")
        print(f"   Distribuție reală: {info['real_dist']}")
        print(f"   Distribuție prezisă: {info['pred_dist']}")
        
        # Analizează bias-ul pentru acest dataset
        for category in info['real_dist'].keys():
            real = info['real_dist'][category]
            pred = info['pred_dist'].get(category, 0)
            bias = (pred - real) / real if real > 0 else 0
            if abs(bias) > 0.1:  # Bias mai mare de 10%
                bias_type = "supraevaluează" if bias > 0 else "subevaluează"
                print(f"   ⚠️  {bias_type} {category} cu {abs(bias)*100:.1f}%")
    
    # PARTEA 3: Recomandări de tuning
    print(f"\n" + "=" * 80)
    print("RECOMANDĂRI PENTRU TUNING ȘI OPTIMIZARE")
    print("=" * 80)
    
    # Calculează bias-ul global
    confirmed_real = real_counts.get('CONFIRMED', 0)
    confirmed_pred = pred_counts.get('CONFIRMED', 0)
    confirmed_bias = (confirmed_pred - confirmed_real) / confirmed_real if confirmed_real > 0 else 0
    
    candidate_real = real_counts.get('CANDIDATE', 0)
    candidate_pred = pred_counts.get('CANDIDATE', 0)
    candidate_bias = (candidate_pred - candidate_real) / candidate_real if candidate_real > 0 else 0
    
    fp_real = real_counts.get('FALSE POSITIVE', 0)
    fp_pred = pred_counts.get('FALSE POSITIVE', 0)
    fp_bias = (fp_pred - fp_real) / fp_real if fp_real > 0 else 0
    
    print(f"📈 ANALIZĂ BIAS GLOBAL:")
    print(f"   CONFIRMED bias: {confirmed_bias*100:+.1f}%")
    print(f"   CANDIDATE bias: {candidate_bias*100:+.1f}%")
    print(f"   FALSE POSITIVE bias: {fp_bias*100:+.1f}%")
    
    print(f"\n🎯 RECOMANDĂRI SPECIFICE:")
    
    if abs(confirmed_bias) > 0.05:
        if confirmed_bias > 0:
            print(f"   • Modelul supraevaluează planetele CONFIRMATE")
            print(f"     → Crește threshold-ul sau scade confirmed_weight la {0.7 - abs(confirmed_bias):.2f}")
        else:
            print(f"   • Modelul subevaluează planetele CONFIRMATE")
            print(f"     → Scade threshold-ul sau crește confirmed_weight la {0.7 + abs(confirmed_bias):.2f}")
    
    if abs(candidate_bias) > 0.05:
        if candidate_bias > 0:
            print(f"   • Modelul supraevaluează CANDIDAȚII")
            print(f"     → Ajustează candidate_weight la {1.0 - abs(candidate_bias):.2f}")
        else:
            print(f"   • Modelul subevaluează CANDIDAȚII")
            print(f"     → Ajustează candidate_weight la {1.0 + abs(candidate_bias):.2f}")
    
    if abs(fp_bias) > 0.05:
        if fp_bias > 0:
            print(f"   • Modelul supraevaluează FALSE POSITIVE")
            print(f"     → Îmbunătățește detectarea caracteristicilor planetare")
        else:
            print(f"   • Modelul subevaluează FALSE POSITIVE")
            print(f"     → Verifică dacă nu ratează false positive")
    
    # Comandă pentru retraining optimizat
    print(f"\n🛠️  COMANDĂ PENTRU RETRAINING OPTIMIZAT:")
    optimized_confirmed = max(0.3, min(1.5, 0.7 - confirmed_bias))
    optimized_candidate = max(0.3, min(1.5, 1.0 - candidate_bias))
    optimized_fp = max(0.3, min(2.0, 1.25 - fp_bias))
    
    print(f"python -c \"")
    print(f"from model import train_from_multiple_csv")
    print(f"train_from_multiple_csv(")
    print(f"    data_paths={all_paths},")
    print(f"    output_model_path='models/optimized_unified.joblib',")
    print(f"    model_type='{best_name.lower().replace(' ', '_')}',")
    print(f"    confirmed_weight={optimized_confirmed:.2f},")
    print(f"    candidate_weight={optimized_candidate:.2f},")
    print(f"    false_positive_weight={optimized_fp:.2f},")
    print(f"    tune=True,")
    print(f"    n_iter=50)\"")
    
    print(f"\n🎉 ANALIZĂ UNIFICATĂ COMPLETĂ!")
    print(f"   ✅ Model antrenat pe toate cele 3 formate simultan")
    print(f"   ✅ Analiză comparativă real vs prezis")
    print(f"   ✅ Recomandări de tuning generate automat")
    print(f"   ✅ Model salvat: {best_model_path}")

def load_multiple_datasets(file_paths: List[str]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Încarcă și combină multiple seturi de date, returnând și informații despre surse"""
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from model import load_multiple_datasets as load_multi
    
    X, y, features = load_multi(file_paths)
    return X, y, features

def separate_results_by_dataset(y_combined, predictions, file_paths):
    """Separă rezultatele pe dataset-uri individuale bazat pe dimensiuni"""
    from model import load_dataset
    
    results = {}
    start_idx = 0
    
    for i, file_path in enumerate(file_paths):
        dataset_name = Path(file_path).stem
        
        # Încarcă dataset-ul pentru a afla dimensiunea
        try:
            X_temp, y_temp, _, _ = load_dataset(file_path)
            dataset_size = len(y_temp)
            
            # Extrage porțiunea corespunzătoare din rezultate
            end_idx = start_idx + dataset_size
            y_dataset = y_combined.iloc[start_idx:end_idx]
            pred_dataset = predictions[start_idx:end_idx]
            
            # Calculează statistici
            real_dist = y_dataset.value_counts().to_dict()
            pred_dist = pd.Series(pred_dataset).value_counts().to_dict()
            accuracy = (y_dataset == pred_dataset).sum() / len(y_dataset)
            
            results[dataset_name] = {
                'total': dataset_size,
                'accuracy': accuracy,
                'real_dist': real_dist,
                'pred_dist': pred_dist
            }
            
            start_idx = end_idx
            
        except Exception as e:
            print(f"   ⚠️  Nu s-a putut analiza {dataset_name}: {str(e)}")
    
    return results

if __name__ == "__main__":
    unified_ml_analysis_improved()
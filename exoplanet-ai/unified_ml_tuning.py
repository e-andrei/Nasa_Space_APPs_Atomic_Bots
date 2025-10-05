#!/usr/bin/env python3
"""
Model unificat pentru analiza și tuning pe toate cele 3 seturi de date simultan
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model import train_from_multiple_csv, ExoplanetClassifier, load_dataset

def unified_ml_analysis():
    print("🌌 ANALIZĂ UNIFICATĂ ML - Toate cele 3 seturi de date")
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
    
    # Antrenează modelul pe toate datele
    all_paths = list(available_datasets.values())
    result = train_from_multiple_csv(
        data_paths=all_paths,
        output_model_path="models/unified_multi_dataset.joblib",
        model_type='random_forest',  # Începem cu RF, apoi testăm XGBoost
        tune=True,  # Activăm tuning-ul
        cv_folds=5,
        n_iter=30,  # Mai multe iterații pentru tuning mai bun
        random_state=42
    )
    
    print(f"✅ Model unificat antrenat cu succes!")
    print(f"   Accuracy globală: {result['metadata']['eval_metrics']['accuracy']:.3f}")
    print(f"   Macro F1: {result['metadata']['eval_metrics']['macro_f1']:.3f}")
    
    # Încarcă modelul pentru analiză
    clf = ExoplanetClassifier.load("models/unified_multi_dataset.joblib")
    
    # PARTEA 2: Analiză pe fiecare set individual
    print(f"\n" + "=" * 80)
    print("ANALIZĂ DETALIATĂ PER SET DE DATE")
    print("=" * 80)
    
    total_real = {'CONFIRMED': 0, 'CANDIDATE': 0, 'FALSE POSITIVE': 0}
    total_predicted = {'CONFIRMED': 0, 'CANDIDATE': 0, 'FALSE POSITIVE': 0}
    dataset_results = {}
    
    for dataset_name, file_path in available_datasets.items():
        print(f"\n📊 Analizez {dataset_name}...")
        
        try:
            # Încarcă datele
            X, y, _, _ = load_dataset(file_path)
            
            # Fă predicții
            predictions = clf.predict(X)
            probabilities = clf.predict_proba(X)
            
            # Analizează rezultatele reale
            real_counts = y.value_counts().to_dict()
            pred_counts = pd.Series(predictions).value_counts().to_dict()
            
            print(f"   📈 {dataset_name} - Numărul de obiecte: {len(X):,}")
            print(f"      CLASIFICĂRI REALE:")
            for category, count in real_counts.items():
                percentage = (count / len(y) * 100)
                print(f"        {category}: {count:,} ({percentage:.1f}%)")
                total_real[category] = total_real.get(category, 0) + count
            
            print(f"      PREDICȚII MODEL:")
            for category, count in pred_counts.items():
                percentage = (count / len(predictions) * 100)
                print(f"        {category}: {count:,} ({percentage:.1f}%)")
                total_predicted[category] = total_predicted.get(category, 0) + count
            
            # Calculează acuratețea pe acest set
            accuracy = (y == predictions).sum() / len(y)
            print(f"      ✅ Acuratețea pe {dataset_name}: {accuracy:.3f} ({accuracy*100:.1f}%)")
            
            # Analiză detaliată a erorilor
            errors = y != predictions
            if errors.sum() > 0:
                print(f"      ❌ Erori: {errors.sum():,} din {len(y):,}")
                error_analysis = analyze_prediction_errors(y, predictions, dataset_name)
                print(f"         Tipuri de erori principale:")
                for error_type, count in error_analysis.items():
                    print(f"           {error_type}: {count}")
            
            # Salvează rezultatele pentru raportul final
            dataset_results[dataset_name] = {
                'real_counts': real_counts,
                'pred_counts': pred_counts,
                'accuracy': accuracy,
                'total_objects': len(X),
                'errors': errors.sum()
            }
            
        except Exception as e:
            print(f"   ❌ Eroare la procesarea {dataset_name}: {str(e)}")
    
    # PARTEA 3: Raport consolidat și recomandări de tuning
    print(f"\n" + "=" * 80)
    print("RAPORT CONSOLIDAT ȘI RECOMANDĂRI TUNING")
    print("=" * 80)
    
    # Afișează totalurile
    total_objects = sum(res['total_objects'] for res in dataset_results.values())
    total_errors = sum(res['errors'] for res in dataset_results.values())
    overall_accuracy = (total_objects - total_errors) / total_objects
    
    print(f"\n📊 REZUMAT GLOBAL ({total_objects:,} obiecte în total):")
    print(f"   Acuratețea globală: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
    print(f"   Total erori: {total_errors:,}")
    
    print(f"\n🔍 COMPARAȚIA GLOBAL: REAL vs PREZIIS")
    print(f"{'Categorie':<15} {'Real':<10} {'Prezis':<10} {'Diferența':<10} {'Diferența %':<12}")
    print("-" * 60)
    
    for category in ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']:
        real = total_real.get(category, 0)
        pred = total_predicted.get(category, 0)
        diff = pred - real
        diff_pct = (diff / real * 100) if real > 0 else 0
        print(f"{category:<15} {real:<10,} {pred:<10,} {diff:<+10,} {diff_pct:<+11.1f}%")
    
    # Recomandări de tuning
    print(f"\n🎯 RECOMANDĂRI PENTRU TUNING:")
    
    # Analizează bias-ul modelului
    confirmed_bias = (total_predicted.get('CONFIRMED', 0) - total_real.get('CONFIRMED', 0)) / total_real.get('CONFIRMED', 1)
    candidate_bias = (total_predicted.get('CANDIDATE', 0) - total_real.get('CANDIDATE', 0)) / total_real.get('CANDIDATE', 1)
    fp_bias = (total_predicted.get('FALSE POSITIVE', 0) - total_real.get('FALSE POSITIVE', 0)) / total_real.get('FALSE POSITIVE', 1)
    
    if confirmed_bias > 0.1:
        print(f"   ⚠️  Modelul supraevaluează planetele CONFIRMATE cu {confirmed_bias*100:.1f}%")
        print(f"      → Recomandare: Crește threshold-ul pentru CONFIRMED")
        print(f"      → Sau: Ajustează class_weight pentru 'CONFIRMED' la 0.8-0.9")
    elif confirmed_bias < -0.1:
        print(f"   ⚠️  Modelul subevaluează planetele CONFIRMATE cu {abs(confirmed_bias)*100:.1f}%")
        print(f"      → Recomandare: Scade threshold-ul pentru CONFIRMED")
        print(f"      → Sau: Crește class_weight pentru 'CONFIRMED' la 1.1-1.2")
    
    if candidate_bias > 0.1:
        print(f"   ⚠️  Modelul supraevaluează CANDIDAȚII cu {candidate_bias*100:.1f}%")
        print(f"      → Recomandare: Crește threshold-ul pentru CANDIDATE")
    elif candidate_bias < -0.1:
        print(f"   ⚠️  Modelul subevaluează CANDIDAȚII cu {abs(candidate_bias)*100:.1f}%")
        print(f"      → Recomandare: Scade threshold-ul pentru CANDIDATE")
    
    if fp_bias > 0.1:
        print(f"   ⚠️  Modelul supraevaluează FALSE POSITIVE cu {fp_bias*100:.1f}%")
        print(f"      → Recomandare: Îmbunătățește detectarea caracteristicilor planetare")
    
    # Testează și cu XGBoost pentru comparație
    print(f"\n🔬 TESTARE COMPARATIVĂ CU XGBOOST:")
    try:
        xgb_result = train_from_multiple_csv(
            data_paths=all_paths,
            output_model_path="models/unified_xgboost_comparison.joblib",
            model_type='xgboost',
            tune=True,
            cv_folds=3,  # Mai puține fold-uri pentru viteză
            n_iter=20,
            random_state=42
        )
        
        print(f"   ✅ XGBoost Results:")
        print(f"      Accuracy: {xgb_result['metadata']['eval_metrics']['accuracy']:.3f}")
        print(f"      Macro F1: {xgb_result['metadata']['eval_metrics']['macro_f1']:.3f}")
        
        rf_f1 = result['metadata']['eval_metrics']['macro_f1']
        xgb_f1 = xgb_result['metadata']['eval_metrics']['macro_f1']
        
        if xgb_f1 > rf_f1:
            improvement = (xgb_f1 - rf_f1) / rf_f1 * 100
            print(f"   🎉 XGBoost este superior cu {improvement:.1f}%!")
            print(f"      → Recomandare: Folosește XGBoost pentru model final")
        else:
            improvement = (rf_f1 - xgb_f1) / xgb_f1 * 100
            print(f"   📊 Random Forest rămâne superior cu {improvement:.1f}%")
            print(f"      → Recomandare: Păstrează Random Forest")
            
    except Exception as e:
        print(f"   ⚠️  Nu s-a putut testa XGBoost: {str(e)}")
    
    # Generează comenzi pentru tuning manual
    print(f"\n🛠️  COMENZI PENTRU TUNING MANUAL:")
    print(f"   1. Pentru ajustarea weighturilor:")
    print(f"      confirmed_weight = {0.7 + confirmed_bias:.2f}")
    print(f"      candidate_weight = {1.0 + candidate_bias:.2f}")
    print(f"      false_positive_weight = {1.25 + fp_bias:.2f}")
    
    print(f"\n   2. Pentru re-antrenarea cu parametri optimizați:")
    print(f"      python -c \"from model import train_from_multiple_csv; train_from_multiple_csv(")
    print(f"         data_paths={all_paths},")
    print(f"         output_model_path='models/tuned_unified.joblib',")
    print(f"         confirmed_weight={0.7 + confirmed_bias:.2f},")
    print(f"         candidate_weight={1.0 + candidate_bias:.2f},")
    print(f"         false_positive_weight={1.25 + fp_bias:.2f},")
    print(f"         tune=True, n_iter=50)\"")
    
    print(f"\n🎉 ANALIZĂ COMPLETĂ FINALIZATĂ!")
    print(f"   Modelul unificat poate procesa toate cele 3 formate de date")
    print(f"   Recomandările de tuning sunt generate automat")
    print(f"   Fișierele salvate: unified_multi_dataset.joblib")

def analyze_prediction_errors(y_true, y_pred, dataset_name):
    """Analizează tipurile de erori în predicții"""
    errors = {}
    
    # Creează o matrice de confuzie simplificată
    for true_val, pred_val in zip(y_true, y_pred):
        if true_val != pred_val:
            error_type = f"{true_val} → {pred_val}"
            errors[error_type] = errors.get(error_type, 0) + 1
    
    # Returnează top 3 erori
    sorted_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_errors[:3])

if __name__ == "__main__":
    unified_ml_analysis()
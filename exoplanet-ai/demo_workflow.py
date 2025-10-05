#!/usr/bin/env python3
"""
Quick demonstration of the complete ML workflow including TOI data analysis
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model import train_from_csv, ExoplanetClassifier, load_dataset, train_from_multiple_csv

def demo_workflow():
    print("🚀 Quick ML Workflow Demonstration - Model Unificat")
    print("=" * 50)
    
    # Step 1: Train unified model on all datasets
    print("Step 1: Antrenarea modelului unificat pe toate datele...")
    
    # Lista tuturor seturilor de date
    all_datasets = [
        "date/cumulative_2025.10.03_23.13.19.csv",  # Kepler
        "date/k2pandc_2025.10.03_23.45.46.csv",     # K2
        "date/TOI_2025.10.04_00.04.19.csv"          # TESS TOI
    ]
    
    # Verifică ce seturi sunt disponibile
    available_datasets = [f for f in all_datasets if Path(f).exists()]
    print(f"Seturi de date disponibile: {len(available_datasets)}")
    for dataset in available_datasets:
        print(f"  ✅ {Path(dataset).name}")
    
    if len(available_datasets) >= 2:
        # Antrenează model pe toate datele simultan
        try:
            result = train_from_multiple_csv(
                data_paths=available_datasets,
                output_model_path="models/demo_unified_model.joblib",
                model_type='xgboost',
                tune=False,  # Skip tuning for demo speed
                random_state=42
            )
            
            print(f"✅ Model unificat antrenat cu succes!")
            print(f"   Accuracy: {result['metadata']['eval_metrics']['accuracy']:.3f}")
            print(f"   Macro F1: {result['metadata']['eval_metrics']['macro_f1']:.3f}")
            print(f"   Procesează {len(available_datasets)} formate simultane")
            
            # Încarcă modelul pentru analiză
            clf = ExoplanetClassifier.load("models/demo_unified_model.joblib")
            
        except Exception as e:
            print(f"   ⚠️  Eroare la antrenarea modelului unificat: {str(e)}")
            print("   Revin la modelul simplu...")
            # Fallback la modelul simplu
            result = train_from_csv(
                data_path="date/cumulative_2025.10.03_23.13.19.csv",
                output_model_path="models/demo_model.joblib",
                model_type='random_forest',
                tune=False,
                random_state=42
            )
            clf = ExoplanetClassifier.load("models/demo_model.joblib")
    else:
        print("   ⚠️  Nu sunt suficiente seturi de date pentru model unificat")
        result = train_from_csv(
            data_path="date/cumulative_2025.10.03_23.13.19.csv",
            output_model_path="models/demo_model.joblib",
            model_type='random_forest',
            tune=False,
            random_state=42
        )
        clf = ExoplanetClassifier.load("models/demo_model.joblib")
    
    # Step 2: Test predictions
    print("\nStep 2: Testarea predicțiilor pe date multiple...")
    X_test, y_test, numeric_cols, categorical_cols = load_dataset("date/cumulative_2025.10.03_23.13.19.csv")
    X_test = X_test.head(5)
    y_test = y_test.head(5)
    
    predictions = clf.predict(X_test)
    probabilities = clf.predict_proba(X_test)
    
    print("✅ Predictions completed:")
    for i in range(len(predictions)):
        pred = predictions[i]
        conf = probabilities[i].max()
        actual = y_test.iloc[i]
        print(f"   Sample {i+1}: Predicted={pred}, Confidence={conf:.3f}, Actual={actual}")
    
    # Step 3: Comprehensive analysis of all datasets
    print("\n" + "=" * 60)
    print("Step 3: Analiză comparativă pe toate seturile de date")
    print("=" * 60)
    
    # Analizează toate seturile de date disponibile
    total_analysis = analyze_all_datasets_unified(available_datasets, clf)
    
    # Step 4: Summary and tuning recommendations
    print("\n" + "=" * 50)
    print("Step 4: Rezumat și recomandări de tuning")
    print("=" * 50)
    
    metrics = clf.evaluate(X_test, y_test)
    print(f"✅ Test Accuracy: {metrics['accuracy']:.3f}")
    print(f"   Test Macro F1: {metrics['macro_f1']:.3f}")
    
    if total_analysis:
        print(f"\n📊 ANALIZA GLOBALĂ MULTI-DATASET:")
        print(f"   Total obiecte analizate: {total_analysis['total_objects']:,}")
        print(f"   Accuracy medie: {total_analysis['avg_accuracy']:.3f}")
        
        print(f"\n🎯 RECOMANDĂRI DE ÎMBUNĂTĂȚIRE:")
        for recommendation in total_analysis['recommendations']:
            print(f"   • {recommendation}")
    
    print("\n🎉 Workflow demonstration completed successfully!")
    print("\nFuncționalități disponibile în aplicația web:")
    print("1. Model unificat care procesează Kepler, K2 și TESS TOI")
    print("2. Analiză comparativă real vs prezis pe toate seturile")
    print("3. Recomandări automate de tuning")
    print("4. Suport pentru antrenare pe date multiple simultan")
    print("5. Optimizare automată a parametrilor")
    
    # Step 3: Analyze TOI data (TESS Objects of Interest)
    print("\n" + "=" * 60)
    print("Step 3: Analiza detaliată a datelor TOI (TESS Objects of Interest)")
    print("=" * 60)
    
    try:
        # Load TOI data
        toi_data = pd.read_csv("date/TOI_2025.10.04_00.04.19.csv", comment='#')
        
        print(f"\n📊 Statistici generale pentru datele TOI:")
        print(f"   Numărul total de obiecte TESS: {len(toi_data):,}")
        
        # Analyze actual classifications
        if 'tfopwg_disp' in toi_data.columns:
            classification_counts = toi_data['tfopwg_disp'].value_counts()
            
            print(f"\n🔍 Distribuția actuală a clasificărilor în datele TOI:")
            total = len(toi_data)
            for category, count in classification_counts.items():
                percentage = (count / total * 100)
                description = get_toi_category_description(category)
                print(f"   {category} ({description}): {count:,} ({percentage:.1f}%)")
            
            # Map to standardized categories
            standardized_counts = map_toi_to_standard_categories(toi_data)
            print(f"\n📈 Maparea la categoriile standardizate ale modelului:")
            total_mapped = sum(standardized_counts.values())
            for category, count in standardized_counts.items():
                percentage = (count / total_mapped * 100)
                print(f"   {category}: {count:,} ({percentage:.1f}%)")
            
            print(f"\n💡 Comparația distribuțiilor:")
            print(f"   Datele TOI conțin {standardized_counts.get('CANDIDATE', 0):,} candidați de planete")
            print(f"   Versus {standardized_counts.get('CONFIRMED', 0):,} planete confirmate")
            print(f"   Și {standardized_counts.get('FALSE POSITIVE', 0):,} false pozitive")
            
            # Calculate ratios
            candidates = standardized_counts.get('CANDIDATE', 0)
            confirmed = standardized_counts.get('CONFIRMED', 0)
            if confirmed > 0:
                ratio = candidates / confirmed
                print(f"   Raportul candidați/confirmați: {ratio:.1f}:1")
        
        print(f"\n🎯 Informații despre potențialul de descoperire:")
        print(f"   • TOI-urile sunt obiecte de interes identificate de TESS")
        print(f"   • Multe dintre 'candidați' pot fi planete reale în așteptarea confirmării")
        print(f"   • Modelul poate ajuta la prioritizarea obiectelor pentru urmărire")
        
    except Exception as e:
        print(f"   ❌ Eroare la încărcarea datelor TOI: {str(e)}")
    
    # Step 4: Model evaluation summary
    print("\n" + "=" * 50)
    print("Step 4: Rezumatul evaluării modelului")
    print("=" * 50)
    
    metrics = clf.evaluate(X_test, y_test)
    print(f"✅ Test Macro F1: {metrics['macro_f1']:.3f}")
    print(f"   Test Accuracy: {metrics['accuracy']:.3f}")
    
    print("\n🎉 Workflow demonstration completed successfully!")
    print("\nFuncționalități disponibile în aplicația web:")
    print("1. Încărcarea și testarea cu fișiere CSV multiple")
    print("2. Ajustarea parametrilor modelului și re-antrenarea")
    print("3. Analiza importanței caracteristicilor")
    print("4. Descărcarea rezultatelor și vizualizarea performanței")
    print("5. Suport pentru formate Kepler, K2/PANDC și TESS TOI")

def analyze_all_datasets_unified(datasets, clf):
    """Analizează toate seturile de date cu modelul unificat"""
    try:
        from model import load_dataset
        
        total_objects = 0
        total_correct = 0
        dataset_results = {}
        recommendations = []
        
        print(f"\n📊 Analizez {len(datasets)} seturi de date cu modelul unificat...")
        
        for dataset_path in datasets:
            dataset_name = Path(dataset_path).stem
            
            try:
                # Încarcă datele
                X, y, _, _ = load_dataset(dataset_path)
                
                # Fă predicții (doar pe primele 1000 pentru viteză)
                sample_size = min(1000, len(X))
                X_sample = X.head(sample_size)
                y_sample = y.head(sample_size)
                
                predictions = clf.predict(X_sample)
                
                # Calculează metrici
                accuracy = (y_sample == predictions).sum() / len(y_sample)
                total_objects += sample_size
                total_correct += (y_sample == predictions).sum()
                
                # Analizează distribuțiile
                real_dist = y_sample.value_counts().to_dict()
                pred_dist = pd.Series(predictions).value_counts().to_dict()
                
                dataset_results[dataset_name] = {
                    'accuracy': accuracy,
                    'real_dist': real_dist,
                    'pred_dist': pred_dist,
                    'sample_size': sample_size
                }
                
                print(f"   📈 {dataset_name}: Accuracy = {accuracy:.3f} (pe {sample_size} eșantioane)")
                
                # Verifică bias-uri significative
                for category in real_dist.keys():
                    real_count = real_dist[category]
                    pred_count = pred_dist.get(category, 0)
                    bias = (pred_count - real_count) / real_count if real_count > 0 else 0
                    
                    if abs(bias) > 0.15:  # Bias mai mare de 15%
                        bias_type = "supraevaluează" if bias > 0 else "subevaluează"
                        recommendations.append(f"Pe {dataset_name}: {bias_type} {category} cu {abs(bias)*100:.1f}%")
                
            except Exception as e:
                print(f"   ❌ Eroare la analizarea {dataset_name}: {str(e)}")
        
        avg_accuracy = total_correct / total_objects if total_objects > 0 else 0
        
        # Adaugă recomandări generale
        if avg_accuracy < 0.9:
            recommendations.append("Accuracy globală sub 90% - consideră tuning mai agresiv")
        if len(recommendations) == 0:
            recommendations.append("Modelul funcționează bine pe toate seturile de date")
        
        return {
            'total_objects': total_objects,
            'avg_accuracy': avg_accuracy,
            'dataset_results': dataset_results,
            'recommendations': recommendations
        }
        
    except Exception as e:
        print(f"   ❌ Eroare în analiza unificată: {str(e)}")
        return None

def get_toi_category_description(category):
    """Returnează descrierea completă pentru categoriile TOI"""
    descriptions = {
        'CP': 'Confirmed Planet',
        'FP': 'False Positive', 
        'PC': 'Planet Candidate',
        'KP': 'Known Planet',
        'FA': 'False Alarm',
        'APC': 'Ambiguous Planet Candidate'
    }
    return descriptions.get(category, 'Unknown')

def get_standard_category_mapping():
    """Returnează maparea de la categoriile TOI la categoriile standard"""
    return {
        'CP': 'CONFIRMED',     # Confirmed Planet
        'KP': 'CONFIRMED',     # Known Planet
        'PC': 'CANDIDATE',     # Planet Candidate
        'APC': 'CANDIDATE',    # Ambiguous Planet Candidate
        'FP': 'FALSE POSITIVE', # False Positive
        'FA': 'FALSE POSITIVE'  # False Alarm
    }

def map_toi_to_standard_categories(toi_data):
    """Mapează categoriile TOI la categoriile standardizate și numără"""
    mapping = get_standard_category_mapping()
    
    if 'tfopwg_disp' in toi_data.columns:
        mapped_categories = toi_data['tfopwg_disp'].map(mapping)
        return mapped_categories.value_counts().to_dict()
    else:
        return {}

if __name__ == "__main__":
    demo_workflow()
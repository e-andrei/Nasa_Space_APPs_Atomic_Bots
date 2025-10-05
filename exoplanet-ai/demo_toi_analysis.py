#!/usr/bin/env python3
"""
Demonstrație completă cu analiza TOI și predicții ale modelului
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model import train_from_csv, ExoplanetClassifier, load_dataset, train_from_multiple_csv

def demo_toi_analysis():
    print("🌟 Analiza completă a datelor TOI cu predicții ale modelului")
    print("=" * 70)
    
    # Step 1: Antrenează model pe date multiple (inclusiv TOI)
    print("Step 1: Antrenarea modelului pe date multiple...")
    
    # Lista fișierelor de date disponibile
    data_files = [
        "date/cumulative_2025.10.03_23.13.19.csv",  # Kepler
        "date/k2pandc_2025.10.03_23.45.46.csv",     # K2
        "date/TOI_2025.10.04_00.04.19.csv"          # TESS TOI
    ]
    
    # Verifică care fișiere există
    existing_files = [f for f in data_files if Path(f).exists()]
    print(f"Fișiere disponibile pentru antrenare: {len(existing_files)}")
    for f in existing_files:
        print(f"  - {f}")
    
    if len(existing_files) >= 2:
        # Antrenează pe date multiple
        result = train_from_multiple_csv(
            data_paths=existing_files,
            output_model_path="models/multi_toi_classifier.joblib",
            model_type='random_forest',
            tune=False,
            random_state=42
        )
        
        print(f"✅ Model multi-dataset antrenat:")
        print(f"   Accuracy: {result['metadata']['eval_metrics']['accuracy']:.3f}")
        print(f"   Macro F1: {result['metadata']['eval_metrics']['macro_f1']:.3f}")
        
        # Încarcă modelul antrenat
        clf = ExoplanetClassifier.load("models/multi_toi_classifier.joblib")
        
    else:
        print("⚠️  Antrenez doar pe datele Kepler...")
        result = train_from_csv(
            data_path="date/cumulative_2025.10.03_23.13.19.csv",
            output_model_path="models/toi_demo_classifier.joblib",
            model_type='random_forest',
            tune=False,
            random_state=42
        )
        clf = ExoplanetClassifier.load("models/toi_demo_classifier.joblib")
    
    # Step 2: Analiza detaliată a datelor TOI
    print("\n" + "=" * 70)
    print("Step 2: Analiza detaliată a datelor TOI")
    print("=" * 70)
    
    try:
        # Încarcă datele TOI
        toi_data = pd.read_csv("date/TOI_2025.10.04_00.04.19.csv", comment='#')
        
        print(f"\n📊 Statistici generale pentru datele TOI:")
        print(f"   Numărul total de obiecte TESS: {len(toi_data):,}")
        
        # Analizează clasificările actuale
        if 'tfopwg_disp' in toi_data.columns:
            classification_counts = toi_data['tfopwg_disp'].value_counts()
            
            print(f"\n🔍 Distribuția actuală a clasificărilor TOI:")
            total = len(toi_data)
            for category, count in classification_counts.items():
                percentage = (count / total * 100)
                description = get_toi_category_description(category)
                print(f"   {category} ({description}): {count:,} ({percentage:.1f}%)")
            
            # Mapare la categoriile standardizate
            standardized_counts = map_toi_to_standard_categories(toi_data)
            print(f"\n📈 Distribuția în categoriile standardizate:")
            for category, count in standardized_counts.items():
                percentage = (count / total * 100)
                print(f"   {category}: {count:,} ({percentage:.1f}%)")
        
        # Step 3: Predicții ale modelului pe datele TOI
        print(f"\n🤖 Predicții ale modelului pentru datele TOI:")
        
        try:
            # Încarcă datele TOI pentru predicții
            X_toi, y_toi, _, _ = load_dataset("date/TOI_2025.10.04_00.04.19.csv")
            
            if len(X_toi) > 0:
                print(f"   ✅ Date TOI procesate cu succes: {len(X_toi):,} obiecte")
                print(f"   Caracteristici disponibile: {len(X_toi.columns)}")
                
                # Fă predicții pe toate datele TOI
                predictions = clf.predict(X_toi)
                probabilities = clf.predict_proba(X_toi)
                
                # Analizează predicțiile
                prediction_counts = pd.Series(predictions).value_counts()
                print(f"\n📊 Predicțiile modelului pentru toate obiectele TOI:")
                for category, count in prediction_counts.items():
                    percentage = (count / len(predictions) * 100)
                    print(f"   Modelul prezice {category}: {count:,} ({percentage:.1f}%)")
                
                # Compară cu clasificările reale dacă sunt disponibile
                if 'tfopwg_disp' in toi_data.columns and len(y_toi) > 0:
                    print(f"\n📋 Comparația predicții vs clasificări reale TOI:")
                    
                    # Mapează clasificările reale la categoriile standard
                    actual_mapped = y_toi.map(get_standard_category_mapping())
                    
                    # Creează un DataFrame pentru comparație
                    comparison_df = pd.DataFrame({
                        'Clasificări_Reale': actual_mapped.value_counts(),
                        'Predicții_Model': prediction_counts
                    }).fillna(0).astype(int)
                    
                    print(comparison_df)
                    
                    # Calculează acuratețea
                    if len(actual_mapped) == len(predictions):
                        agreement = (actual_mapped == predictions).sum()
                        accuracy = agreement / len(predictions) * 100
                        print(f"\n✅ Acuratețea modelului pe datele TOI: {accuracy:.1f}%")
                        print(f"   ({agreement:,} din {len(predictions):,} obiecte clasificate corect)")
                        
                        # Analizează erorile
                        errors = actual_mapped != predictions
                        if errors.sum() > 0:
                            print(f"\n❌ Analiză erori (primele 10):")
                            error_analysis = pd.DataFrame({
                                'Actual': actual_mapped[errors],
                                'Predicted': pd.Series(predictions)[errors]
                            }).head(10)
                            for idx, row in error_analysis.iterrows():
                                print(f"   Obiect {idx}: Real={row['Actual']}, Prezis={row['Predicted']}")
                
                # Afișează câteva predicții cu încredere mare
                print(f"\n🎯 Primele 10 predicții cu încrederea cea mai mare:")
                max_probs = np.max(probabilities, axis=1)
                high_confidence_idx = np.argsort(max_probs)[-10:][::-1]
                
                for i, idx in enumerate(high_confidence_idx):
                    pred = predictions[idx]
                    conf = max_probs[idx]
                    actual = y_toi.iloc[idx] if len(y_toi) > idx else "Unknown"
                    print(f"   {i+1}. TOI obiect {idx}: Prezis={pred}, Încredere={conf:.3f}, Real={actual}")
                
            else:
                print(f"   ⚠️  Nu s-au putut procesa datele TOI pentru predicții")
                
        except Exception as e:
            print(f"   ❌ Eroare la procesarea datelor TOI: {str(e)}")
            
            # Încearcă cu caracteristici comune
            print(f"   🔄 Încerc cu caracteristici reduse...")
            try:
                # Folosește doar caracteristicile comune
                common_features = ['orbital_period', 'planet_radius', 'stellar_teff', 'stellar_radius']
                available_features = [f for f in common_features if f in X_toi.columns]
                
                if available_features:
                    X_reduced = X_toi[available_features]
                    predictions_reduced = clf.predict(X_reduced)
                    print(f"   ✅ Predicții cu {len(available_features)} caracteristici: {len(predictions_reduced)} obiecte")
                else:
                    print(f"   ❌ Nu sunt disponibile caracteristici comune")
                    
            except Exception as e2:
                print(f"   ❌ Eroare și cu caracteristici reduse: {str(e2)}")
    
    except Exception as e:
        print(f"❌ Eroare la încărcarea datelor TOI: {str(e)}")
    
    # Step 4: Rezumat
    print("\n" + "=" * 70)
    print("Rezumat analiza TOI")
    print("=" * 70)
    print("✅ Analiza completă realizată cu succes!")
    print("\nInformații importante:")
    print("• Datele TOI conțin obiecte TESS (Transiting Exoplanet Survey Satellite)")
    print("• Clasificările includ: CP (Confirmed), PC (Candidate), FP (False Positive)")
    print("• Modelul a fost antrenat pentru a recunoaște aceste categorii")
    print("• Predicțiile pot fi folosite pentru validarea obiectelor candidate")

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
        'CP': 'CONFIRMED',
        'KP': 'CONFIRMED',
        'PC': 'CANDIDATE',
        'APC': 'CANDIDATE',
        'FP': 'FALSE POSITIVE',
        'FA': 'FALSE POSITIVE'
    }

def map_toi_to_standard_categories(toi_data):
    """Mapează categoriile TOI la categoriile standardizate și numără"""
    mapping = get_standard_category_mapping()
    
    if 'tfopwg_disp' in toi_data.columns:
        mapped_categories = toi_data['tfopwg_disp'].map(mapping)
        return mapped_categories.value_counts()
    else:
        return {}

if __name__ == "__main__":
    demo_toi_analysis()
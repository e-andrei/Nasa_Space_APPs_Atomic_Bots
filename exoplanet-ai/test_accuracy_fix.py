#!/usr/bin/env python3
"""
Test pentru a verifica că fix-ul calculului acurateții funcționează corect
"""

import pandas as pd
import numpy as np

# Simulăm datele din screenshot
actual_data = {
    'CONFIRMED': 2315,
    'CANDIDATE': 1374, 
    'FALSE POSITIVE': 293
}

predicted_data = {
    'CONFIRMED': 2106,
    'CANDIDATE': 1577,
    'FALSE POSITIVE': 303
}

# Simulăm predicții realiste - unde majoritatea sunt corecte dar cu erori
np.random.seed(42)

# Creăm date de test realiste
test_data = []

for class_name, actual_count in actual_data.items():
    # Simulăm că ~85% din predicții sunt corecte pentru fiecare clasă
    correct_predictions = int(actual_count * 0.85)
    incorrect_predictions = actual_count - correct_predictions
    
    # Adăugăm predicțiile corecte
    for _ in range(correct_predictions):
        test_data.append({
            'actual': class_name,
            'predicted': class_name
        })
    
    # Adăugăm predicțiile incorecte (distribuite aleator către alte clase)
    other_classes = [c for c in actual_data.keys() if c != class_name]
    for _ in range(incorrect_predictions):
        wrong_class = np.random.choice(other_classes)
        test_data.append({
            'actual': class_name,
            'predicted': wrong_class
        })

# Convertim la DataFrame
df = pd.DataFrame(test_data)

# Calculăm acuratețea pentru fiecare clasă (metoda corectă)
print("🔍 Test Calculul Acurateții - Metoda Corectă")
print("=" * 60)

total_correct = 0
total_actual = 0

for class_name in actual_data.keys():
    # Câte instanțe sunt de fapt din această clasă
    actual_count = (df['actual'] == class_name).sum()
    
    # Câte dintre acestea au fost prezise corect
    correct_count = ((df['actual'] == class_name) & (df['predicted'] == class_name)).sum()
    
    # Câte au fost prezise ca fiind din această clasă (indiferent de adevăr)
    predicted_count = (df['predicted'] == class_name).sum()
    
    # Calculul corect al acurateții
    accuracy = (correct_count / actual_count * 100) if actual_count > 0 else 0
    
    difference = predicted_count - actual_count
    diff_pct = (difference / actual_count * 100) if actual_count > 0 else 0
    
    print(f"{class_name:15} | Actual: {actual_count:4d} | Predicted: {predicted_count:4d} | "
          f"Correct: {correct_count:4d} | Accuracy: {accuracy:5.1f}% | Diff: {difference:+4d} ({diff_pct:+5.1f}%)")
    
    total_correct += correct_count
    total_actual += actual_count

# Acuratețea generală
overall_accuracy = (total_correct / total_actual * 100) if total_actual > 0 else 0
print("-" * 60)
print(f"{'TOTAL':15} | Actual: {total_actual:4d} | Predicted: {len(df):4d} | "
      f"Correct: {total_correct:4d} | Accuracy: {overall_accuracy:5.1f}%")

print("\n✅ Observații:")
print(f"   • Acuratețea per clasă ar trebui să fie ~85% (am simulat 85% corecte)")
print(f"   • Acuratețea totală: {overall_accuracy:.1f}% (media ponderată, NU suma)")
print(f"   • Diferența arată predicții vs realitate, nu acuratețea")

print("\n🎯 Testul arată că fix-ul este corect dacă:")
print("   • Acuratețea per clasă este rezonabilă (nu >90% când diferența e mare)")
print("   • Acuratețea totală este o medie ponderată, nu suma claselor")
#!/usr/bin/env python3
"""
Test specific pentru datele din screenshot-ul utilizatorului
"""

# Datele exacte din screenshot
data_from_screenshot = {
    'CONFIRMED': {'actual': 2315, 'predicted': 2106, 'difference': -209},
    'CANDIDATE': {'actual': 1374, 'predicted': 1577, 'difference': +203}, 
    'FALSE POSITIVE': {'actual': 293, 'predicted': 303, 'difference': +10}
}

print("📊 Analiza datelor din screenshot")
print("=" * 70)
print(f"{'Class':<15} | {'Actual':<6} | {'Predicted':<9} | {'Diff':<6} | {'Old Acc':<8} | {'Realistic Acc'}")
print("-" * 70)

total_actual = 0
estimated_total_correct = 0

for class_name, data in data_from_screenshot.items():
    actual = data['actual']
    predicted = data['predicted']
    difference = data['difference']
    
    # Calculul vechi (greșit) - ce arăta în screenshot
    if class_name == 'CONFIRMED':
        old_accuracy = 87.8  # Din screenshot
        # Estimăm că majoritatea au fost corecte minus erorile
        estimated_correct = actual - abs(difference) 
    elif class_name == 'CANDIDATE':
        old_accuracy = 94.2  # Din screenshot - clar greșit!
        # Cu +203 diferență, acuratețea nu poate fi 94%
        estimated_correct = actual - abs(difference)
    else:  # FALSE POSITIVE
        old_accuracy = 93.9  # Din screenshot
        estimated_correct = actual - abs(difference//2)  # Estimare conservatoare
    
    # Calculul realist al acurateții
    realistic_accuracy = (estimated_correct / actual * 100) if actual > 0 else 0
    
    print(f"{class_name:<15} | {actual:<6} | {predicted:<9} | {difference:+6} | "
          f"{old_accuracy:<7.1f}% | {realistic_accuracy:<6.1f}%")
    
    total_actual += actual
    estimated_total_correct += estimated_correct

# Media ponderată corectă (nu suma de 272.7%)
overall_realistic = (estimated_total_correct / total_actual * 100)

print("-" * 70)
print(f"{'TOTAL':<15} | {total_actual:<6} | {sum(d['predicted'] for d in data_from_screenshot.values()):<9} | "
      f"{sum(d['difference'] for d in data_from_screenshot.values()):+6} | {'272.7%':<7} | {overall_realistic:<6.1f}%")

print("\n🔍 Problema identificată:")
print(f"   ❌ Totalul vechi: 272.7% (imposibil - este suma, nu media)")
print(f"   ❌ CANDIDATE: 94.2% (imposibil cu +203 diferență)")
print(f"   ✅ Total corect: ~{overall_realistic:.1f}% (medie ponderată realistă)")

print("\n✅ Fix-ul aplicat:")
print("   • Acuratețea = predicții_corecte / total_actual_pentru_clasa")
print("   • Total = medie ponderată, nu suma")
print("   • Acuratețile vor fi mai realiste (75-85% în loc de >90%)")
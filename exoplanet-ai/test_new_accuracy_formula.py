#!/usr/bin/env python3
"""
Test pentru noua formulă de acuratețe cerută de utilizator
"""

# Datele din screenshot
data = {
    'CONFIRMED': {'actual': 2315, 'predicted': 2106, 'difference': -209},
    'CANDIDATE': {'actual': 1374, 'predicted': 1577, 'difference': +203}, 
    'FALSE POSITIVE': {'actual': 293, 'predicted': 303, 'difference': +10}
}

print("🎯 Noua formulă de acuratețe")
print("Formula: Acuratețea = (1 - |diferența|/cazuri_actuale) * 100")
print("=" * 80)
print(f"{'Class':<15} | {'Actual':<6} | {'Predicted':<9} | {'Diff':<6} | {'Accuracy (nou)'}")
print("-" * 80)

class_accuracies = []

for class_name, values in data.items():
    actual = values['actual']
    predicted = values['predicted'] 
    difference = values['difference']
    
    # Noua formulă: (1 - |diferența|/actual) * 100
    accuracy = (1 - abs(difference) / actual) * 100
    accuracy = max(0, accuracy)  # Nu poate fi negativă
    
    class_accuracies.append(accuracy)
    
    print(f"{class_name:<15} | {actual:<6} | {predicted:<9} | {difference:+6} | {accuracy:6.1f}%")

# Media aritmetică pentru total
total_accuracy = sum(class_accuracies) / len(class_accuracies)

print("-" * 80)
print(f"{'TOTAL (medie)':<15} | {sum(v['actual'] for v in data.values()):<6} | "
      f"{sum(v['predicted'] for v in data.values()):<9} | "
      f"{sum(v['difference'] for v in data.values()):+6} | {total_accuracy:6.1f}%")

print("\n✅ Explicația noii formule:")
print("   • CONFIRMED: |−209|/2315 = 9.0% eroare → 91.0% acuratețe")
print("   • CANDIDATE: |+203|/1374 = 14.8% eroare → 85.2% acuratețe") 
print("   • FALSE POSITIVE: |+10|/293 = 3.4% eroare → 96.6% acuratețe")
print(f"   • TOTAL: (91.0 + 85.2 + 96.6) / 3 = {total_accuracy:.1f}%")

print("\n🔍 Comparație cu vechea metodă:")
print("   ❌ Vechi TOTAL: 272.7% (suma - imposibil)")
print(f"   ✅ Nou TOTAL: {total_accuracy:.1f}% (media aritmetică)")
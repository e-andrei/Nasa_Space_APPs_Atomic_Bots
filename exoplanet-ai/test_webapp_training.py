#!/usr/bin/env python3
"""
Test script to verify web app training functionality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model import train_from_multiple_csv

def test_web_app_training():
    """Test the training functionality that would be used by the web app."""
    print("🧪 Testing Web App Training Functionality...")
    
    # Test multi-file training with XGBoost (like in web app)
    try:
        print("Testing multi-file XGBoost training...")
        result = train_from_multiple_csv(
            data_paths=[
                "date/cumulative_2025.10.03_23.13.19.csv",
                "date/k2pandc_2025.10.03_23.45.46.csv",
                "date/TOI_2025.10.04_00.04.19.csv"
            ],
            output_model_path="models/webapp_test_xgb.joblib",
            model_type='xgboost',
            tune=False,  # Skip tuning for speed
            random_state=42
        )
        
        print("✅ Multi-file XGBoost training successful")
        print(f"   Model saved to: {result['model_path']}")
        print(f"   Datasets: {result['metadata']['n_datasets']}")
        print(f"   Total samples: {result['metadata']['n_samples']}")
        
        # Test Random Forest too
        print("\nTesting multi-file Random Forest training...")
        result2 = train_from_multiple_csv(
            data_paths=[
                "date/cumulative_2025.10.03_23.13.19.csv",
                "date/k2pandc_2025.10.03_23.45.46.csv"
            ],
            output_model_path="models/webapp_test_rf.joblib",
            model_type='random_forest',
            tune=False,
            random_state=42
        )
        
        print("✅ Multi-file Random Forest training successful")
        print(f"   Model saved to: {result2['model_path']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Web app training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_web_app_training()
    print(f"\n{'🎉 Web App Training Test PASSED' if success else '❌ Web App Training Test FAILED'}")
    
    if success:
        print("\n📋 Summary of fixes:")
        print("✅ XGBoost pickling issue fixed")
        print("✅ Multi-file training working")
        print("✅ Both XGBoost and Random Forest supported")
        print("✅ Model persistence working")
        print("🎯 The web app should now work correctly for training!")
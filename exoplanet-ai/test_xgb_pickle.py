#!/usr/bin/env python3
"""
Test script to check if XGBoost pickling issue is fixed
"""

import sys
import joblib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model import train_from_csv

def test_xgboost_pickling():
    """Test if XGBoost model can be pickled and unpickled successfully."""
    print("🧪 Testing XGBoost pickling fix...")
    
    try:
        # Train a small XGBoost model
        result = train_from_csv(
            data_path="date/cumulative_2025.10.03_23.13.19.csv",
            output_model_path="models/test_xgb_pickle.joblib",
            model_type='xgboost',
            tune=False,
            random_state=42
        )
        
        print("✅ XGBoost model trained and saved successfully")
        
        # Try to load it back
        model_data = joblib.load("models/test_xgb_pickle.joblib")
        print("✅ XGBoost model loaded successfully")
        
        # Test the classifier
        clf = result['classifier']
        print(f"✅ Classifier type: {type(clf.pipeline.named_steps['classifier'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ XGBoost pickling test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_xgboost_pickling()
    print(f"\n{'🎉 Test PASSED' if success else '❌ Test FAILED'}")
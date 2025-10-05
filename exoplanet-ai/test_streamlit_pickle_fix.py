#!/usr/bin/env python3
"""
Test XGBoost pickling through Streamlit app import structure.
This simulates how the Streamlit app imports and uses the model.
"""

import sys
from pathlib import Path

# Add src to path like Streamlit app does
root = Path(__file__).resolve().parent
src_dir = root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import using the same method as Streamlit app
import importlib.util
spec = importlib.util.spec_from_file_location("model", src_dir / "model.py")
if spec and spec.loader:
    # Check if already loaded
    if "model" in sys.modules:
        model_mod = sys.modules["model"]
    else:
        # Load fresh
        model_mod = importlib.util.module_from_spec(spec)
        sys.modules["model"] = model_mod
        spec.loader.exec_module(model_mod)

print("✅ Model module loaded successfully")
print(f"   Module: {model_mod}")
print(f"   XGBClassifierFixed: {model_mod.XGBClassifierFixed}")

# Test training and pickling like the app does
print("\n🧪 Testing XGBoost training and pickling...")

# Load a small dataset for testing
csv_path = "date/cumulative_2025.10.03_23.13.19.csv"
if Path(csv_path).exists():
    result = model_mod.train_from_csv(
        data_path=csv_path,
        output_model_path='test_streamlit_pickle.joblib',
        model_type='xgboost',
        random_state=42
    )
    
    print("✅ Training completed successfully")
    print(f"   Accuracy: {result.get('accuracy', 'N/A')}")
    
    # Test loading the saved model
    import joblib
    loaded_model = joblib.load('test_streamlit_pickle.joblib')
    print("✅ Model loaded successfully after pickling")
    print(f"   Pipeline: {type(loaded_model['pipeline'])}")
    
    # Clean up
    Path('test_streamlit_pickle.joblib').unlink(missing_ok=True)
    Path('test_streamlit_pickle.json').unlink(missing_ok=True)
    
    print("\n🎉 STREAMLIT PICKLE TEST PASSED")
    print("✅ XGBoost can be trained and pickled through Streamlit import structure")
    
else:
    print(f"❌ Test data not found: {csv_path}")
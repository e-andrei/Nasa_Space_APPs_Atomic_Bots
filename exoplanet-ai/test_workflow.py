#!/usr/bin/env python3
"""
Test script to validate the complete ML workflow for exoplanet classification.
Tests: data loading, training, prediction, model saving/loading.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model import (
    load_kepler_data, 
    train_from_csv, 
    train_from_multiple_csv,
    ExoplanetClassifier
)

def test_single_file_training():
    """Test training on a single CSV file."""
    print("🧪 Testing single file training...")
    
    data_path = "date/cumulative_2025.10.03_23.13.19.csv"
    output_path = "models/test_single_classifier.joblib"
    
    try:
        result = train_from_csv(
            data_path=data_path,
            output_model_path=output_path,
            model_type='random_forest',
            tune=False,  # Skip tuning for faster testing
            random_state=42
        )
        
        print("✅ Single file training successful")
        print(f"   Model saved to: {result['model_path']}")
        print(f"   Features: {len(result['features'])}")
        print(f"   Samples: {result['metadata']['n_samples']}")
        
        return True
    except Exception as e:
        print(f"❌ Single file training failed: {e}")
        return False

def test_multi_file_training():
    """Test training on multiple CSV files."""
    print("\n🧪 Testing multi-file training...")
    
    data_paths = [
        "date/cumulative_2025.10.03_23.13.19.csv",
        "date/k2pandc_2025.10.03_23.45.46.csv",
        "date/TOI_2025.10.04_00.04.19.csv"
    ]
    output_path = "models/test_multi_classifier.joblib"
    
    try:
        result = train_from_multiple_csv(
            data_paths=data_paths,
            output_model_path=output_path,
            model_type='random_forest',
            tune=False,  # Skip tuning for faster testing
            random_state=42
        )
        
        print("✅ Multi-file training successful")
        print(f"   Model saved to: {result['model_path']}")
        print(f"   Datasets: {result['metadata']['n_datasets']}")
        print(f"   Total samples: {result['metadata']['n_samples']}")
        print(f"   Features: {len(result['features'])}")
        
        return True
    except Exception as e:
        print(f"❌ Multi-file training failed: {e}")
        return False

def test_model_loading_and_prediction():
    """Test loading a trained model and making predictions."""
    print("\n🧪 Testing model loading and prediction...")
    
    model_path = "models/test_single_classifier.joblib"
    data_path = "date/cumulative_2025.10.03_23.13.19.csv"
    
    try:
        # Load model
        classifier = ExoplanetClassifier.load(model_path)
        print("✅ Model loaded successfully")
        
        # Load test data
        X, y, features = load_kepler_data(data_path)
        X_sample = X.head(10)  # Just test with 10 samples
        
        # Make predictions
        predictions = classifier.predict(X_sample)
        probabilities = classifier.predict_proba(X_sample)
        
        print(f"✅ Predictions made for {len(X_sample)} samples")
        print(f"   Prediction classes: {np.unique(predictions)}")
        print(f"   Probability shape: {probabilities.shape}")
        
        # Test evaluation
        metrics = classifier.evaluate(X_sample, y.head(10))
        print(f"✅ Model evaluation completed")
        print(f"   Macro F1: {metrics['macro_f1']:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Model loading/prediction failed: {e}")
        return False

def test_data_format_compatibility():
    """Test that all three data formats can be loaded."""
    print("\n🧪 Testing data format compatibility...")
    
    files = [
        ("Kepler KOI", "date/cumulative_2025.10.03_23.13.19.csv"),
        ("K2/PANDC", "date/k2pandc_2025.10.03_23.45.46.csv"),
        ("TESS TOI", "date/TOI_2025.10.04_00.04.19.csv")
    ]
    
    all_passed = True
    
    for name, filepath in files:
        try:
            X, y, features = load_kepler_data(filepath)
            print(f"✅ {name}: {len(X)} samples, {len(features)} features")
            print(f"   Classes: {y.value_counts().to_dict()}")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            all_passed = False
    
    return all_passed

def main():
    """Run all tests."""
    print("🚀 Starting Exoplanet ML Workflow Tests")
    print("=" * 50)
    
    # Ensure models directory exists
    Path("models").mkdir(exist_ok=True)
    
    tests = [
        test_data_format_compatibility,
        test_single_file_training,
        test_multi_file_training,
        test_model_loading_and_prediction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! The ML workflow is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
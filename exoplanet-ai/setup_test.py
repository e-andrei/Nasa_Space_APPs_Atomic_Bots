#!/usr/bin/env python3
"""
Simplified setup test for debugging purposes.
"""

import sys
import os
from pathlib import Path

def test_basic_training():
    """Test basic model training with minimal dependencies."""
    print("Testing basic model training...")
    
    # Add src to path
    sys.path.insert(0, str(Path("src").absolute()))
    
    try:
        # Create minimal test data
        import pandas as pd
        import numpy as np
        
        # Create simple synthetic data
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'period': np.random.uniform(1, 100, n_samples),
            'depth': np.random.uniform(0.001, 0.01, n_samples), 
            'duration': np.random.uniform(1, 10, n_samples),
            'snr': np.random.uniform(5, 50, n_samples),
        }
        
        # Create target with realistic distribution
        dispositions = np.random.choice(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'], 
                                      n_samples, p=[0.1, 0.3, 0.6])
        
        X = pd.DataFrame(data)
        y = pd.Series(dispositions)
        
        print(f"Created test data: {X.shape}, classes: {y.value_counts().to_dict()}")
        
        # Test model
        from model import ExoplanetClassifier
        
        classifier = ExoplanetClassifier()
        classifier.build_pipeline(list(X.columns))
        
        print("Training model...")
        results = classifier.train(X, y, tune_params=False)
        print("✅ Training successful!")
        
        # Test predictions
        predictions = classifier.predict(X[:5])
        probabilities = classifier.predict_proba(X[:5])
        
        print(f"✅ Predictions: {predictions}")
        print(f"✅ Probabilities shape: {probabilities.shape}")
        
        # Test evaluation
        metrics = classifier.evaluate(X, y, return_dict=True)
        print(f"✅ F1-Score: {metrics['macro_f1']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔭 Simple Model Training Test")
    print("=" * 40)
    
    success = test_basic_training()
    
    if success:
        print("\n✅ Basic training test passed!")
        print("The model training pipeline is working correctly.")
    else:
        print("\n❌ Basic training test failed!")
        print("There may be an issue with the model code.")
    
    exit(0 if success else 1)
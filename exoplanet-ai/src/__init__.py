# Exoplanet AI Package
"""
Exoplanet classification system for NASA Space Apps Challenge 2024.

This package provides tools for:
- Loading and preprocessing exoplanet catalog data
- Training machine learning models for exoplanet classification
- Making predictions on new exoplanet candidates
- Explaining model decisions with SHAP and feature importance
- Serving models via web interface and batch processing
"""

__version__ = "1.0.0"
__author__ = "NASA Space Apps Team"

from .data import load_dataset, create_sample_data
from .model import ExoplanetClassifier
from .explain import ModelExplainer
from .serve import ExoplanetPredictor

__all__ = [
    'load_dataset',
    'create_sample_data', 
    'ExoplanetClassifier',
    'ModelExplainer',
    'ExoplanetPredictor'
]
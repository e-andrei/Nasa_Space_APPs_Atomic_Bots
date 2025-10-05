#!/usr/bin/env python3
"""
Comprehensive demonstration of the Exoplanet ML Web Application workflow.
This script demonstrates how to use all features of the application.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    st.title("🚀 Exoplanet ML Web App - Complete Workflow Guide")
    
    st.markdown("""
    This guide demonstrates the complete machine learning workflow for exoplanet classification,
    including model selection, testing, and retraining with different parameters.
    """)
    
    # Workflow Steps
    st.header("📋 Complete Workflow Steps")
    
    workflow_steps = [
        {
            "step": "1. Model Loading",
            "description": "Load an existing model or train a new one",
            "actions": [
                "Set model path in sidebar (e.g., 'models/exoplanet_classifier.joblib')",
                "View model metadata and features",
                "Check training metrics and configuration"
            ]
        },
        {
            "step": "2. Data Upload & Testing",
            "description": "Test the model on new data",
            "actions": [
                "Upload CSV file(s) in 'Upload CSV' tab",
                "Preview data structure and feature mapping",
                "Run predictions and view results",
                "Download predictions with confidence scores"
            ]
        },
        {
            "step": "3. Model Analysis",
            "description": "Understand model behavior and performance",
            "actions": [
                "Check feature importance (tree-based or permutation)",
                "Analyze threshold effects on predictions",
                "Explore advanced dataset statistics",
                "Review confusion matrix and classification metrics"
            ]
        },
        {
            "step": "4. Model Retraining",
            "description": "Train new models with different parameters",
            "actions": [
                "Upload training data (single or multiple files)",
                "Select model type (XGBoost, Random Forest)",
                "Adjust class weights and hyperparameters",
                "Enable/disable hyperparameter tuning",
                "Train and automatically load new model"
            ]
        }
    ]
    
    for step_info in workflow_steps:
        with st.expander(f"**{step_info['step']}**: {step_info['description']}", expanded=False):
            for action in step_info['actions']:
                st.write(f"• {action}")
    
    # Available Models
    st.header("🤖 Available Model Types")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("XGBoost")
        st.write("""
        **Best for:**
        - High accuracy on structured data
        - Feature importance analysis
        - Handling missing values
        
        **Hyperparameters:**
        - n_estimators: 50-300
        - max_depth: 3-10
        - learning_rate: 0.01-0.3
        - regularization: L1/L2
        """)
    
    with col2:
        st.subheader("Random Forest")
        st.write("""
        **Best for:**
        - Robust, interpretable results
        - Less prone to overfitting
        - Faster training
        
        **Hyperparameters:**
        - n_estimators: 50-200
        - max_depth: 5-15 or None
        - min_samples_split: 2-10
        - min_samples_leaf: 1-4
        """)
    
    # Class Weights
    st.header("⚖️ Class Weight Configuration")
    
    st.write("""
    Adjust class weights to handle imbalanced datasets and bias the model toward specific outcomes:
    """)
    
    weight_info = pd.DataFrame({
        'Class': ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE', 'OTHER'],
        'Default Weight': [0.7, 1.0, 1.25, 1.1],
        'Effect': [
            'Lower weight = fewer confirmations (conservative)',
            'Balanced weight for planet candidates',
            'Higher weight = better false positive detection',
            'Weight for other/unknown classes'
        ]
    })
    
    st.dataframe(weight_info, width='stretch')
    
    # Data Formats
    st.header("📊 Supported Data Formats")
    
    format_tabs = st.tabs(["Kepler KOI", "K2/PANDC", "TESS TOI"])
    
    with format_tabs[0]:
        st.write("**Kepler Objects of Interest (KOI)**")
        st.code("""
        Key columns: koi_disposition, koi_period, koi_depth, koi_duration,
                    koi_prad, koi_teq, koi_steff, koi_srad, koi_score
        Target: koi_disposition (CONFIRMED, CANDIDATE, FALSE POSITIVE)
        Features: 16 including false positive flags
        """)
    
    with format_tabs[1]:
        st.write("**K2 Planets and Candidates**")
        st.code("""
        Key columns: disposition, pl_orbper, pl_rade, pl_eqt,
                    st_teff, st_rad, st_logg
        Target: disposition (CONFIRMED, CANDIDATE, FALSE POSITIVE, REFUTED)
        Features: 8 core planetary and stellar parameters
        """)
    
    with format_tabs[2]:
        st.write("**TESS Objects of Interest (TOI)**")
        st.code("""
        Key columns: tfopwg_disp, pl_orbper, pl_rade, pl_eqt,
                    st_teff, st_rad, st_logg
        Target: tfopwg_disp (CP, PC, FP, APC, KP, FA)
        Features: 8 core parameters (similar to K2)
        """)
    
    # Performance Tips
    st.header("⚡ Performance Tips")
    
    tips = [
        "**Multi-file training**: Combine Kepler, K2, and TESS data for better generalization",
        "**Hyperparameter tuning**: Enable for final models, disable for quick testing",
        "**Class weights**: Adjust based on your scientific goals (conservative vs comprehensive)",
        "**Feature importance**: Use to understand which parameters matter most",
        "**Threshold tuning**: Adjust confidence thresholds to control prediction uncertainty",
        "**Cross-validation**: Use 5-10 folds for robust performance estimates"
    ]
    
    for tip in tips:
        st.write(f"• {tip}")
    
    # Example Commands
    st.header("💻 Command Line Usage")
    
    st.write("You can also train models via command line:")
    
    st.code("""
    # Single file training
    python src/model.py --data "data.csv" --output "model.joblib" --model-type "xgboost" --tune
    
    # Multi-file training
    python src/model.py --data "kepler.csv" "k2.csv" "toi.csv" --output "combined_model.joblib"
    
    # With hyperparameter tuning
    python src/model.py --data "data.csv" --tune --cv-folds 10 --n-iter 50
    """, language='bash')
    
    # Troubleshooting
    st.header("🔧 Troubleshooting")
    
    with st.expander("Common Issues and Solutions", expanded=False):
        st.write("""
        **Model won't load:**
        - Check file path is correct
        - Ensure model was saved successfully
        - Try regenerating the model
        
        **Poor prediction accuracy:**
        - Check data quality and preprocessing
        - Try different model types
        - Adjust class weights
        - Use hyperparameter tuning
        
        **Training fails:**
        - Verify CSV format and columns
        - Check for sufficient data samples
        - Ensure target column exists
        - Try smaller dataset first
        
        **Memory issues:**
        - Reduce dataset size for testing
        - Disable hyperparameter tuning
        - Use Random Forest instead of XGBoost
        """)

if __name__ == "__main__":
    main()
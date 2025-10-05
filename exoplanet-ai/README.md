# 🔭 Exoplanet Classification System

An AI-powered system to classify exoplanet candidates as **CONFIRMED**, **CANDIDATE**, or **FALSE POSITIVE** using machine learning on tabular astronomical data.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd exoplanet-ai

# Install dependencies
pip install -r requirements.txt
```

### 2. Create Sample Data (for testing)

```bash
python src/data.py
```

This creates `data/sample_kepler.csv` with 1000 synthetic exoplanet candidates.

### 3. Train the Model

```bash
# Quick training (no hyperparameter tuning)
python src/model.py --data data/sample_kepler.csv

# Full training with hyperparameter tuning (recommended)
python src/model.py --data data/sample_kepler.csv --tune --cv 5
```

### 4. Launch the Web App

```bash
streamlit run app/streamlit_app.py
```

Visit `http://localhost:8501` to use the interactive interface!

## 📁 Project Structure

```
exoplanet-ai/
├── data/                    # CSV datasets (raw & processed)
├── notebooks/               # Jupyter notebooks for EDA & experiments
├── src/                     # Core source code
│   ├── data.py             # Data loading & preprocessing
│   ├── model.py            # ML pipeline & training
│   ├── explain.py          # Model interpretability (SHAP, etc.)
│   └── serve.py            # Batch inference utilities
├── app/
│   └── streamlit_app.py    # Web interface
├── models/                 # Saved models & metadata
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🎯 Features

### Core Functionality
- **3-class classification**: CONFIRMED, CANDIDATE, FALSE POSITIVE
- **XGBoost** with hyperparameter tuning
- **Class imbalance handling** with SMOTE
- **Robust evaluation** using stratified K-fold CV and PR-AUC metrics

### Data Support
- **Kepler DR25** catalog format
- **TESS TOI** catalog format  
- **Generic CSV** with automatic feature detection
- **Missing value handling** and feature engineering

### Model Interpretability
- **SHAP** explanations for individual predictions
- **Permutation importance** for global feature ranking
- **Feature importance** comparison across methods

### User Interfaces
- **Streamlit web app** for interactive predictions
- **Command-line tools** for batch processing
- **CSV upload/download** for easy data handling

## 🔬 Usage Examples

### Training a Model

```bash
# Basic training
python src/model.py --data data/kepler_dr25.csv

# Advanced training with options
python src/model.py \
  --data data/kepler_dr25.csv \
  --catalog kepler \
  --tune \
  --smote \
  --cv 5 \
  --output models/
```

### Batch Predictions

```bash
# Make predictions on new data
python src/serve.py predict --input data/new_candidates.csv --output predictions.csv

# Evaluate model on labeled data
python src/serve.py evaluate --input data/test_set.csv --truth-column disposition

# Get model information
python src/serve.py info

# Create prediction template
python src/serve.py template --output template.csv
```

### Using the Python API

```python
from src.model import ExoplanetClassifier
from src.data import load_dataset

# Load data
X, y, numeric_cols, cat_cols = load_dataset("data/sample_kepler.csv")

# Train model
classifier = ExoplanetClassifier()
classifier.build_pipeline(numeric_cols, cat_cols)
classifier.train(X, y, tune_params=True)

# Make predictions
predictions = classifier.predict(X)
probabilities = classifier.predict_proba(X)

# Save model
classifier.save("models/my_model.joblib")

# Load model later
loaded_classifier = ExoplanetClassifier.load("models/my_model.joblib")
```

## 📊 Web Interface Features

The Streamlit app provides:

1. **🏠 Home**: Overview and quick start guide
2. **📊 Model Performance**: Detailed metrics, confusion matrix, per-class performance
3. **🔍 Make Predictions**: 
   - CSV file upload for batch predictions
   - Manual input for single sample prediction
   - Downloadable results with probabilities
4. **📈 Feature Analysis**: Feature importance plots and analysis
5. **⚙️ Model Info**: Model configuration and metadata

## 🔧 Configuration

### Supported Data Formats

Your CSV should contain exoplanet features like:

**Required column**: `disposition` (target variable)

**Recommended features**:
- `orbital_period`: Orbital period in days
- `transit_depth`: Transit depth in ppm
- `transit_duration`: Transit duration in hours  
- `planet_radius`: Planet radius in Earth radii
- `stellar_teff`: Stellar effective temperature in K
- `stellar_radius`: Stellar radius in solar radii
- `stellar_mass`: Stellar mass in solar masses
- `snr`: Signal-to-noise ratio
- `impact_parameter`: Impact parameter (0-1)

### Disposition Mapping

The system automatically maps various disposition formats:
- `CONFIRMED` → CONFIRMED
- `CANDIDATE`, `PC` → CANDIDATE  
- `FALSE POSITIVE`, `FALSE_POSITIVE`, `FP`, `AFP`, `NTP` → FALSE POSITIVE

### Model Hyperparameters

Default XGBoost configuration:
```python
{
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'objective': 'multi:softprob'
}
```

Hyperparameter tuning searches over:
- `n_estimators`: [300, 500, 700, 1000]
- `max_depth`: [4, 5, 6, 7, 8]  
- `learning_rate`: [0.05, 0.1, 0.15, 0.2]
- `subsample`: [0.8, 0.9, 1.0]
- `colsample_bytree`: [0.8, 0.9, 1.0]

## 📈 Performance Metrics

The system tracks multiple metrics optimized for imbalanced classification:

- **Accuracy**: Overall classification accuracy
- **Macro Average Precision**: Mean AP across all classes
- **Per-class Precision/Recall/F1**: Detailed performance per class
- **ROC-AUC**: Area under ROC curve (OvR and OvO for multiclass)
- **Confusion Matrix**: Detailed classification breakdown

## 🛠️ Advanced Usage

### Custom Data Preprocessing

```python
from src.data import load_dataset

# Load with specific catalog type
X, y, num_cols, cat_cols = load_dataset("data/my_data.csv", catalog_type="kepler")

# Create custom sample data
from src.data import create_sample_data
create_sample_data("data/custom_sample.csv", n_samples=5000)
```

### Model Explainability

```python
from src.explain import ModelExplainer
from src.model import ExoplanetClassifier

# Load trained model
classifier = ExoplanetClassifier.load("models/exoplanet_classifier.joblib")

# Create explainer
explainer = ModelExplainer(classifier.pipeline, X_train)

# Generate comprehensive explanation report
explainer.generate_explanation_report(X_test, y_test, output_dir="explanations/")

# Explain single prediction
explanation = explainer.explain_single_prediction(sample)
```

### Handling Class Imbalance

```python
# Use SMOTE for oversampling
classifier.build_pipeline(numeric_cols, cat_cols, use_smote=True)

# Or use class weights in XGBoost
# (modify model.py to add class_weight parameter)
```

## 📚 Data Sources

Recommended datasets for training:

1. **NASA Exoplanet Archive**:
   - Kepler DR25 KOI table
   - TESS Objects of Interest (TOI)
   - Confirmed planets table

2. **Kaggle Datasets**:
   - Kepler Exoplanet Search Results
   - TESS Exoplanet Candidates

3. **Direct Downloads**:
   ```bash
   # Example: Kepler DR25 (large file!)
   wget https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&format=csv
   ```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're in the project root directory and have installed all dependencies

2. **Model Not Found**: Train a model first using `python src/model.py`

3. **Memory Issues**: For large datasets, reduce `n_iter` in hyperparameter tuning or use smaller subsets

4. **SHAP Errors**: SHAP is optional - the system works without it if import fails

5. **Feature Mismatch**: Ensure prediction data has the same features as training data

### Performance Tips

- **Use GPU**: Install `xgboost[gpu]` for faster training
- **Parallel Processing**: Increase `n_jobs=-1` for multi-core usage  
- **Memory Management**: Use chunked processing for very large datasets
- **Feature Selection**: Remove low-importance features to speed up inference

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- **NASA Exoplanet Archive** for providing high-quality datasets
- **XGBoost** team for the excellent gradient boosting library
- **Streamlit** for making web app development so easy
- **SHAP** for model interpretability tools
- **NASA Space Apps Challenge** for the inspiration

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@software{exoplanet_classifier,
  title={Exoplanet Classification System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/exoplanet-ai}
}
```

---

**Built for the NASA Space Apps Challenge 2024** 🚀

Happy exoplanet hunting! 🔭✨
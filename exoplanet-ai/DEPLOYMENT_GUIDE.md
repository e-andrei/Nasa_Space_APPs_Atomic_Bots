# Deployment Guide for Exoplanet AI Streamlit App

## Problem Fixed

The app was failing in cloud deployments with:
```
Failed to load model: [Errno 2] No such file or directory: 'models/unified_xgb_tuned.joblib'
```

This happened because the app was using relative paths that depend on the current working directory, which varies in cloud deployments.

## Solution Implemented

✅ **Fixed path resolution** - All paths are now resolved relative to the script location, not the working directory.

✅ **Added robust path helpers** - `get_project_root()` and `resolve_path()` functions ensure consistent path resolution.

✅ **Better error messages** - Clear feedback when models are missing in deployment.

## Deployment Requirements

### 1. File Structure
Ensure your deployment includes these directories:
```
exoplanet-ai/
├── app/
│   └── streamlit_app.py
├── models/
│   ├── unified_xgb_tuned.joblib    # Your default model
│   ├── unified_xgb_tuned.json      # Model metadata
│   └── ... (other model files)
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── data.py
│   └── ... (other source files)
└── requirements.txt
```

### 2. Entry Point
Make sure your cloud service runs the app from the correct location:
```bash
# From the exoplanet-ai directory
streamlit run app/streamlit_app.py
```

### 3. Common Cloud Service Configurations

#### Streamlit Cloud
```toml
# .streamlit/config.toml (optional)
[server]
headless = true
enableCORS = false
```

#### Heroku
```
# Procfile
web: streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

#### Railway/Render
```bash
# Build command (if needed)
pip install -r requirements.txt

# Start command
streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

#### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
```

### 4. Environment Variables (if needed)
Some cloud services may require these:
```bash
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## Verification

After deployment, check that:
1. ✅ App loads without "No such file or directory" errors
2. ✅ Model dropdown shows available models
3. ✅ File upload and prediction works
4. ✅ Manual input generates predictions

## Troubleshooting

### If you still get model loading errors:

1. **Check file structure** - Ensure the `models/` directory is included in your deployment
2. **Verify model files** - Make sure `.joblib` files are not corrupted during upload
3. **Check working directory** - The app should run from the `exoplanet-ai/` directory
4. **Large files** - Some services have file size limits; consider using model hosting services for very large models

### Model file size optimization:

If model files are too large for your deployment service:
1. Use model compression techniques
2. Host models externally (AWS S3, Google Cloud Storage)
3. Implement lazy loading from external sources

## Testing Locally

Before deploying, test the path resolution:
```python
# From exoplanet-ai directory
python -c "from app.streamlit_app import get_project_root, resolve_path; print('Root:', get_project_root()); print('Model exists:', resolve_path('models/unified_xgb_tuned.joblib').exists())"
```

Should output:
```
Root: /path/to/exoplanet-ai
Model exists: True
```

## Success! 🎉

The app should now deploy successfully to any cloud service that supports Python and Streamlit, regardless of the working directory structure.
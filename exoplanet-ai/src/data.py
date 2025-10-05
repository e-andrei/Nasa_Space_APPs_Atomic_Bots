# src/data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard disposition mapping for different catalogs
DISPOSITION_MAP = {
    "CONFIRMED": "CONFIRMED",
    "CANDIDATE": "CANDIDATE", 
    "FALSE POSITIVE": "FALSE POSITIVE",
    "FALSE_POSITIVE": "FALSE POSITIVE",
    "FP": "FALSE POSITIVE",
    "PC": "CANDIDATE",  # Planet Candidate
    "CP": "CONFIRMED",  # Confirmed Planet
    "AFP": "FALSE POSITIVE",  # Astrophysical False Positive
    "NTP": "FALSE POSITIVE"   # Non-transiting phenomenon
}

def load_kepler_dr25(file_path):
    """Load and preprocess Kepler DR25 catalog data."""
    df = pd.read_csv(file_path)
    
    # Rename columns to standard names if needed
    column_mapping = {
        'koi_disposition': 'disposition',
        'koi_period': 'orbital_period',
        'koi_depth': 'transit_depth',
        'koi_duration': 'transit_duration',
        'koi_prad': 'planet_radius',
        'koi_sma': 'semi_major_axis',
        'koi_impact': 'impact_parameter',
        'koi_teq': 'equilibrium_temp',
        'koi_insol': 'insolation',
        'koi_dor': 'distance_over_rstar',
        'koi_ror': 'rp_rs',
        'koi_srho': 'stellar_density',
        'koi_fittype': 'fit_type',
        'koi_score': 'disposition_score',
        'ra': 'ra',
        'dec': 'dec'
    }
    
    # Apply column mapping
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)
    
    return df

def load_tess_toi(file_path):
    """Load and preprocess TESS TOI catalog data."""
    df = pd.read_csv(file_path)
    
    # TESS TOI column mapping
    column_mapping = {
        'tfopwg_disp': 'disposition',
        'pl_orbper': 'orbital_period', 
        'pl_trandep': 'transit_depth',
        'pl_trandur': 'transit_duration',
        'pl_rade': 'planet_radius',
        'pl_orbsmax': 'semi_major_axis',
        'pl_imppar': 'impact_parameter',
        'pl_eqt': 'equilibrium_temp',
        'st_teff': 'stellar_teff',
        'st_logg': 'stellar_logg',
        'st_met': 'stellar_metallicity',
        'st_rad': 'stellar_radius',
        'st_mass': 'stellar_mass'
    }
    
    # Apply column mapping
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)
    
    return df

def load_dataset(file_path, catalog_type='auto'):
    """
    Load exoplanet catalog data with automatic format detection.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    catalog_type : str
        Type of catalog ('kepler', 'tess', 'auto')
        
    Returns:
    --------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target labels
    numeric_cols : list
        List of numeric column names
    cat_cols : list
        List of categorical column names
    """
    
    # Auto-detect catalog type based on file name or columns
    if catalog_type == 'auto':
        if 'kepler' in file_path.lower() or 'koi' in file_path.lower():
            catalog_type = 'kepler'
        elif 'tess' in file_path.lower() or 'toi' in file_path.lower():
            catalog_type = 'tess'
        else:
            catalog_type = 'generic'
    
    # Load data based on catalog type
    if catalog_type == 'kepler':
        df = load_kepler_dr25(file_path)
    elif catalog_type == 'tess':
        df = load_tess_toi(file_path)
    else:
        df = pd.read_csv(file_path)
    
    logger.info(f"Loaded {len(df)} rows from {file_path}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Process target variable
    if 'disposition' not in df.columns:
        raise ValueError("Target column 'disposition' not found. Please rename your target column to 'disposition'")
    
    # Clean and map disposition values
    y_raw = df['disposition'].astype(str).str.upper().str.strip()
    y_mapped = y_raw.map(DISPOSITION_MAP)
    
    # Keep only valid dispositions
    valid_mask = y_mapped.isin(DISPOSITION_MAP.values())
    df = df.loc[valid_mask].copy()
    y = y_mapped.loc[valid_mask]
    
    logger.info(f"After filtering valid dispositions: {len(df)} rows")
    logger.info(f"Class distribution:\n{y.value_counts()}")
    
    # Feature selection
    X, numeric_cols, cat_cols = select_features(df)
    
    return X, y, numeric_cols, cat_cols

def select_features(df):
    """
    Select and engineer features for the model.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    X : pd.DataFrame
        Feature matrix
    numeric_cols : list
        List of numeric column names
    cat_cols : list
        List of categorical column names
    """
    
    # Define preferred features (in order of importance)
    preferred_numeric = [
        'orbital_period', 'transit_depth', 'transit_duration', 
        'planet_radius', 'semi_major_axis', 'impact_parameter',
        'equilibrium_temp', 'insolation', 'rp_rs', 'distance_over_rstar',
        'stellar_teff', 'stellar_logg', 'stellar_metallicity', 
        'stellar_radius', 'stellar_mass', 'stellar_density',
        'disposition_score', 'snr'
    ]
    
    preferred_categorical = [
        'fit_type', 'pipeline', 'method'
    ]
    
    # Get all numeric columns
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target and ID columns
    exclude_patterns = ['disposition', 'id', 'name', 'host', 'kepid', 'tic']
    all_numeric = [col for col in all_numeric if not any(pattern in col.lower() for pattern in exclude_patterns)]
    
    # Remove potential leakage columns
    leakage_patterns = ['disp', 'label', 'flag', 'alert', 'comment', 'note']
    all_numeric = [col for col in all_numeric if not any(pattern in col.lower() for pattern in leakage_patterns)]
    
    # Prioritize preferred features that exist
    numeric_cols = []
    for col in preferred_numeric:
        if col in all_numeric:
            numeric_cols.append(col)
    
    # Add remaining numeric columns
    for col in all_numeric:
        if col not in numeric_cols:
            numeric_cols.append(col)
    
    # Get categorical columns
    all_categorical = df.select_dtypes(include=['object']).columns.tolist()
    all_categorical = [col for col in all_categorical if col != 'disposition']
    
    # Filter high-cardinality categoricals
    cat_cols = []
    for col in preferred_categorical:
        if col in all_categorical and df[col].nunique() <= 30:
            cat_cols.append(col)
    
    for col in all_categorical:
        if col not in cat_cols and df[col].nunique() <= 30:
            cat_cols.append(col)
    
    # Create feature matrix
    feature_cols = numeric_cols + cat_cols
    X = df[feature_cols].copy()
    
    logger.info(f"Selected {len(numeric_cols)} numeric features: {numeric_cols}")
    logger.info(f"Selected {len(cat_cols)} categorical features: {cat_cols}")
    
    return X, numeric_cols, cat_cols

def create_sample_data(output_path="data/sample_kepler.csv", n_samples=1000):
    """
    Create sample data for testing the model.
    
    Parameters:
    -----------
    output_path : str
        Path to save the sample data
    n_samples : int
        Number of samples to generate
    """
    
    np.random.seed(42)
    
    # Generate realistic exoplanet features
    data = {
        'orbital_period': np.random.lognormal(mean=2, sigma=1.5, size=n_samples),
        'transit_depth': np.random.lognormal(mean=-8, sigma=1, size=n_samples),
        'transit_duration': np.random.lognormal(mean=1, sigma=0.5, size=n_samples),
        'planet_radius': np.random.lognormal(mean=0.5, sigma=0.8, size=n_samples),
        'stellar_teff': np.random.normal(5500, 800, n_samples),
        'stellar_logg': np.random.normal(4.4, 0.3, n_samples),
        'stellar_metallicity': np.random.normal(0.0, 0.2, n_samples),
        'stellar_radius': np.random.lognormal(mean=0.0, sigma=0.3, size=n_samples),
        'stellar_mass': np.random.lognormal(mean=0.0, sigma=0.3, size=n_samples),
        'snr': np.random.lognormal(mean=2, sigma=0.8, size=n_samples),
        'impact_parameter': np.random.uniform(0, 1, n_samples),
        'disposition_score': np.random.uniform(0, 1, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate realistic disposition based on features
    # Higher scores and better SNR -> more likely confirmed
    # Poor metrics -> more likely false positive
    disposition_prob = (
        0.3 * (df['disposition_score'] > 0.7) +
        0.2 * (df['snr'] > 10) + 
        0.2 * (df['transit_depth'] > 100) +
        0.1 * (df['orbital_period'] < 100) +
        0.2 * np.random.random(n_samples)
    )
    
    dispositions = []
    for prob in disposition_prob:
        if prob > 0.8:
            dispositions.append('CONFIRMED')
        elif prob > 0.4:
            dispositions.append('CANDIDATE')
        else:
            dispositions.append('FALSE POSITIVE')
    
    df['disposition'] = dispositions
    
    # Add some realistic correlations and noise
    df.loc[df['disposition'] == 'FALSE POSITIVE', 'snr'] *= 0.5
    df.loc[df['disposition'] == 'CONFIRMED', 'disposition_score'] *= 1.2
    
    # Clip values to realistic ranges
    df['disposition_score'] = np.clip(df['disposition_score'], 0, 1)
    df['impact_parameter'] = np.clip(df['impact_parameter'], 0, 1)
    df['stellar_teff'] = np.clip(df['stellar_teff'], 3000, 8000)
    
    df.to_csv(output_path, index=False)
    logger.info(f"Created sample data with {n_samples} rows at {output_path}")
    logger.info(f"Class distribution:\n{df['disposition'].value_counts()}")
    
    return df

if __name__ == "__main__":
    # Create sample data for testing
    sample_df = create_sample_data()
    print("Sample data created successfully!")
    print(f"Shape: {sample_df.shape}")
    print(f"Columns: {list(sample_df.columns)}")
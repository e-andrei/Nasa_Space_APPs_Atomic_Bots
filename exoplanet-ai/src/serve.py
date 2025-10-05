# src/serve.py
import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from typing import Union, List, Dict, Optional
import argparse
from datetime import datetime

from model import ExoplanetClassifier
from data import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExoplanetPredictor:
    """
    Batch inference utility for exoplanet classification.
    """
    
    def __init__(self, model_path: str = "models/exoplanet_classifier.joblib", 
                 metadata_path: str = "models/metadata.json"):
        """
        Initialize the predictor.
        
        Parameters:
        -----------
        model_path : str
            Path to the trained model
        metadata_path : str
            Path to the model metadata
        """
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.classifier = None
        self.metadata = {}
        self.load_model()
    
    def load_model(self):
        """Load the trained model and metadata."""
        try:
            self.classifier = ExoplanetClassifier.load(self.model_path, self.metadata_path)
            
            # Load metadata
            if Path(self.metadata_path).exists():
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_single(self, features: Dict[str, float]) -> Dict:
        """
        Make prediction for a single sample.
        
        Parameters:
        -----------
        features : dict
            Dictionary of feature values
            
        Returns:
        --------
        result : dict
            Prediction result with probabilities
        """
        if self.classifier is None:
            raise ValueError("Model not loaded")
        
        # Convert to DataFrame
        sample_df = pd.DataFrame([features])
        
        try:
            # Make prediction
            prediction = self.classifier.predict(sample_df)[0]
            probabilities = self.classifier.predict_proba(sample_df)[0]
            
            # Create result
            result = {
                'prediction': prediction,
                'confidence': float(max(probabilities)),
                'probabilities': {
                    cls: float(prob) 
                    for cls, prob in zip(self.classifier.pipeline.classes_, probabilities)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(self, input_path: str, output_path: Optional[str] = None, 
                      include_probabilities: bool = True) -> pd.DataFrame:
        """
        Make predictions for a batch of samples.
        
        Parameters:
        -----------
        input_path : str
            Path to input CSV file
        output_path : str, optional
            Path to save results CSV
        include_probabilities : bool
            Whether to include probability columns
            
        Returns:
        --------
        results_df : pd.DataFrame
            DataFrame with predictions and probabilities
        """
        if self.classifier is None:
            raise ValueError("Model not loaded")
        
        logger.info(f"Loading data from {input_path}")
        
        # Load data
        try:
            df = pd.read_csv(input_path)
            logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
        
        # Make predictions
        try:
            logger.info("Making predictions...")
            predictions = self.classifier.predict(df)
            probabilities = self.classifier.predict_proba(df)
            
            # Create results DataFrame
            results_df = df.copy()
            results_df['prediction'] = predictions
            results_df['confidence'] = np.max(probabilities, axis=1)
            
            # Add probability columns if requested
            if include_probabilities:
                for i, cls in enumerate(self.classifier.pipeline.classes_):
                    results_df[f'prob_{cls}'] = probabilities[:, i]
            
            # Add metadata
            results_df['model_version'] = self.metadata.get('model_type', 'unknown')
            results_df['prediction_timestamp'] = datetime.now().isoformat()
            
            logger.info(f"Predictions completed for {len(results_df)} samples")
            
            # Summary statistics
            pred_counts = pd.Series(predictions).value_counts()
            logger.info("Prediction distribution:")
            for cls, count in pred_counts.items():
                logger.info(f"  {cls}: {count} ({count/len(predictions)*100:.1f}%)")
            
            # Save results if output path provided
            if output_path:
                self.save_results(results_df, output_path)
            
            return results_df
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    def save_results(self, results_df: pd.DataFrame, output_path: str):
        """
        Save prediction results to file.
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Results DataFrame
        output_path : str
            Output file path
        """
        try:
            # Create output directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save results
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def evaluate_batch(self, input_path: str, truth_column: str = 'disposition') -> Dict:
        """
        Evaluate model performance on a labeled dataset.
        
        Parameters:
        -----------
        input_path : str
            Path to labeled CSV file
        truth_column : str
            Name of the ground truth column
            
        Returns:
        --------
        metrics : dict
            Evaluation metrics
        """
        if self.classifier is None:
            raise ValueError("Model not loaded")
        
        logger.info(f"Evaluating model on {input_path}")
        
        # Load data
        try:
            df = pd.read_csv(input_path)
            
            if truth_column not in df.columns:
                raise ValueError(f"Truth column '{truth_column}' not found in data")
                
        except Exception as e:
            logger.error(f"Failed to load evaluation data: {e}")
            raise
        
        # Separate features and labels
        y_true = df[truth_column]
        X = df.drop(columns=[truth_column])
        
        # Make predictions
        try:
            predictions = self.classifier.predict(X)
            probabilities = self.classifier.predict_proba(X)
            
            # Calculate metrics
            from sklearn.metrics import (
                accuracy_score, classification_report, 
                confusion_matrix, average_precision_score
            )
            
            accuracy = accuracy_score(y_true, predictions)
            report = classification_report(y_true, predictions, output_dict=True)
            cm = confusion_matrix(y_true, predictions)
            
            # Per-class average precision
            ap_scores = {}
            classes = self.classifier.pipeline.classes_
            for i, cls in enumerate(classes):
                if cls in y_true.values:
                    y_binary = (y_true == cls).astype(int)
                    if len(np.unique(y_binary)) > 1:
                        ap_scores[cls] = average_precision_score(y_binary, probabilities[:, i])
            
            metrics = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'average_precision_scores': ap_scores,
                'macro_average_precision': np.mean(list(ap_scores.values())),
                'n_samples': len(y_true),
                'classes': classes.tolist()
            }
            
            logger.info(f"Evaluation completed:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Macro AP: {metrics['macro_average_precision']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
        --------
        info : dict
            Model information
        """
        if self.classifier is None:
            return {"status": "no_model_loaded"}
        
        info = {
            "status": "loaded",
            "model_path": self.model_path,
            "metadata_path": self.metadata_path,
            "classes": getattr(self.classifier.pipeline, 'classes_', []).tolist(),
            "metadata": self.metadata
        }
        
        return info
    
    def create_prediction_template(self, output_path: str = "prediction_template.csv"):
        """
        Create a CSV template for making predictions.
        
        Parameters:
        -----------
        output_path : str
            Path to save the template
        """
        if self.classifier is None:
            raise ValueError("Model not loaded")
        
        # Get feature names from metadata
        numeric_features = self.metadata.get('numeric_features', [])
        categorical_features = self.metadata.get('categorical_features', [])
        
        all_features = numeric_features + categorical_features
        
        if not all_features:
            logger.warning("No feature information available in metadata")
            # Create basic template with common exoplanet features
            all_features = [
                'orbital_period', 'transit_depth', 'transit_duration',
                'planet_radius', 'stellar_teff', 'stellar_radius',
                'stellar_mass', 'snr', 'impact_parameter'
            ]
        
        # Create template DataFrame with example values
        template_data = {}
        for feature in all_features:
            if feature in numeric_features or feature in [
                'orbital_period', 'transit_depth', 'transit_duration',
                'planet_radius', 'stellar_teff', 'stellar_radius',
                'stellar_mass', 'snr', 'impact_parameter'
            ]:
                # Numeric features - provide example values
                if 'period' in feature.lower():
                    template_data[feature] = [10.5]
                elif 'depth' in feature.lower():
                    template_data[feature] = [1000.0]
                elif 'duration' in feature.lower():
                    template_data[feature] = [3.5]
                elif 'radius' in feature.lower():
                    template_data[feature] = [1.2]
                elif 'teff' in feature.lower():
                    template_data[feature] = [5500]
                elif 'mass' in feature.lower():
                    template_data[feature] = [1.0]
                elif 'snr' in feature.lower():
                    template_data[feature] = [15.0]
                elif 'impact' in feature.lower():
                    template_data[feature] = [0.3]
                else:
                    template_data[feature] = [0.0]
            else:
                # Categorical features
                template_data[feature] = ["example_value"]
        
        template_df = pd.DataFrame(template_data)
        template_df.to_csv(output_path, index=False)
        
        logger.info(f"Prediction template saved to {output_path}")
        logger.info(f"Template includes {len(all_features)} features")

def main():
    """Command-line interface for batch predictions."""
    parser = argparse.ArgumentParser(description='Exoplanet Classification Batch Prediction')
    
    parser.add_argument('command', choices=['predict', 'evaluate', 'info', 'template'],
                       help='Command to execute')
    parser.add_argument('--input', '-i', help='Input CSV file path')
    parser.add_argument('--output', '-o', help='Output CSV file path')
    parser.add_argument('--model', '-m', default='models/exoplanet_classifier.joblib',
                       help='Path to model file')
    parser.add_argument('--metadata', default='models/metadata.json',
                       help='Path to metadata file')
    parser.add_argument('--truth-column', default='disposition',
                       help='Name of ground truth column for evaluation')
    parser.add_argument('--no-probabilities', action='store_true',
                       help='Skip probability columns in output')
    
    args = parser.parse_args()
    
    # Initialize predictor
    try:
        predictor = ExoplanetPredictor(args.model, args.metadata)
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        return 1
    
    # Execute command
    try:
        if args.command == 'predict':
            if not args.input:
                logger.error("Input file required for prediction")
                return 1
            
            output_path = args.output or f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            results = predictor.predict_batch(
                args.input, 
                output_path,
                include_probabilities=not args.no_probabilities
            )
            
            print(f"Predictions completed. Results saved to: {output_path}")
            print(f"Processed {len(results)} samples")
            
        elif args.command == 'evaluate':
            if not args.input:
                logger.error("Input file required for evaluation")
                return 1
            
            metrics = predictor.evaluate_batch(args.input, args.truth_column)
            
            print("Evaluation Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Macro Average Precision: {metrics['macro_average_precision']:.4f}")
            print(f"Samples: {metrics['n_samples']}")
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(metrics, f, indent=2, default=str)
                print(f"Detailed metrics saved to: {args.output}")
                
        elif args.command == 'info':
            info = predictor.get_model_info()
            print("Model Information:")
            print(json.dumps(info, indent=2, default=str))
            
        elif args.command == 'template':
            output_path = args.output or "prediction_template.csv"
            predictor.create_prediction_template(output_path)
            print(f"Prediction template created: {output_path}")
            
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
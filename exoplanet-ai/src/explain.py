# src/explain.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelExplainer:
    """
    Model explainability utilities for exoplanet classification.
    """
    
    def __init__(self, model, X_train, feature_names=None):
        """
        Initialize the explainer.
        
        Parameters:
        -----------
        model : sklearn.pipeline.Pipeline
            Trained model pipeline
        X_train : pd.DataFrame
            Training data for SHAP background
        feature_names : list, optional
            Feature names for visualization
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or list(X_train.columns)
        self.shap_explainer = None
        self.permutation_importance_ = None
        
    def compute_permutation_importance(self, X_test, y_test, n_repeats=10, random_state=42):
        """
        Compute permutation importance for features.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test data
        y_test : pd.Series
            Test labels
        n_repeats : int
            Number of permutation repeats
        random_state : int
            Random state for reproducibility
            
        Returns:
        --------
        importance_df : pd.DataFrame
            DataFrame with feature importance scores
        """
        logger.info("Computing permutation importance...")
        
        # Compute permutation importance
        perm_importance = permutation_importance(
            self.model, X_test, y_test,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        self.permutation_importance_ = importance_df
        logger.info("Permutation importance computed successfully")
        
        return importance_df
    
    def plot_permutation_importance(self, top_n=20, figsize=(10, 8), save_path=None):
        """
        Plot permutation importance.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to plot
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        if self.permutation_importance_ is None:
            raise ValueError("Permutation importance not computed. Run compute_permutation_importance first.")
        
        # Select top features
        top_features = self.permutation_importance_.head(top_n)
        
        # Create plot
        plt.figure(figsize=figsize)
        plt.barh(range(len(top_features)), top_features['importance_mean'], 
                xerr=top_features['importance_std'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Permutation Importance')
        plt.title(f'Top {top_n} Feature Importance (Permutation)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Permutation importance plot saved to {save_path}")
        
        plt.show()
        
    def setup_shap_explainer(self, explainer_type='tree', max_evals=100):
        """
        Set up SHAP explainer.
        
        Parameters:
        -----------
        explainer_type : str
            Type of SHAP explainer ('tree', 'linear', 'kernel')
        max_evals : int
            Maximum evaluations for kernel explainer
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not available. Install with: pip install shap")
        
        logger.info(f"Setting up SHAP explainer (type: {explainer_type})")
        
        # Get the actual classifier from the pipeline
        if hasattr(self.model, 'named_steps'):
            classifier = self.model.named_steps.get('classifier') or self.model.named_steps.get('clf')
        else:
            classifier = self.model
            
        if explainer_type == 'tree':
            # For tree-based models like XGBoost
            if hasattr(classifier, 'predict_proba'):
                self.shap_explainer = shap.TreeExplainer(classifier)
            else:
                raise ValueError("Tree explainer requires a tree-based model")
                
        elif explainer_type == 'linear':
            # For linear models
            self.shap_explainer = shap.LinearExplainer(classifier, self.X_train)
            
        elif explainer_type == 'kernel':
            # Model-agnostic explainer (slower but works with any model)
            background = shap.sample(self.X_train, min(100, len(self.X_train)))
            self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, background)
            
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")
            
        logger.info("SHAP explainer set up successfully")
        
    def explain_predictions(self, X_explain, max_samples=100):
        """
        Generate SHAP explanations for predictions.
        
        Parameters:
        -----------
        X_explain : pd.DataFrame
            Data to explain
        max_samples : int
            Maximum number of samples to explain
            
        Returns:
        --------
        shap_values : np.ndarray
            SHAP values for the predictions
        """
        if self.shap_explainer is None:
            self.setup_shap_explainer()
            
        # Limit samples for performance
        if len(X_explain) > max_samples:
            X_explain = X_explain.sample(n=max_samples, random_state=42)
            logger.info(f"Limited explanation to {max_samples} samples")
            
        logger.info(f"Generating SHAP explanations for {len(X_explain)} samples...")
        
        # Transform data through pipeline preprocessing
        if hasattr(self.model, 'named_steps') and 'preprocessor' in self.model.named_steps:
            X_transformed = self.model.named_steps['preprocessor'].transform(X_explain)
        else:
            X_transformed = X_explain
            
        # Generate SHAP values
        shap_values = self.shap_explainer.shap_values(X_transformed)
        
        logger.info("SHAP explanations generated successfully")
        return shap_values
    
    def plot_shap_summary(self, X_explain, max_samples=100, plot_type='dot', save_path=None):
        """
        Create SHAP summary plot.
        
        Parameters:
        -----------
        X_explain : pd.DataFrame
            Data to explain
        max_samples : int
            Maximum number of samples to explain
        plot_type : str
            Type of plot ('dot', 'bar', 'violin')
        save_path : str, optional
            Path to save the plot
        """
        shap_values = self.explain_predictions(X_explain, max_samples)
        
        # Transform data for plotting
        if hasattr(self.model, 'named_steps') and 'preprocessor' in self.model.named_steps:
            X_transformed = self.model.named_steps['preprocessor'].transform(X_explain.head(max_samples))
            feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
        else:
            X_transformed = X_explain.head(max_samples).values
            feature_names = X_explain.columns
        
        plt.figure(figsize=(12, 8))
        
        if isinstance(shap_values, list):
            # Multi-class case
            for i, class_shap_values in enumerate(shap_values):
                plt.subplot(len(shap_values), 1, i+1)
                shap.summary_plot(
                    class_shap_values, X_transformed, 
                    feature_names=feature_names,
                    plot_type=plot_type, show=False
                )
                plt.title(f'Class {i} SHAP Summary')
        else:
            # Binary or single output case
            shap.summary_plot(
                shap_values, X_transformed,
                feature_names=feature_names,
                plot_type=plot_type, show=False
            )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {save_path}")
            
        plt.tight_layout()
        plt.show()
        
    def plot_feature_importance_comparison(self, X_test, y_test, top_n=15, save_path=None):
        """
        Compare different feature importance methods.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test data
        y_test : pd.Series
            Test labels
        top_n : int
            Number of top features to show
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Permutation importance
        if self.permutation_importance_ is None:
            self.compute_permutation_importance(X_test, y_test)
            
        perm_top = self.permutation_importance_.head(top_n)
        axes[0].barh(range(len(perm_top)), perm_top['importance_mean'])
        axes[0].set_yticks(range(len(perm_top)))
        axes[0].set_yticklabels(perm_top['feature'])
        axes[0].set_xlabel('Permutation Importance')
        axes[0].set_title('Permutation Importance')
        axes[0].invert_yaxis()
        
        # Tree feature importance (if available)
        try:
            if hasattr(self.model, 'named_steps'):
                classifier = self.model.named_steps.get('classifier') or self.model.named_steps.get('clf')
            else:
                classifier = self.model
                
            if hasattr(classifier, 'feature_importances_'):
                tree_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': classifier.feature_importances_
                }).sort_values('importance', ascending=False).head(top_n)
                
                axes[1].barh(range(len(tree_importance)), tree_importance['importance'])
                axes[1].set_yticks(range(len(tree_importance)))
                axes[1].set_yticklabels(tree_importance['feature'])
                axes[1].set_xlabel('Tree Feature Importance')
                axes[1].set_title('XGBoost Feature Importance')
                axes[1].invert_yaxis()
            else:
                axes[1].text(0.5, 0.5, 'Tree importance\nnot available', 
                           ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('Tree Feature Importance (N/A)')
                
        except Exception as e:
            logger.warning(f"Could not plot tree importance: {e}")
            axes[1].text(0.5, 0.5, f'Error: {str(e)}', 
                       ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance comparison saved to {save_path}")
            
        plt.show()
        
    def explain_single_prediction(self, sample, class_names=None):
        """
        Explain a single prediction in detail.
        
        Parameters:
        -----------
        sample : pd.Series or pd.DataFrame
            Single sample to explain
        class_names : list, optional
            Names of the classes
            
        Returns:
        --------
        explanation : dict
            Detailed explanation of the prediction
        """
        if isinstance(sample, pd.Series):
            sample = sample.to_frame().T
            
        # Get prediction
        prediction = self.model.predict(sample)[0]
        probabilities = self.model.predict_proba(sample)[0]
        
        explanation = {
            'prediction': prediction,
            'probabilities': dict(zip(self.model.classes_, probabilities)),
            'top_features': None
        }
        
        # Add class names if provided
        if class_names:
            explanation['class_names'] = dict(zip(self.model.classes_, class_names))
            
        # Get feature contributions (permutation-based)
        if self.permutation_importance_ is not None:
            top_features = self.permutation_importance_.head(10)
            explanation['top_features'] = top_features.to_dict('records')
            
        return explanation
        
    def generate_explanation_report(self, X_test, y_test, output_dir="explanations"):
        """
        Generate a comprehensive explanation report.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test data
        y_test : pd.Series
            Test labels
        output_dir : str
            Directory to save explanation artifacts
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Generating comprehensive explanation report...")
        
        # 1. Permutation importance
        self.compute_permutation_importance(X_test, y_test)
        self.plot_permutation_importance(save_path=f"{output_dir}/permutation_importance.png")
        
        # 2. Feature importance comparison
        self.plot_feature_importance_comparison(
            X_test, y_test, 
            save_path=f"{output_dir}/feature_importance_comparison.png"
        )
        
        # 3. SHAP analysis (if available)
        if SHAP_AVAILABLE:
            try:
                self.setup_shap_explainer()
                self.plot_shap_summary(
                    X_test, max_samples=200,
                    save_path=f"{output_dir}/shap_summary.png"
                )
            except Exception as e:
                logger.warning(f"SHAP analysis failed: {e}")
                
        # 4. Save importance data
        if self.permutation_importance_ is not None:
            self.permutation_importance_.to_csv(f"{output_dir}/permutation_importance.csv", index=False)
            
        logger.info(f"Explanation report saved to {output_dir}/")

def main():
    """Example usage of the explainer."""
    from model import ExoplanetClassifier
    from data import load_dataset
    
    # Load data and model
    X, y, _, _ = load_dataset("data/sample_kepler.csv")
    classifier = ExoplanetClassifier.load()
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create explainer
    explainer = ModelExplainer(classifier.pipeline, X_train)
    
    # Generate explanations
    explainer.generate_explanation_report(X_test, y_test)
    
    print("Explanation report generated!")

if __name__ == "__main__":
    main()
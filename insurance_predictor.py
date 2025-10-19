import uuid
import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class InsurancePredictor:
    """
    A flexible prediction system for insurance renewal prediction that can work with multiple models.
    """
    
    AVAILABLE_MODELS = {
        'XGB': 'models/xgboost_model.pkl',
        'NN': 'models/neural_network_model.pkl',
        'TAB': 'models/tabnet_model.pkl',
        'BRF': 'models/balanced_rf_model.pkl',
        'EEC': 'models/easy_ensemble_model.pkl',
        'LR': 'models/logistic_regression_model.pkl'
    }
    
    def __init__(self, model_type='XGB'):
        """
        Initialize predictor with specified model type.
        
        Args:
            model_type: Type of model to use ('XGB', 'NN', 'TAB', 'BRF', 'EEC', 'LR')
        """
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model type must be one of {list(self.AVAILABLE_MODELS.keys())}")
            
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        
        # Load model and preprocessor
        self._load_model_and_preprocessor()
        
    def _load_model_and_preprocessor(self):
        """Load the model and preprocessor from saved files"""
        model_path = self.AVAILABLE_MODELS[self.model_type]
        
        # Load model artifacts
        artifacts = joblib.load(model_path)
        self.model = artifacts['model']
        self.preprocessor = artifacts['preprocessor']
        self.feature_names = artifacts['feature_names']
        
        # Load feature config
        with open('models/feature_config.json', 'r') as f:
            self.feature_config = json.load(f)
    
    def switch_model(self, model_type):
        """
        Switch to a different model type.
        
        Args:
            model_type: New model type to use
        """
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model type must be one of {list(self.AVAILABLE_MODELS.keys())}")
        
        self.model_type = model_type
        self._load_model_and_preprocessor()
    
    def _validate_input(self, data):
        """Validate input data format and features"""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        missing_cols = set(self.feature_names) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate data types
        for col, dtype in self.feature_config['dtypes'].items():
            if col in data.columns and str(data[col].dtype) != dtype:
                try:
                    data[col] = data[col].astype(dtype)
                except:
                    raise ValueError(f"Column {col} must be of type {dtype}")
    
    def predict(self, input_data: pd.DataFrame, output_dir: str = None) -> dict:
        """
        Make predictions and generate visualizations for new data.
        
        Args:
            input_data: DataFrame with insurance features
            output_dir: Optional directory for outputs (creates UUID-based dir if None)
            
        Returns:
            dict containing:
            - predictions: DataFrame with predictions and probabilities
            - visualizations: dict of plot paths
            - shap_values: Feature importance explanations (if model supports it)
        """
        # Input validation
        self._validate_input(input_data)
        
        # Create output directory with UUID if not provided
        if output_dir is None:
            output_dir = f"Visualizations/Predictions_{uuid.uuid4().hex[:8]}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Preprocess data
        X_processed = self.preprocessor.transform(input_data)
        
        # Generate predictions
        predictions = self.model.predict(X_processed)
        
        # Get probabilities (handle different model interfaces)
        try:
            probabilities = self.model.predict_proba(X_processed)[:,1]
        except:
            # Fallback for models that don't support predict_proba
            probabilities = predictions.astype(float)
        
        # Create prediction DataFrame
        results_df = pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities
        }, index=input_data.index)
        
        # Generate visualizations
        viz_paths = self._generate_visualizations(
            input_data, predictions, probabilities, output_dir)
        
        result = {
            'predictions': results_df,
            'visualizations': viz_paths,
        }
        
        # Add SHAP values if supported
        if self.model_type in ['XGB', 'BRF', 'TAB']:  # Tree-based models support SHAP
            shap_values = self._generate_shap_explanations(
                X_processed, input_data.columns, output_dir)
            result['shap_values'] = shap_values
        
        return result

    def _generate_visualizations(self, data, preds, probs, output_dir):
        """Generate streamlit-friendly visualizations"""
        viz_paths = {}
        
        # 1. Prediction probability distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(probs, bins=50)
        plt.title('Prediction Probability Distribution')
        plt.xlabel('Probability of Renewal')
        plt.ylabel('Count')
        path = f"{output_dir}/prediction_dist.png"
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()
        viz_paths['prediction_dist'] = path
        
        # 2. Prediction counts
        plt.figure(figsize=(8, 6))
        sns.countplot(x=preds)
        plt.title('Prediction Distribution')
        plt.xlabel('Predicted Renewal (0=No, 1=Yes)')
        plt.ylabel('Count')
        path = f"{output_dir}/prediction_counts.png"
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()
        viz_paths['prediction_counts'] = path
        
        return viz_paths

    def _generate_shap_explanations(self, X_processed, feature_names, output_dir):
        """Generate SHAP values and plots for feature importance"""
        try:
            # For tree-based models
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_processed)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
            
            # Save SHAP summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, 
                X_processed,
                feature_names=feature_names,
                show=False
            )
            plt.title('SHAP Feature Importance')
            path = f"{output_dir}/shap_summary.png"
            plt.savefig(path, bbox_inches='tight', dpi=300)
            plt.close()
            
            # Calculate and save feature importance scores
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(shap_values).mean(0)
            }).sort_values('importance', ascending=False)
            
            feature_importance.to_csv(f"{output_dir}/feature_importance.csv", index=False)
            
            return {
                'values': shap_values,
                'plot_path': path,
                'importance_csv': f"{output_dir}/feature_importance.csv"
            }
        except Exception as e:
            print(f"SHAP analysis failed: {str(e)}")
            return None
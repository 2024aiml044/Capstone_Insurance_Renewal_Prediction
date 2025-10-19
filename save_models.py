# Save models and preprocessing pipeline
import joblib
import json
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save feature configuration
feature_config = {
    'dtypes': {col: str(df[col].dtype) for col in df.columns},
    'numeric_features': numeric_features,
    'categorical_features': categorical_features
}

with open('models/feature_config.json', 'w') as f:
    json.dump(feature_config, f)

# Function to save model artifacts
def save_model_artifacts(model, model_name):
    artifacts = {
        'model': model,
        'preprocessor': preprocessor,
        'feature_names': df.columns.tolist()
    }
    joblib.dump(artifacts, f'models/{model_name}_model.pkl')
    print(f"Saved {model_name} model artifacts")

# Save all models
if 'lr' in locals():
    save_model_artifacts(lr, 'logistic_regression')
    
if 'best_xgb' in locals():
    save_model_artifacts(best_xgb, 'xgboost')
    
if 'nn_model' in locals():
    save_model_artifacts(nn_model, 'neural_network')
    
if 'brf' in locals():
    save_model_artifacts(brf, 'balanced_rf')
    
if 'eec' in locals():
    save_model_artifacts(eec, 'easy_ensemble')
    
if 'tabnet' in locals():
    save_model_artifacts(tabnet, 'tabnet')

print("All model artifacts saved to models/ directory")
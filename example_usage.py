import pandas as pd
from insurance_predictor import InsurancePredictor

# Example usage
def main():
    # Initialize predictor with default model (XGBoost)
    predictor = InsurancePredictor()
    
    # Load some test data
    test_data = pd.read_csv('Dataset/test_66516Ee.csv')
    
    # Make predictions
    results = predictor.predict(test_data)
    
    # Print predictions
    print("\nPrediction Results:")
    print(results['predictions'].head())
    
    # Print visualization paths
    print("\nGenerated Visualizations:")
    for name, path in results['visualizations'].items():
        print(f"{name}: {path}")
    
    # Switch to a different model
    print("\nSwitching to Neural Network model...")
    predictor.switch_model('NN')
    
    # Make predictions with new model
    results_nn = predictor.predict(test_data, output_dir='Visualizations/nn_predictions')
    
    print("\nNeural Network Prediction Results:")
    print(results_nn['predictions'].head())

if __name__ == "__main__":
    main()
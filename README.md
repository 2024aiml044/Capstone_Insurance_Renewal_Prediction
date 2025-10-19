# Insurance Renewal Prediction System

This project provides an end-to-end solution for predicting insurance renewal probabilities, including model training, evaluation, and a Streamlit-based prediction interface.

## Project Structure

```
├── Dataset/
│   ├── train_ZoGVYWq.csv
│   └── test_66516Ee.csv
├── models/                    # Saved model artifacts
│   ├── xgboost_model.pkl
│   ├── neural_network_model.pkl
│   └── feature_config.json
├── Visualizations/           # Output visualizations and results
├── final_insurance_notebook_full_analysis_original.ipynb  # Training notebook
├── insurance_predictor.py    # Prediction module
├── main.py                   # Streamlit interface
└── example_usage.py         # Example code
```

## Setup and Deployment

### 1. Environment Setup

```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Model Training

Before using the prediction interface, you must train and save the models:

1. Open `final_insurance_notebook_full_analysis_original.ipynb`
2. Run all cells in the notebook
3. This will:
   - Train all models
   - Save models to the `models/` directory
   - Generate initial visualizations and metrics

### 3. Running the Streamlit Interface

```bash
streamlit run main.py
```

The interface will be available at `http://localhost:8501`

## Usage Guide

### Model Types Available

1. **XGBoost (XGB)**
   - Best overall performance
   - Provides feature importance
   - Recommended for most cases

2. **Neural Network (NN)**
   - Good for complex patterns
   - Fast inference
   - No feature importance

3. **TabNet (TAB)**
   - Interpretable deep learning
   - Provides feature importance
   - Good balance of performance and interpretability

4. **Balanced Random Forest (BRF)**
   - Handles imbalanced data well
   - Provides feature importance
   - More robust to outliers

5. **Easy Ensemble Classifier (EEC)**
   - Robust to noise
   - Good for imbalanced datasets
   - Ensemble of AdaBoost learners

6. **Logistic Regression (LR)**
   - Simple baseline model
   - Most interpretable
   - Fast training and inference

### Using the Streamlit Interface

1. **Select Model**
   - Choose model type from sidebar
   - Each model has different strengths

2. **Upload Data**
   - Upload CSV file with required features
   - Data must match training data format
   - Missing columns will cause errors

3. **View Results**
   - Predictions tab shows individual predictions
   - Visualizations tab shows distributions
   - Feature Importance tab (for supported models)
   - Download tab for saving results

4. **Experiment Tracking**
   - Each prediction run gets unique ID
   - Results saved in Visualizations/Predictions_[ID]
   - All artifacts are preserved for reference

### Input Data Requirements

Your input CSV must contain these columns:
- [List of required columns from feature_config.json]
- Data types must match training data
- No missing values allowed

### Output Directory Structure

Each prediction run creates:
```
Visualizations/
└── Predictions_[timestamp]_[uuid]/
    ├── prediction_dist.png
    ├── prediction_counts.png
    ├── shap_summary.png (if applicable)
    ├── feature_importance.csv
    └── predictions.csv
```

## Development and Extension

### Adding New Models

1. Train new model in notebook
2. Save model artifacts to `models/`
3. Add model type to `InsurancePredictor.AVAILABLE_MODELS`
4. Update Streamlit interface if needed

### Customizing Visualizations

Modify `_generate_visualizations()` in `insurance_predictor.py` to:
- Add new plots
- Modify existing visualizations
- Change output formats

## Troubleshooting

Common issues and solutions:

1. **ModuleNotFoundError**
   - Ensure all requirements are installed
   - Check virtual environment is activated

2. **Model Loading Errors**
   - Verify notebook was run completely
   - Check models/ directory exists with .pkl files

3. **Input Data Errors**
   - Compare columns with training data
   - Check data types match
   - Ensure no missing values

4. **Visualization Errors**
   - Check write permissions in Visualizations/
   - Verify matplotlib backend compatibility

## Performance Considerations

- First prediction may be slow due to model loading
- Subsequent predictions are faster
- Large datasets may require more memory
- Consider batch processing for very large files

## Security Notes

- No sensitive data is stored
- Each prediction run is isolated
- Results are saved locally only
- Clean up old prediction directories regularly

## Support

For issues and questions:
1. Check troubleshooting section
2. Verify input data format
3. Check logs in Visualizations/
4. Raise issue on repository
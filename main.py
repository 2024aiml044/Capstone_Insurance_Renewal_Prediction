# IMPORTANT: This is the prediction interface only!
# The model training is done in final_insurance_notebook_full_analysis_original.ipynb
# This file expects trained models to exist in the models/ directory
# Run the notebook first to train and save models before using this Streamlit interface

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import streamlit as st
from insurance_predictor import InsurancePredictor  # Handles loading saved models
import uuid
from datetime import datetime

# Streamlit page config
st.set_page_config(
    page_title="Insurance Renewal Prediction",
    page_icon="🎯",
    layout="wide"
)

def load_and_preprocess_data(file):
    """Load and preprocess uploaded data"""
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def main():
    st.title("Insurance Renewal Prediction")
    st.write("Upload data to predict insurance renewal probability")

    # Sidebar for model selection
    with st.sidebar:
        st.header("Model Configuration")
        model_type = st.selectbox(
            "Select Model",
            ["XGB", "NN", "TAB", "BRF", "EEC", "LR"],
            help="XGB=XGBoost, NN=Neural Network, TAB=TabNet, BRF=Balanced Random Forest, EEC=Easy Ensemble, LR=Logistic Regression"
        )
        
        st.markdown("---")
        st.markdown("""
        ### Model Information
        - XGBoost: Best overall performance
        - Neural Network: Good for complex patterns
        - TabNet: Interpretable deep learning
        - Balanced RF: Handles imbalanced data
        - Easy Ensemble: Robust to noise
        - Logistic Regression: Simple baseline
        """)

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file for prediction", type=['csv'])
    
    if uploaded_file:
        # Load data
        data = load_and_preprocess_data(uploaded_file)
        
        if data is not None:
            st.write("### Data Preview")
            st.dataframe(data.head())
            
            # Create experiment directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_id = f"{timestamp}_{uuid.uuid4().hex[:6]}"
            output_dir = f"Visualizations/Predictions_{exp_id}"
            
            try:
                # Initialize predictor and make predictions
                predictor = InsurancePredictor(model_type=model_type)
                
                with st.spinner('Making predictions...'):
                    results = predictor.predict(data, output_dir=output_dir)
                
                # Display results in tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Predictions", 
                    "Visualizations", 
                    "Feature Importance",
                    "Download Results"
                ])
                
                with tab1:
                    st.write("### Prediction Results")
                    st.dataframe(results['predictions'])
                    
                    # Summary statistics
                    total = len(results['predictions'])
                    renewals = (results['predictions']['prediction'] == 1).sum()
                    non_renewals = (results['predictions']['prediction'] == 0).sum()
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Predictions", total)
                    col2.metric("Predicted Renewals", renewals)
                    col3.metric("Predicted Non-renewals", non_renewals)
                
                with tab2:
                    st.write("### Prediction Distribution")
                    st.image(results['visualizations']['prediction_dist'])
                    
                    st.write("### Prediction Counts")
                    st.image(results['visualizations']['prediction_counts'])
                
                with tab3:
                    if 'shap_values' in results:
                        st.write("### Feature Importance (SHAP)")
                        st.image(results['shap_values']['plot_path'])
                        
                        # Load and display feature importance table
                        importance_df = pd.read_csv(results['shap_values']['importance_csv'])
                        st.write("### Feature Importance Ranking")
                        st.dataframe(importance_df)
                    else:
                        st.info("SHAP feature importance not available for this model type")
                
                with tab4:
                    st.write("### Download Results")
                    
                    # Prepare predictions for download
                    predictions_csv = results['predictions'].to_csv(index=False)
                    st.download_button(
                        label="Download Predictions CSV",
                        data=predictions_csv,
                        file_name=f"predictions_{exp_id}.csv",
                        mime="text/csv"
                    )
                    
                    if 'shap_values' in results:
                        importance_csv = pd.read_csv(results['shap_values']['importance_csv']).to_csv(index=False)
                        st.download_button(
                            label="Download Feature Importance CSV",
                            data=importance_csv,
                            file_name=f"feature_importance_{exp_id}.csv",
                            mime="text/csv"
                        )
                
                # Display experiment ID for reference
                st.sidebar.success(f"Experiment ID: {exp_id}")
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.error("Please ensure your data has all required features and correct data types")

if __name__ == "__main__":
    main()
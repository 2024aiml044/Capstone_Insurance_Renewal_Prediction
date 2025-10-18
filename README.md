# Insurance Renewal Prediction Project

This repository contains a comprehensive analysis and prediction model for insurance policy renewals. The project includes exploratory data analysis, feature engineering, and machine learning models to predict whether a customer will renew their insurance policy.

## Main Files

- `insurance_renewal_prediction_final.ipynb`: The main notebook containing all analysis, including:
  - Exploratory Data Analysis (EDA)
  - Feature Engineering
  - Model Development (Logistic Regression, XGBoost, Neural Networks)
  - Model Comparison and Evaluation
  - Visualizations and Interpretations

## Directory Structure

- `Dataset/`: Contains training and test datasets
- `Visualizations/`: Contains generated plots and visualizations
  - `Results/`: Model prediction outputs and experiment results
- `Archive/`: Contains previous iterations of the notebook (v1-v4)

## Requirements

The project uses Python with the following main libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- tensorflow
- shap

## Project Structure

1. **Data Loading and Cleaning**
   - Missing value analysis and handling
   - Data type conversions

2. **Exploratory Data Analysis**
   - Feature distributions
   - Target variable analysis
   - Correlation studies
   - Feature importance analysis

3. **Feature Engineering**
   - Age conversion (days to years)
   - Late payment aggregations
   - Ratio calculations
   - Categorical encoding

4. **Modeling**
   - Logistic Regression baseline
   - XGBoost implementation
   - Neural Network development
   - Model comparison and evaluation

5. **Results**
   - Performance metrics
   - Feature importance visualization
   - SHAP value analysis
   - Class imbalance handling results
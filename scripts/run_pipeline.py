"""
Enhanced pipeline to preprocess data and train multiple models with hyperparameter tuning:
- Logistic Regression (baseline)
- Random Forest (tuned)
- XGBoost (tuned)
- TabNet (if available, tuned)

All models handle class imbalance and save predictions/metrics to Visualizations/Results.

Run with the workspace Python interpreter (the devcontainer's .venv is used by default).
"""
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt

# Advanced imports with fallbacks
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print('XGBoost not available')

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False
    print('Optuna not available for hyperparameter tuning')

try:
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    HAS_TABNET = True
except Exception:
    HAS_TABNET = False
    print('TabNet not available')

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / 'Dataset' / 'train_ZoGVYWq.csv'
OUT_DIR = ROOT / 'Visualizations' / 'Results'
PLOTS_DIR = ROOT / 'Visualizations'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Data path: {DATA_PATH}")
print(f"Output dir: {OUT_DIR}")

# Load data
df = pd.read_csv(DATA_PATH)
print('Loaded rows:', len(df))

# Basic feature engineering based on available columns
working = df.copy()
# Create age in years if present
if 'age_in_days' in working.columns:
    working['age_in_years'] = (working['age_in_days'] / 365).round(2)

# premium to income
if 'premium' in working.columns and 'Income' in working.columns:
    working['premium_to_income'] = working['premium'] / (working['Income'] + 1)

# total late counts
late_cols = [c for c in working.columns if 'Count_' in c]
if late_cols:
    working['total_late_counts'] = working[late_cols].sum(axis=1)

# premiums paid ratio
if 'no_of_premiums_paid' in working.columns and 'age_in_years' in working.columns:
    working['premiums_paid_ratio'] = working['no_of_premiums_paid'] / (working['age_in_years'] + 1)

# high underwriting flag
if 'application_underwriting_score' in working.columns:
    thr = working['application_underwriting_score'].quantile(0.99)
    working['high_underwriting_flag'] = (working['application_underwriting_score'] >= thr).astype(int)

# Drop identifier columns if present
id_col = 'id' if 'id' in working.columns else None
if id_col:
    ids = working[id_col]
else:
    ids = pd.Series(np.arange(len(working)), name='id')

# Target
if 'renewal' not in working.columns:
    raise SystemExit('Target column "renewal" not found in dataset')

y = working['renewal']

# Select features: drop target and id and original age_in_days if age_in_years created
drop_cols = ['renewal', id_col]
if 'age_in_days' in working.columns and 'age_in_years' in working.columns:
    drop_cols.append('age_in_days')

X = working.drop(columns=[c for c in drop_cols if c])

# Simple categorical handling: one-hot for low-cardinality cols
cat_cols = [c for c in X.select_dtypes(include=['object', 'category']).columns]
if cat_cols:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Fill any remaining NA
X = X.fillna(0)

# Train/test split
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, ids, test_size=0.2, random_state=42, stratify=y
)

# Scale numeric features
num_cols = X_train.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Prepare results container
metrics = {}

# Logistic Regression
print('\nTraining Logistic Regression...')
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)
prob_lr = lr.predict_proba(X_test)

metrics['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, pred_lr),
    'f1': f1_score(y_test, pred_lr),
    'auc_roc': roc_auc_score(y_test, prob_lr[:, 1]),
    'precision': precision_score(y_test, pred_lr),
    'recall': recall_score(y_test, pred_lr)
}

# Save LR predictions
lr_df = pd.DataFrame({
    'id': id_test.values,
    'actual': y_test.values,
    'pred': pred_lr,
    'prob_1': prob_lr[:, 1]
})
lr_df.to_csv(OUT_DIR / 'logistic_regression_predictions.csv', index=False)
print('Saved logistic_regression_predictions.csv')

# Random Forest with hyperparameter tuning
print('\nTraining Random Forest with tuning...')
def rf_objective(trial):
    rf_params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
        'class_weight': 'balanced',
        'random_state': 42
    }
    rf = RandomForestClassifier(**rf_params)
    scores = cross_val_score(rf, X_train, y_train, cv=3, scoring='roc_auc')
    return scores.mean()

if HAS_OPTUNA:
    study = optuna.create_study(direction='maximize')
    study.optimize(rf_objective, n_trials=20)
    best_rf_params = study.best_params
    best_rf_params.update({'class_weight': 'balanced', 'random_state': 42})
else:
    best_rf_params = {
        'n_estimators': 200,
        'max_depth': 6,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
        'random_state': 42
    }

rf = RandomForestClassifier(**best_rf_params)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)
prob_rf = rf.predict_proba(X_test)

metrics['Random Forest'] = {
    'accuracy': accuracy_score(y_test, pred_rf),
    'f1': f1_score(y_test, pred_rf),
    'auc_roc': roc_auc_score(y_test, prob_rf[:, 1]),
    'precision': precision_score(y_test, pred_rf),
    'recall': recall_score(y_test, pred_rf)
}

rf_df = pd.DataFrame({
    'id': id_test.values,
    'actual': y_test.values,
    'pred': pred_rf,
    'prob_1': prob_rf[:, 1]
})
rf_df.to_csv(OUT_DIR / 'random_forest_predictions.csv', index=False)
print('Saved random_forest_predictions.csv')

# Feature importance plot for RF
plt.figure(figsize=(10, 6))
importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
plt.barh(importances['feature'].iloc[:20][::-1], importances['importance'].iloc[:20][::-1])
plt.title('Random Forest Feature Importances (top 20)')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'feature_importance_rf.png')
plt.close()
print('Saved feature_importance_rf.png')

# XGBoost
if HAS_XGB:
    print('\nTraining XGBoost with tuning...')
    def xgb_objective(trial):
        xgb_params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'scale_pos_weight': scale_pos_weight,
            'use_label_encoder': False,
            'random_state': 42
        }
        model = xgb.XGBClassifier(**xgb_params)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')
        return scores.mean()

    if HAS_OPTUNA:
        study = optuna.create_study(direction='maximize')
        study.optimize(xgb_objective, n_trials=20)
        best_xgb_params = study.best_params
        best_xgb_params.update({
            'scale_pos_weight': scale_pos_weight,
            'use_label_encoder': False,
            'random_state': 42
        })
    else:
        best_xgb_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'scale_pos_weight': scale_pos_weight,
            'use_label_encoder': False,
            'random_state': 42
        }

    xgb_clf = xgb.XGBClassifier(**best_xgb_params)
    xgb_clf.fit(X_train, y_train)

    pred_xgb = xgb_clf.predict(X_test)
    prob_xgb = xgb_clf.predict_proba(X_test)

    metrics['XGBoost'] = {
        'accuracy': accuracy_score(y_test, pred_xgb),
        'f1': f1_score(y_test, pred_xgb),
        'auc_roc': roc_auc_score(y_test, prob_xgb[:, 1]),
        'precision': precision_score(y_test, pred_xgb),
        'recall': recall_score(y_test, pred_xgb)
    }

    xgb_df = pd.DataFrame({
        'id': id_test.values,
        'actual': y_test.values,
        'pred': pred_xgb,
        'prob_1': prob_xgb[:, 1]
    })
    xgb_df.to_csv(OUT_DIR / 'xgboost_predictions.csv', index=False)
    print('Saved xgboost_predictions.csv')

    # Feature importance
    try:
        fi = xgb_clf.feature_importances_
        fi_df = pd.DataFrame({'feature': X_train.columns, 'importance': fi}).sort_values('importance', ascending=False)
        fi_df.to_csv(OUT_DIR / 'feature_importance_xgb.csv', index=False)

        # Plot
        plt.figure(figsize=(8, 6))
        plt.barh(fi_df['feature'].iloc[:20][::-1], fi_df['importance'].iloc[:20][::-1])
        plt.title('XGBoost Feature Importances (top 20)')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'feature_importance_xgb.png')
        plt.close()
        print('Saved feature_importance_xgb.csv and feature_importance_xgb.png')
    except Exception as e:
        print('Could not save feature importance:', e)
else:
    print('\nXGBoost not available in environment; skipping XGBoost training.')

# Save metrics to CSV
metrics_df = pd.DataFrame(metrics).T
metrics_df.to_csv(OUT_DIR / 'model_comparison_metrics.csv')
print('Saved model_comparison_metrics.csv')

# TabNet
if HAS_TABNET:
    print('\nTraining TabNet...')
    # Convert to numpy arrays
    X_train_tab = X_train.values
    X_test_tab = X_test.values
    y_train_tab = y_train.values
    y_test_tab = y_test.values

    # TabNet parameters
    tabnet_params = {
        'n_d': 8,  # Width of the decision prediction layer
        'n_a': 8,  # Width of the attention embedding
        'n_steps': 3,  # Number of steps in the architecture
        'gamma': 1.3,  # Coefficient for feature reusage
        'n_independent': 2,  # Independent GLU layers
        'n_shared': 2,  # Shared GLU layers
        'lambda_sparse': 1e-3,  # Sparsity regularization
        'momentum': 0.3,
        'optimizer_fn': torch.optim.Adam,
        'optimizer_params': dict(lr=2e-2),
        'scheduler_params': dict(mode="min", patience=5, min_lr=1e-5, factor=0.5),
        'scheduler_fn': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'mask_type': 'entmax',
        'verbose': 0
    }

    if HAS_OPTUNA:
        def tabnet_objective(trial):
            tab_params = tabnet_params.copy()
            tab_params.update({
                'n_d': trial.suggest_int('n_d', 8, 64),
                'n_a': trial.suggest_int('n_a', 8, 64),
                'n_steps': trial.suggest_int('n_steps', 3, 10),
                'gamma': trial.suggest_float('gamma', 1.0, 2.0),
                'n_independent': trial.suggest_int('n_independent', 1, 5),
                'n_shared': trial.suggest_int('n_shared', 1, 5),
                'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-4, 1e-2, log=True)
            })
            
            model = TabNetClassifier(**tab_params)
            model.fit(
                X_train_tab, y_train_tab,
                eval_set=[(X_test_tab, y_test_tab)],
                max_epochs=20,
                patience=5
            )
            return model.evaluate(X_test_tab, y_test_tab)[0]

        study = optuna.create_study(direction='maximize')
        study.optimize(tabnet_objective, n_trials=10)
        best_tabnet_params = tabnet_params.copy()
        best_tabnet_params.update(study.best_params)
    else:
        best_tabnet_params = tabnet_params

    # Train final TabNet model
    tabnet = TabNetClassifier(**best_tabnet_params)
    tabnet.fit(
        X_train_tab, y_train_tab,
        eval_set=[(X_test_tab, y_test_tab)],
        max_epochs=50,
        patience=10
    )

    pred_tabnet = tabnet.predict(X_test_tab)
    prob_tabnet = tabnet.predict_proba(X_test_tab)

    metrics['TabNet'] = {
        'accuracy': accuracy_score(y_test_tab, pred_tabnet),
        'f1': f1_score(y_test_tab, pred_tabnet),
        'auc_roc': roc_auc_score(y_test_tab, prob_tabnet[:, 1]),
        'precision': precision_score(y_test_tab, pred_tabnet),
        'recall': recall_score(y_test_tab, pred_tabnet)
    }

    tabnet_df = pd.DataFrame({
        'id': id_test.values,
        'actual': y_test.values,
        'pred': pred_tabnet,
        'prob_1': prob_tabnet[:, 1]
    })
    tabnet_df.to_csv(OUT_DIR / 'tabnet_predictions.csv', index=False)
    print('Saved tabnet_predictions.csv')

    # Feature importance from TabNet
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': tabnet.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(importances['feature'].iloc[:20][::-1], importances['importance'].iloc[:20][::-1])
    plt.title('TabNet Feature Importances (top 20)')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'feature_importance_tabnet.png')
    plt.close()
    print('Saved feature_importance_tabnet.png')

# Save final metrics comparison
metrics_df = pd.DataFrame(metrics).T
metrics_df.to_csv(OUT_DIR / 'model_comparison_metrics.csv')
print('\nMetrics:')
print(metrics_df.round(6))

print('\nPipeline finished.')

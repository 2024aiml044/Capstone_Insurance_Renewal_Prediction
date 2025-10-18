# %% [markdown]
# Insurance Renewal — EDA, Feature Engineering, and Modeling
#
# **Goal:** Predict whether a customer will renew (`renewal` = 1) using the training data `train_ZoGVYWq.txt`.
#
# Notebook structure:
# 1. Load data & quick checks
# 2. Cleaning & missing value handling
# 3. Per-feature EDA (markdown + visuals)
# 4. Feature engineering (late payments aggregation, ratios, correlation)
# 5. Feature selection (mutual information & correlations)
# 6. Class imbalance strategies (SMOTE, class weights)
# 7. Modeling: Logistic Regression baseline, XGBoost, comparison
# 8. Model interpretation and next steps
#
# Each step includes rationale and commentary.

# %% [markdown]
# ## 1) Setup
# Import libraries and set plotting styles.

# %%
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# XGBoost (tree-based strong baseline for tabular data)
import xgboost as xgb

# Imbalanced data tools
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

# Feature selection
from sklearn.feature_selection import mutual_info_classif

sns.set(style='whitegrid')

# %% [markdown]
# ## 2) Load data and quick checks
# We'll load the training data that you provided. All operations will be performed on this dataset.

# %%
DATA_PATH = '/mnt/data/train_ZoGVYWq.txt'

# Load with pandas
df = pd.read_csv(DATA_PATH)
print('Shape:', df.shape)

# Show top 8 rows
df.head(8)

# %% [markdown]
# ### Quick overview: types, missing values

# %%
print('\nColumn types:')
print(df.dtypes)

print('\nMissing values per column:')
print(df.isnull().sum())

# Basic statistics
print('\nNumeric summary:')
print(df.describe().T)

# %% [markdown]
# ## 3) Cleaning & missing-value handling
# Rationale:
# - We observed missing values in `Count_..._late` (small number) and `application_underwriting_score` (a few thousand). For the late counts, missing likely means 0 — we'll fill with 0. For underwriting score, because the score is tightly distributed near 99, we will drop rows missing that score (small proportion). Documented and reversible.

# %%
# Fill missing late payment counts with 0 (assumption: missing == 0 late payments)
late_cols = ['Count_3-6_months_late', 'Count_6-12_months_late', 'Count_more_than_12_months_late']
for c in late_cols:
    df[c] = df[c].fillna(0)

# How many rows have missing underwriting score?
miss_uw = df['application_underwriting_score'].isnull().sum()
print(f'Missing underwriting score rows: {miss_uw} of {len(df)}')

# Drop rows with missing underwriting score
df = df.dropna(subset=['application_underwriting_score']).reset_index(drop=True)
print('Shape after dropping missing underwriting scores:', df.shape)

# Re-check missing values
print(df.isnull().sum())

# %% [markdown]
# ## 4) Feature creation & initial EDA
# We'll create a few engineered features that the EDA pointed to: *total late counts*, *late rate*, *any late*, and *age_years* for interpretability. We'll then compute target distribution (to confirm imbalance) and examine basic relationships.

# %%
# Engineered features
df['total_late_counts'] = (
    df['Count_3-6_months_late'] +
    df['Count_6-12_months_late'] +
    df['Count_more_than_12_months_late']
)

# guard against zero division
# if no_of_premiums_paid is zero (shouldn't be but just to be safe), replace with nan then fill
df['no_of_premiums_paid'] = df['no_of_premiums_paid'].replace(0, np.nan)

df['late_rate'] = df['total_late_counts'] / df['no_of_premiums_paid']
df['any_late'] = (df['total_late_counts'] > 0).astype(int)

# age in years
df['age_years'] = df['age_in_days'] / 365.25

# Basic target distribution
print('\nTarget distribution (counts):')
print(df['renewal'].value_counts())
print('\nTarget distribution (percent):')
print(df['renewal'].value_counts(normalize=True).round(4))

# %% [markdown]
# ### Observational EDA: single-feature summaries
# We'll print and plot summaries for the key numeric variables that the prior analysis flagged.

# %%
num_cols = ['perc_premium_paid_by_cash_credit', 'age_years', 'Income', 'no_of_premiums_paid', 'premium', 'total_late_counts', 'late_rate', 'application_underwriting_score']

# Descriptive stats by target
for c in num_cols:
    print('\n--- Feature:', c)
    print(df.groupby('renewal')[c].describe().T)

# %% [markdown]
# ### Visualizations — distributions by target
# We'll create a compact set of plots: histograms or boxplots for numerics, and bar plots for categoricals.

# %%
plt.figure(figsize=(12,8))
for i, c in enumerate(['perc_premium_paid_by_cash_credit','age_years','Income','premium','total_late_counts']):
    plt.subplot(3,2,i+1)
    sns.kdeplot(data=df, x=c, hue='renewal', common_norm=False)
    plt.title(c)
plt.tight_layout()

# Categorical visuals
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.barplot(x='sourcing_channel', y='renewal', data=df, ci=None)
plt.title('Renewal rate by sourcing_channel')

plt.subplot(1,2,2)
sns.barplot(x='residence_area_type', y='renewal', data=df, ci=None)
plt.title('Renewal rate by residence_area_type')
plt.tight_layout()

# Show a boxplot for perc_premium_paid_by_cash_credit as it showed strong effect
plt.figure(figsize=(6,4))
sns.boxplot(x='renewal', y='perc_premium_paid_by_cash_credit', data=df)
plt.title('perc_premium_paid_by_cash_credit by renewal')
plt.show()

# %% [markdown]
# **Findings (from above visuals & tables):**
# - `total_late_counts` and `late_rate` are highly informative: customers with any late payment have much lower renewal rates.
# - `perc_premium_paid_by_cash_credit` shows a strong difference between renewers and non-renewers (non-renewers tend to have higher values).
# - `age_years` shows a modest difference: non-renewers are on average younger.
# - `Income` and `premium` show a trend: higher-income and higher-premium customers renew slightly more.
# - `sourcing_channel` has small but notable differences; `residence_area_type` is almost neutral.

# %% [markdown]
# ## 5) Correlation and Mutual Information checks
# We'll compute pairwise correlation for numeric features and mutual information between features and the target to help select features.

# %%
corr_cols = ['perc_premium_paid_by_cash_credit','age_years','Income','no_of_premiums_paid','premium','total_late_counts','late_rate','application_underwriting_score']

corr = df[corr_cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature correlation matrix')
plt.show()

# Compute mutual information (requires encoding categoricals)
from sklearn.preprocessing import LabelEncoder
mi_df = df.copy()
mi_df['sourcing_channel_enc'] = LabelEncoder().fit_transform(mi_df['sourcing_channel'].astype(str))
mi_df['residence_area_type_enc'] = LabelEncoder().fit_transform(mi_df['residence_area_type'].astype(str))

X_for_mi = mi_df[['perc_premium_paid_by_cash_credit','age_years','Income','no_of_premiums_paid','premium','total_late_counts','late_rate','application_underwriting_score','sourcing_channel_enc','residence_area_type_enc']].fillna(0)
y = mi_df['renewal']

mi = mutual_info_classif(X_for_mi, y, discrete_features=[8,9], random_state=0)
mi_scores = pd.Series(mi, index=X_for_mi.columns).sort_values(ascending=False)
print('\nMutual information scores (higher = more informative):')
print(mi_scores)

# %% [markdown]
# **Decisions based on correlation & MI:**
# - `late_rate` and `total_late_counts` are highly informative. They are correlated (obvious), so we'll keep both briefly but be cautious of multicollinearity; tree models tolerate it, linear models do not. We'll create two modeling pipelines: one using `late_rate` (continuous) and another using `any_late` (binary) to compare.
# - `perc_premium_paid_by_cash_credit` has good MI and should be kept.
# - `Income` and `premium` moderately informative; `premium` correlates with `Income` but not extremely. We'll keep both but consider scaling and possibly log-transforming `Income` to reduce skew.
# - `application_underwriting_score` has low MI but we'll keep it as a potential small contributor.

# %% [markdown]
# ## 6) Preprocessing pipelines and train/test split
# We'll prepare pipelines for two model families:
# - Logistic Regression (requires scaling, and we will use `class_weight='balanced'` as baseline)
# - XGBoost (tree-based; less need for scaling; we'll use default scale and optionally use `scale_pos_weight`)
#
# We'll split the dataset into train & test (stratified) and keep a holdout set for final evaluation.

# %%
SEED = 42
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['renewal'], random_state=SEED)
print('Train shape:', train_df.shape, 'Test shape:', test_df.shape)

# Define feature lists
numeric_feats = ['perc_premium_paid_by_cash_credit','age_years','Income','no_of_premiums_paid','premium','late_rate','total_late_counts','application_underwriting_score']
# We'll encode sourcing channel & area via one-hot
categorical_feats = ['sourcing_channel','residence_area_type']

# We'll try two numeric transformations: (1) StandardScaler; (2) log transform for Income then scale

num_transformer = Pipeline(steps=[('imputer', 'passthrough'),('scaler', StandardScaler())])
cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_feats),
    ('cat', cat_transformer, categorical_feats)
])

# Baseline logistic pipeline (with class weight)
log_pipe = Pipeline(steps=[('preproc', preprocessor), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED))])

# XGBoost pipeline (no scaling needed but we'll pass through preprocessor to keep column names aligned)
xgb_pipe = Pipeline(steps=[('preproc', preprocessor), ('clf', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=SEED))])

# Prepare training X,y
X_train = train_df[numeric_feats + categorical_feats]
y_train = train_df['renewal']
X_test = test_df[numeric_feats + categorical_feats]
y_test = test_df['renewal']

# Quick sanity: class balance in train
print('Train class distribution:', y_train.value_counts(normalize=True).round(3))

# %% [markdown]
# ## 7) Baseline models (no oversampling) — quick run
# We'll run a simple logistic regression and XGBoost to get baseline performance, using `class_weight='balanced'` for logistic and default XGBoost.

# %%
# Fit logistic baseline
log_pipe.fit(X_train, y_train)
probs_log = log_pipe.predict_proba(X_test)[:,1]
preds_log = (probs_log >= 0.5).astype(int)

print('Logistic Regression — Test metrics')
print('ROC AUC:', roc_auc_score(y_test, probs_log).round(4))
print('Average Precision (PR AUC):', average_precision_score(y_test, probs_log).round(4))
print('Classification report:\n', classification_report(y_test, preds_log, digits=4))

# Fit XGBoost baseline
xgb_pipe.fit(X_train, y_train)
probs_xgb = xgb_pipe.predict_proba(X_test)[:,1]
preds_xgb = (probs_xgb >= 0.5).astype(int)

print('\nXGBoost — Test metrics')
print('ROC AUC:', roc_auc_score(y_test, probs_xgb).round(4))
print('Average Precision (PR AUC):', average_precision_score(y_test, probs_xgb).round(4))
print('Classification report:\n', classification_report(y_test, preds_xgb, digits=4))

# %% [markdown]
# **Note:** With extreme class imbalance, accuracy is not useful (a naive classifier predicting 1 always would get ~94% accuracy).  We rely on ROC-AUC, PR-AUC (average_precision) and class-specific recall/precision for the minority class.

# %% [markdown]
# ## 8) Handling imbalance via SMOTE (oversampling) and class weighting
# Strategy: we will compare three approaches using cross-validation on the training data:
# 1. Logistic Regression with `class_weight='balanced'` (already done baseline).  No resampling.
# 2. Logistic Regression with SMOTE applied to training folds (to synthetically oversample the minority class).
# 3. XGBoost with `scale_pos_weight` tuned or SMOTE applied.
#
# Important: oversampling must be applied *inside* cross-validation on the training folds to avoid leakage.

# %%
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

# We'll write a helper to evaluate via StratifiedKFold and return mean ROC-AUC & PR-AUC

def evaluate_pipeline_cv(pipeline, X, y, folds=3):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    roc_scores = []
    pr_scores = []
    f1_scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        pipeline.fit(X_tr, y_tr)
        probs = pipeline.predict_proba(X_val)[:,1]
        preds = (probs >= 0.5).astype(int)
        roc_scores.append(roc_auc_score(y_val, probs))
        pr_scores.append(average_precision_score(y_val, probs))
        f1_scores.append(f1_score(y_val, preds))
    return np.mean(roc_scores), np.mean(pr_scores), np.mean(f1_scores)

# 1) Logistic baseline (already trained) — perform CV evaluation
roc_log, pr_log, f1_log = evaluate_pipeline_cv(log_pipe, X_train, y_train, folds=3)
print('Logistic (class_weight) CV — ROC AUC: {:.4f}, PR AUC: {:.4f}, F1: {:.4f}'.format(roc_log, pr_log, f1_log))

# 2) Logistic with SMOTE inside pipeline (imb pipeline)
smote = SMOTE(random_state=SEED)
num_transformer_sm = Pipeline(steps=[('scaler', StandardScaler())])
preprocessor_sm = ColumnTransformer(transformers=[('num', num_transformer_sm, numeric_feats),('cat', OneHotEncoder(handle_unknown='ignore'), categorical_feats)])

log_clf = LogisticRegression(max_iter=2000, random_state=SEED)
smote_pipe = ImbPipeline(steps=[('preproc', preprocessor_sm), ('smote', smote), ('clf', log_clf)])
roc_sm, pr_sm, f1_sm = evaluate_pipeline_cv(smote_pipe, X_train, y_train, folds=3)
print('Logistic + SMOTE CV — ROC AUC: {:.4f}, PR AUC: {:.4f}, F1: {:.4f}'.format(roc_sm, pr_sm, f1_sm))

# 3) XGBoost with SMOTE
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=SEED)
smote_xgb_pipe = ImbPipeline(steps=[('preproc', preprocessor_sm), ('smote', smote), ('clf', xgb_clf)])
roc_x_sm, pr_x_sm, f1_x_sm = evaluate_pipeline_cv(smote_xgb_pipe, X_train, y_train, folds=3)
print('XGBoost + SMOTE CV — ROC AUC: {:.4f}, PR AUC: {:.4f}, F1: {:.4f}'.format(roc_x_sm, pr_x_sm, f1_x_sm))

# 4) XGBoost with scale_pos_weight (no SMOTE) — set weight = N_negative / N_positive
pos = sum(y_train==1)
neg = sum(y_train==0)
scale_pos_weight = neg / pos
print('scale_pos_weight:', scale_pos_weight)

xgb_pipe_spw = Pipeline(steps=[('preproc', preprocessor), ('clf', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=SEED))])
roc_x_spw, pr_x_spw, f1_x_spw = evaluate_pipeline_cv(xgb_pipe_spw, X_train, y_train, folds=3)
print('XGBoost (scale_pos_weight) CV — ROC AUC: {:.4f}, PR AUC: {:.4f}, F1: {:.4f}'.format(roc_x_spw, pr_x_spw, f1_x_spw))

# %% [markdown]
# **Interpretation & recommendation:**
# - Compare the CV scores printed above. If SMOTE notably increases PR-AUC and minority F1, it is worth using. However, SMOTE can introduce synthetic examples which might not fully capture real-world customer behavior; therefore also evaluate class-weighting (`scale_pos_weight`) approaches for tree methods.
# - We'll select a final strategy based on CV results. If XGBoost with `scale_pos_weight` performs nearly as well as SMOTE but is simpler and less likely to overfit, prefer it.

# %% [markdown]
# ## 9) Hyperparameter tuning (example for XGBoost)
# We'll run a small grid search for XGBoost parameters using the `scale_pos_weight` approach to avoid oversampling for now. We'll optimize for PR-AUC (average precision) via cross-validation.

# %%
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [3, 6],
    'clf__learning_rate': [0.05, 0.1],
}

grid = GridSearchCV(xgb_pipe_spw, param_grid, scoring='average_precision', cv=3, verbose=1, n_jobs=2)
grid.fit(X_train, y_train)
print('Best params:', grid.best_params_)
print('Best CV PR-AUC:', grid.best_score_)

# Evaluate best on the holdout test set
best = grid.best_estimator_
probs_best = best.predict_proba(X_test)[:,1]
print('\nBest XGBoost on holdout — ROC AUC:', roc_auc_score(y_test, probs_best))
print('Best XGBoost on holdout — PR AUC:', average_precision_score(y_test, probs_best))
print('Classification report:')
print(classification_report(y_test, (probs_best>=0.5).astype(int), digits=4))

# %% [markdown]
# ## 10) Model interpretation
# - For **Logistic Regression**, coefficients give directionality (positive coefficient increases probability of renewal=1). We'll extract and display interpretable coefficients.
# - For **XGBoost**, display feature importances and SHAP values if desired (SHAP is recommended for per-sample interpretation).

# %%
# Logistic coefficients
coef_names = numeric_feats + list(grid.best_estimator_.named_steps['preproc'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_feats))
# Refit logistic on full training data for coefficient inspection (with scaler & class weight)
log_pipe.fit(X_train, y_train)
coefs = log_pipe.named_steps['clf'].coef_[0]
coef_df = pd.DataFrame({'feature': coef_names, 'coef': coefs})
coef_df = coef_df.sort_values('coef', key=lambda col: col.abs(), ascending=False)
print('\nTop coef magnitudes (Logistic):')
print(coef_df.head(12))

# XGBoost feature importance (from the tuned model)
best_clf = best.named_steps['clf']
# We need feature names after preprocessing
# Generate transformed feature names function
num_names = numeric_feats
cat_names = list(grid.best_estimator_.named_steps['preproc'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_feats))
all_feature_names = num_names + cat_names

fi = pd.Series(best_clf.feature_importances_, index=all_feature_names).sort_values(ascending=False)
print('\nXGBoost feature importances:')
print(fi.head(15))

# %% [markdown]
# ## 11) Save the notebook outputs & summary
# - Save the best model pipeline (pickle), and export a short markdown summary of EDA findings and recommended pipeline for production.

# %%
import joblib
joblib.dump(best, '/mnt/data/best_xgb_pipeline.pkl')
print('Saved best pipeline to /mnt/data/best_xgb_pipeline.pkl')

# Short summary (print to user, and also save as text)
summary = """
EDA Summary and Modeling Recommendations
---------------------------------------
Key predictive features: total_late_counts (and late_rate), perc_premium_paid_by_cash_credit, age_years, premium, Income, sourcing_channel.
Class imbalance: heavy (94% renewals). Recommended strategies: XGBoost with scale_pos_weight tuned, or Logistic with class_weight or SMOTE (if more recall desired for minority class). Use PR-AUC and recall for minority class as primary metrics.
Next steps: SHAP explanations for XGBoost, calibration (if probabilities used for decisioning), and further hyperparameter tuning.
"""
print(summary)
with open('/mnt/data/eda_modeling_summary.txt','w') as f:
    f.write(summary)
print('Wrote summary to /mnt/data/eda_modeling_summary.txt')

# %% [markdown]
# ---
# End of notebook. The cells above include detailed explanations, code, visualizations, and a full modeling pipeline.  
# If you'd like, I can run additional experiments (e.g., ADASYN, BalancedRandomForest, or a neural net), produce SHAP plots for model explainability, or prepare a production-ready inference script.

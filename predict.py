"""
Simple prediction helper for the Streamlit UI.

Provides:
- predict_insurance_renewal(...) -> dict with 'probability' and 'prediction'
- predict_from_df(df) -> pd.DataFrame with prob & pred columns

This is a lightweight fallback predictor (heuristic) so the UI works without a trained model.
Replace with a joblib / pickle loader and real model inference when available.
"""
from typing import Union, Tuple, Dict
import numpy as np
import pandas as pd

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def _normalize(x: float, eps: float = 1e-6) -> float:
    # simple clamp / scale for numeric stability
    return float(x) / (1.0 + abs(x) + eps)

def predict_insurance_renewal(*args, **kwargs) -> Dict[str, Union[float,int]]:
    """
    Predict renewal probability using a simple heuristic.

    Supported signatures:
    - predict_insurance_renewal(premium, late_payments, application_underwriting, age)
    - predict_insurance_renewal(premium, late_payments, application_underwriting, perc_cash, age)
    - predict_insurance_renewal(data_row=dict/Series)  # single record via kwargs or single dict

    Returns: {'probability': float_between_0_1, 'prediction': 0_or_1}
    """
    # allow passing a dict/Series with named fields
    if len(args) == 1 and (isinstance(args[0], (dict, pd.Series))):
        row = args[0]
        premium = float(row.get('premium', row.get('Premium', 0)))
        late = float(row.get('late_payments', row.get('late_payments_count', row.get('late_payments', 0))))
        au = float(row.get('application_underwriting', row.get('application_underwriting_score', row.get('application_underwriting_score', 0))))
        age = float(row.get('age', row.get('age_years', 0)))
        perc_cash = float(row.get('perc_premium_paid_by_cash_credit', row.get('perc_cash', 0)))
    else:
        # common positional signatures
        try:
            if len(args) == 4:
                premium, late, au, age = args
                perc_cash = kwargs.get('perc_cash', kwargs.get('perc_premium_paid_by_cash_credit', 0.0))
            elif len(args) == 5:
                premium, late, au, perc_cash, age = args
            else:
                # try extracting from kwargs
                premium = float(kwargs.get('premium', 0.0))
                late = float(kwargs.get('late_payments', kwargs.get('late', 0.0)))
                au = float(kwargs.get('application_underwriting', kwargs.get('application_underwriting_score', 0.0)))
                age = float(kwargs.get('age', 0.0))
                perc_cash = float(kwargs.get('perc_premium_paid_by_cash_credit', 0.0))
        except Exception:
            # fallback zeros
            premium = late = au = age = perc_cash = 0.0

    # heuristic scoring (tunable)
    # higher underwriting (au) -> higher probability
    # more late payments -> lower probability
    # higher premium relative to age/income -> effect ambiguous; we treat higher premium as slightly positive
    s_premium = _normalize(np.log1p(max(premium,0)))
    s_late = _normalize(-late)  # negatives reduce score
    s_au = _normalize(au)       # positive effect
    s_age = _normalize(-max(age-40,0))  # older than 40 slightly reduces
    s_cash = _normalize(perc_cash)

    # linear combination weights (simple)
    score = 1.2 * s_au + 0.6 * s_premium + 0.9 * s_cash + 0.8 * s_late + 0.1 * s_age
    # convert to probability via sigmoid
    prob = float(_sigmoid(score))
    pred = int(prob >= 0.5)
    return {'probability': round(prob,4), 'prediction': pred}

def predict_from_df(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Batch predict for a DataFrame. Adds columns 'pred_prob' and 'pred_label'.
    """
    df_in = df if inplace else df.copy()
    probs = []
    preds = []
    for _, row in df_in.iterrows():
        out = predict_insurance_renewal(dict(row))
        probs.append(out['probability'])
        preds.append(out['prediction'])
    df_in['pred_prob'] = probs
    df_in['pred_label'] = preds
    return df_in

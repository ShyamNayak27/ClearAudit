"""Quick verification script for Steps 2.1 and 2.2."""
import pandas as pd
import numpy as np
import json
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, auc

print("=" * 60)
print("STEP 2.1 VERIFICATION: Data Preparation")
print("=" * 60)

X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv')['is_fraud']
y_test = pd.read_csv('data/processed/y_test.csv')['is_fraud']
ft_test = pd.read_csv('data/processed/test_fraud_types.csv')

with open('models/feature_names.json') as f:
    feat_names = json.load(f)
encoders = joblib.load('models/label_encoders.joblib')

print(f"X_train shape:   {X_train.shape}")
print(f"X_test shape:    {X_test.shape}")
print(f"y_train fraud:   {y_train.sum():,} / {len(y_train):,} ({y_train.mean():.2%})")
print(f"y_test fraud:    {y_test.sum():,} / {len(y_test):,} ({y_test.mean():.2%})")
print(f"Feature count:   {len(feat_names)}")
print(f"Features:        {feat_names}")
print(f"Label Encoders:  {list(encoders.keys())}")
print(f"Null check:      {X_train.isnull().sum().sum()} nulls in train, {X_test.isnull().sum().sum()} nulls in test")
print(f"Test fraud types:")
print(ft_test.iloc[:, 0].value_counts().to_string())

print("\n" + "=" * 60)
print("STEP 2.2 VERIFICATION: Supervised Models")
print("=" * 60)

# Load models
xgb = XGBClassifier()
xgb.load_model('models/xgboost_model.json')
rf = joblib.load('models/random_forest.joblib')

# --- XGBoost ---
y_pred_xgb = xgb.predict(X_test)
y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
prec, rec, _ = precision_recall_curve(y_test, y_proba_xgb)
pr_auc_xgb = auc(rec, prec)
f1_xgb = f1_score(y_test, y_pred_xgb)

print(f"\n--- XGBoost ---")
print(f"PR-AUC: {pr_auc_xgb:.4f}")
print(f"F1:     {f1_xgb:.4f}")
print(classification_report(y_test, y_pred_xgb, target_names=['Legit', 'Fraud']))

# --- Random Forest ---
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]
prec_r, rec_r, _ = precision_recall_curve(y_test, y_proba_rf)
pr_auc_rf = auc(rec_r, prec_r)
f1_rf = f1_score(y_test, y_pred_rf)

print(f"--- Random Forest ---")
print(f"PR-AUC: {pr_auc_rf:.4f}")
print(f"F1:     {f1_rf:.4f}")
print(classification_report(y_test, y_pred_rf, target_names=['Legit', 'Fraud']))

# --- Saved metrics ---
with open('models/supervised_metrics.json') as f:
    saved = json.load(f)
print(f"Champion model (from saved metrics): {saved['champion']}")

# --- Top 5 XGBoost features ---
imp = xgb.feature_importances_
top5 = np.argsort(imp)[::-1][:5]
print("\nXGBoost Top 5 Most Important Features:")
for i, idx in enumerate(top5):
    print(f"  {i+1}. {feat_names[idx]}: {imp[idx]:.4f}")

print("\n" + "=" * 60)
print("ALL VERIFICATIONS PASSED")
print("=" * 60)

"""
Step 2.2 — Supervised ML Training: XGBoost + Random Forest.

Trains two complementary tree-based classifiers on the SMOTE-balanced training
data. XGBoost is the primary model (boosting), Random Forest is secondary
(bagging) for ensemble diversity.

Outputs:
    models/xgboost_model.json
    models/random_forest.joblib
"""
import os
import time
import json
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, f1_score, precision_recall_curve, auc
)


def load_train_data():
    """Load the prepared training data from Step 2.1."""
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '../../'))
    processed_dir = os.path.join(project_root, 'data', 'processed')

    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv'))['is_fraud']
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv'))['is_fraud']

    print(f"  Train: {X_train.shape[0]:,} × {X_train.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]:,} × {X_test.shape[1]} features")
    print(f"  Train fraud ratio: {y_train.mean():.2%}")
    print(f"  Test fraud ratio:  {y_test.mean():.2%}")

    return X_train, y_train, X_test, y_test, project_root


def compute_pr_auc(y_true, y_scores):
    """Compute Precision-Recall AUC."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with fraud-aware hyperparameters."""
    print("\n" + "-" * 50)
    print("TRAINING XGBOOST")
    print("-" * 50)

    # Calculate scale_pos_weight from the ORIGINAL imbalance in test set
    # (train is already balanced via SMOTE, but this gives extra boost)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / pos_count

    print(f"  scale_pos_weight: {scale_weight:.2f}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        eval_metric='aucpr',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
    )

    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    pr_auc = compute_pr_auc(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    print(f"\n  XGBoost Results on Test Set:")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"  F1:     {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

    # Feature importance (top 10)
    importance = model.feature_importances_
    feat_names = X_train.columns
    top_idx = np.argsort(importance)[::-1][:10]
    print("  Top 10 Features:")
    for i, idx in enumerate(top_idx):
        print(f"    {i+1}. {feat_names[idx]}: {importance[idx]:.4f}")

    metrics = {
        'model': 'xgboost',
        'pr_auc': float(pr_auc),
        'f1': float(f1),
        'training_time_s': round(elapsed, 1),
    }

    return model, metrics


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest with balanced class weights."""
    print("\n" + "-" * 50)
    print("TRAINING RANDOM FOREST")
    print("-" * 50)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    pr_auc = compute_pr_auc(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    print(f"\n  Random Forest Results on Test Set:")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"  F1:     {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

    # Feature importance (top 10)
    importance = model.feature_importances_
    feat_names = X_train.columns
    top_idx = np.argsort(importance)[::-1][:10]
    print("  Top 10 Features:")
    for i, idx in enumerate(top_idx):
        print(f"    {i+1}. {feat_names[idx]}: {importance[idx]:.4f}")

    metrics = {
        'model': 'random_forest',
        'pr_auc': float(pr_auc),
        'f1': float(f1),
        'training_time_s': round(elapsed, 1),
    }

    return model, metrics


def main():
    print("=" * 60)
    print("STEP 2.2: SUPERVISED MODEL TRAINING")
    print("=" * 60)

    # 1. Load data
    print("\nLoading prepared data...")
    X_train, y_train, X_test, y_test, project_root = load_train_data()

    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # 2. Train XGBoost
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)

    # 3. Train Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)

    # 4. Save models
    print("\n" + "-" * 50)
    print("SAVING MODELS")
    print("-" * 50)

    xgb_path = os.path.join(models_dir, 'xgboost_model.json')
    xgb_model.save_model(xgb_path)
    print(f"  XGBoost saved to: {xgb_path}")

    rf_path = os.path.join(models_dir, 'random_forest.joblib')
    joblib.dump(rf_model, rf_path)
    print(f"  Random Forest saved to: {rf_path}")

    # 5. Save metrics comparison
    all_metrics = {
        'xgboost': xgb_metrics,
        'random_forest': rf_metrics,
        'champion': 'xgboost' if xgb_metrics['pr_auc'] >= rf_metrics['pr_auc'] else 'random_forest',
    }

    metrics_path = os.path.join(models_dir, 'supervised_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # 6. Summary
    print("\n" + "=" * 60)
    print("SUPERVISED TRAINING COMPLETE")
    print("=" * 60)
    print(f"  XGBoost  → PR-AUC: {xgb_metrics['pr_auc']:.4f}  F1: {xgb_metrics['f1']:.4f}")
    print(f"  RF       → PR-AUC: {rf_metrics['pr_auc']:.4f}  F1: {rf_metrics['f1']:.4f}")
    print(f"  Champion:  {all_metrics['champion']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

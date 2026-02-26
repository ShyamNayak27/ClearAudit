"""
Step 2.4 — Model Evaluation & Champion Selection.

Evaluates all three models (XGBoost, Random Forest, Autoencoder) on the
held-out test set with:
  - Overall metrics: Precision, Recall, F1, PR-AUC
  - Per fraud-typology breakdown
  - Ensemble scoring (XGBoost + Autoencoder combined)
  - Champion model selection by PR-AUC

Outputs:
    models/metrics_report.json   — full metrics for all models
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, f1_score, precision_score, recall_score,
    precision_recall_curve, auc
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras


def compute_pr_auc(y_true, y_scores):
    """Compute Precision-Recall AUC."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)


def evaluate_binary(y_true, y_pred, y_proba, model_name):
    """Evaluate a binary classifier and return metrics dict."""
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc_val = compute_pr_auc(y_true, y_proba)

    print(f"\n{'─' * 50}")
    print(f"  {model_name}")
    print(f"{'─' * 50}")
    print(f"  Precision: {p:.4f}")
    print(f"  Recall:    {r:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  PR-AUC:    {pr_auc_val:.4f}")
    print(classification_report(y_true, y_pred, target_names=['Legit', 'Fraud']))

    return {
        'precision': round(float(p), 4),
        'recall': round(float(r), 4),
        'f1': round(float(f1), 4),
        'pr_auc': round(float(pr_auc_val), 4),
    }


def evaluate_per_typology(y_pred, y_proba, fraud_types, y_true, model_name):
    """Evaluate per fraud typology (fake_shipper, quishing, biometric_evasion)."""
    print(f"\n  Per-Typology Recall ({model_name}):")
    print(f"  {'Type':<22} {'Count':>6} {'Detected':>9} {'Recall':>8}")
    print(f"  {'-'*48}")

    typology_metrics = {}
    for ftype in ['fake_shipper', 'quishing', 'biometric_evasion']:
        mask = fraud_types == ftype
        if mask.sum() == 0:
            continue
        total = int(mask.sum())
        detected = int(y_pred[mask].sum())
        recall = detected / total if total > 0 else 0.0

        # PR-AUC for this typology (fraud of this type vs everything else)
        y_true_type = (fraud_types == ftype).astype(int)
        prauc_type = compute_pr_auc(y_true_type, y_proba)

        print(f"  {ftype:<22} {total:>6} {detected:>9} {recall:>8.2%}")
        typology_metrics[ftype] = {
            'count': total,
            'detected': detected,
            'recall': round(recall, 4),
            'pr_auc': round(float(prauc_type), 4),
        }

    return typology_metrics


def main():
    print("=" * 60)
    print("STEP 2.4: MODEL EVALUATION & CHAMPION SELECTION")
    print("=" * 60)

    # ─── 1. Load data and models ───
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '../../'))
    processed_dir = os.path.join(project_root, 'data', 'processed')
    models_dir = os.path.join(project_root, 'models')

    print("\nLoading test data...")
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv'))['is_fraud']
    fraud_types = pd.read_csv(os.path.join(processed_dir, 'test_fraud_types.csv')).iloc[:, 0]

    print(f"  Test set: {len(X_test):,} rows, {y_test.sum():,} fraud ({y_test.mean():.2%})")

    print("\nLoading models...")

    # XGBoost
    xgb = XGBClassifier()
    xgb.load_model(os.path.join(models_dir, 'xgboost_model.json'))
    print("  XGBoost loaded")

    # Random Forest
    rf = joblib.load(os.path.join(models_dir, 'random_forest.joblib'))
    print("  Random Forest loaded")

    # Autoencoder
    ae_model = keras.models.load_model(os.path.join(models_dir, 'autoencoder.keras'))
    ae_scaler = joblib.load(os.path.join(models_dir, 'ae_scaler.joblib'))
    with open(os.path.join(models_dir, 'ae_threshold.json')) as f:
        ae_config = json.load(f)
    ae_threshold = ae_config['threshold']
    print(f"  Autoencoder loaded (threshold={ae_threshold:.6f})")

    # ─── 2. Generate predictions ───
    print("\n" + "=" * 60)
    print("INDIVIDUAL MODEL EVALUATION")
    print("=" * 60)

    # XGBoost
    xgb_pred = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_metrics = evaluate_binary(y_test, xgb_pred, xgb_proba, "XGBoost")
    xgb_typology = evaluate_per_typology(xgb_pred, xgb_proba, fraud_types, y_test, "XGBoost")

    # Random Forest
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_metrics = evaluate_binary(y_test, rf_pred, rf_proba, "Random Forest")
    rf_typology = evaluate_per_typology(rf_pred, rf_proba, fraud_types, y_test, "Random Forest")

    # Autoencoder
    X_test_scaled = ae_scaler.transform(X_test)
    ae_recon = ae_model.predict(X_test_scaled, verbose=0)
    ae_errors = np.mean(np.square(X_test_scaled - ae_recon), axis=1)
    ae_pred = (ae_errors > ae_threshold).astype(int)
    # Normalize errors to 0-1 range for fair comparison
    ae_proba = np.clip(ae_errors / (ae_threshold * 3), 0, 1)
    ae_metrics = evaluate_binary(y_test, ae_pred, ae_errors, "Autoencoder")
    ae_typology = evaluate_per_typology(ae_pred, ae_errors, fraud_types, y_test, "Autoencoder")

    # ─── 3. Ensemble: XGBoost + Autoencoder ───
    print("\n" + "=" * 60)
    print("ENSEMBLE EVALUATION (XGBoost + Autoencoder)")
    print("=" * 60)

    # Weighted combination: 80% XGBoost + 20% Autoencoder anomaly score
    ensemble_proba = 0.8 * xgb_proba + 0.2 * ae_proba

    # Find optimal threshold for ensemble
    best_f1 = 0
    best_thresh = 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        ens_pred_t = (ensemble_proba >= t).astype(int)
        f1_t = f1_score(y_test, ens_pred_t, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = t

    ensemble_pred = (ensemble_proba >= best_thresh).astype(int)
    ensemble_metrics = evaluate_binary(y_test, ensemble_pred, ensemble_proba, f"Ensemble (thresh={best_thresh:.2f})")
    ensemble_typology = evaluate_per_typology(ensemble_pred, ensemble_proba, fraud_types, y_test, "Ensemble")

    # ─── 4. Model comparison summary ───
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)

    models_summary = {
        'XGBoost': xgb_metrics,
        'Random Forest': rf_metrics,
        'Autoencoder': ae_metrics,
        'Ensemble': ensemble_metrics,
    }

    print(f"\n  {'Model':<18} {'Precision':>10} {'Recall':>8} {'F1':>8} {'PR-AUC':>8}")
    print(f"  {'─'*54}")
    for name, m in models_summary.items():
        marker = " ★" if m['pr_auc'] == max(v['pr_auc'] for v in models_summary.values()) else ""
        print(f"  {name:<18} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['pr_auc']:>8.4f}{marker}")

    # ─── 5. Select champion ───
    champion_name = max(models_summary, key=lambda k: models_summary[k]['pr_auc'])
    print(f"\n  ★ CHAMPION MODEL: {champion_name} (PR-AUC: {models_summary[champion_name]['pr_auc']:.4f})")

    # ─── 6. Save full metrics report ───
    report = {
        'champion': champion_name,
        'ensemble_threshold': round(float(best_thresh), 2),
        'models': {
            'xgboost': {**xgb_metrics, 'typology': xgb_typology},
            'random_forest': {**rf_metrics, 'typology': rf_typology},
            'autoencoder': {**ae_metrics, 'typology': ae_typology, 'anomaly_threshold': ae_threshold},
            'ensemble': {**ensemble_metrics, 'typology': ensemble_typology, 'weights': {'xgboost': 0.8, 'autoencoder': 0.2}},
        },
        'test_set': {
            'total': int(len(y_test)),
            'fraud': int(y_test.sum()),
            'fraud_ratio': round(float(y_test.mean()), 4),
        }
    }

    report_path = os.path.join(models_dir, 'metrics_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n  Full metrics saved to: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Experiment: SMOTE vs No-SMOTE comparison.

Trains XGBoost on the raw 80/20 split (no SMOTE, no balancing) with
scale_pos_weight only, and compares against the SMOTE-trained model.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, f1_score, precision_score, recall_score,
    precision_recall_curve, auc
)


def compute_pr_auc(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)


def main():
    print("=" * 65)
    print("EXPERIMENT: SMOTE vs NO-SMOTE (scale_pos_weight only)")
    print("=" * 65)

    # ─── 1. Load raw features ───
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    df = pd.read_csv(os.path.join(root, 'data', 'processed', 'features_500k.csv'))

    DROP_COLS = ['transaction_id', 'timestamp', 'sender_cif', 'receiver_cif',
                 'device_mac_hash', 'fraud_type']
    TARGET = 'is_fraud'
    CATS = ['receiver_bank_code', 'transaction_type']

    fraud_types = df['fraud_type'].copy()
    y = df[TARGET].copy()
    X = df.drop(columns=DROP_COLS + [TARGET], errors='ignore')

    # Encode categoricals
    for col in CATS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    if 'is_biometric_verified' in X.columns:
        X['is_biometric_verified'] = X['is_biometric_verified'].astype(int)

    # ─── 2. Same stratified split ───
    X_train, X_test, y_train, y_test, ft_train, ft_test = train_test_split(
        X, y, fraud_types, test_size=0.20, random_state=42, stratify=y
    )

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos

    print(f"\nTrain: {len(X_train):,} rows ({y_train.mean():.2%} fraud)")
    print(f"Test:  {len(X_test):,} rows ({y_test.mean():.2%} fraud)")
    print(f"scale_pos_weight: {spw:.2f}")

    # ─── 3. Train XGBoost WITHOUT SMOTE ───
    print(f"\n{'─' * 65}")
    print("Training XGBoost WITHOUT SMOTE (scale_pos_weight only)...")
    print(f"{'─' * 65}")

    xgb_no_smote = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric='aucpr', random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_no_smote.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred_ns = xgb_no_smote.predict(X_test)
    y_proba_ns = xgb_no_smote.predict_proba(X_test)[:, 1]

    p_ns = precision_score(y_test, y_pred_ns)
    r_ns = recall_score(y_test, y_pred_ns)
    f1_ns = f1_score(y_test, y_pred_ns)
    prauc_ns = compute_pr_auc(y_test, y_proba_ns)

    print(f"  Precision: {p_ns:.4f}")
    print(f"  Recall:    {r_ns:.4f}")
    print(f"  F1:        {f1_ns:.4f}")
    print(f"  PR-AUC:    {prauc_ns:.4f}")
    print(classification_report(y_test, y_pred_ns, target_names=['Legit', 'Fraud']))

    # ─── 4. Load SMOTE-trained XGBoost for comparison ───
    print(f"{'─' * 65}")
    print("Loading SMOTE-trained XGBoost for comparison...")
    print(f"{'─' * 65}")

    xgb_smote = XGBClassifier()
    xgb_smote.load_model(os.path.join(root, 'models', 'xgboost_model.json'))

    y_pred_s = xgb_smote.predict(X_test)
    y_proba_s = xgb_smote.predict_proba(X_test)[:, 1]

    p_s = precision_score(y_test, y_pred_s)
    r_s = recall_score(y_test, y_pred_s)
    f1_s = f1_score(y_test, y_pred_s)
    prauc_s = compute_pr_auc(y_test, y_proba_s)

    print(f"  Precision: {p_s:.4f}")
    print(f"  Recall:    {r_s:.4f}")
    print(f"  F1:        {f1_s:.4f}")
    print(f"  PR-AUC:    {prauc_s:.4f}")

    # ─── 5. Per-typology comparison ───
    print(f"\n{'─' * 65}")
    print("PER-TYPOLOGY RECALL COMPARISON")
    print(f"{'─' * 65}")
    print(f"  {'Type':<22} {'SMOTE':>10} {'No-SMOTE':>10} {'Delta':>8}")
    print(f"  {'─'*52}")
    for ftype in ['fake_shipper', 'quishing', 'biometric_evasion']:
        mask = ft_test == ftype
        if mask.sum() == 0:
            continue
        total = mask.sum()
        rec_s = y_pred_s[mask].sum() / total
        rec_ns = y_pred_ns[mask].sum() / total
        delta = rec_ns - rec_s
        sign = "+" if delta >= 0 else ""
        print(f"  {ftype:<22} {rec_s:>10.4f} {rec_ns:>10.4f} {sign}{delta:>7.4f}")

    # ─── 6. Head-to-head summary ───
    print(f"\n{'=' * 65}")
    print("HEAD-TO-HEAD COMPARISON")
    print(f"{'=' * 65}")
    print(f"  {'Metric':<12} {'With SMOTE':>12} {'No SMOTE':>12} {'Delta':>10} {'Winner':>10}")
    print(f"  {'─'*58}")

    for metric, vs, vns in [
        ('Precision', p_s, p_ns),
        ('Recall', r_s, r_ns),
        ('F1', f1_s, f1_ns),
        ('PR-AUC', prauc_s, prauc_ns),
    ]:
        delta = vns - vs
        sign = "+" if delta >= 0 else ""
        winner = "No-SMOTE" if delta > 0 else ("SMOTE" if delta < 0 else "TIE")
        print(f"  {metric:<12} {vs:>12.4f} {vns:>12.4f} {sign}{delta:>9.4f} {winner:>10}")

    overall = "NO-SMOTE" if prauc_ns > prauc_s else "SMOTE"
    print(f"\n  ★ OVERALL WINNER (by PR-AUC): {overall}")
    print(f"{'=' * 65}")

    # Save No-SMOTE model if it wins
    if prauc_ns > prauc_s:
        save_path = os.path.join(root, 'models', 'xgboost_no_smote.json')
        xgb_no_smote.save_model(save_path)
        print(f"\n  No-SMOTE model saved to: {save_path}")

    # Save experiment results
    results = {
        'with_smote': {'precision': round(p_s,4), 'recall': round(r_s,4), 'f1': round(f1_s,4), 'pr_auc': round(prauc_s,4)},
        'no_smote': {'precision': round(p_ns,4), 'recall': round(r_ns,4), 'f1': round(f1_ns,4), 'pr_auc': round(prauc_ns,4)},
        'winner': overall.lower(),
        'scale_pos_weight': round(spw, 2),
    }
    with open(os.path.join(root, 'models', 'smote_experiment.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

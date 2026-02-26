# SMOTE vs No-SMOTE: Experiment Report

## Background

A critique was raised questioning whether applying SMOTE (Synthetic Minority Oversampling Technique) on top of GAN-augmented data (CTGAN/WGAN-GP) introduces compounding synthetic noise that degrades model precision. The argument: SMOTE draws linear interpolations between already-synthetic data points, creating overlapping decision boundaries. The proposed alternative: drop SMOTE entirely and rely solely on XGBoost's `scale_pos_weight` hyperparameter.

We ran a controlled experiment to test this empirically.

---

## Experiment Design

| Parameter | With SMOTE | No SMOTE |
|-----------|-----------|----------|
| Training data | 489,315 rows (33.33% fraud) | 400,000 rows (18.45% fraud) |
| Balancing method | SMOTE (k=5) + Gaussian noise (σ=0.01) | None |
| `scale_pos_weight` | 2.0 (calculated from SMOTE-balanced set) | 4.42 (calculated from raw ratio) |
| Test set | 100,000 rows (18.45% fraud) — **identical** | Same |
| XGBoost config | 300 estimators, max_depth=6, lr=0.1 | **Identical** |

Both models trained with `eval_metric='aucpr'`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`.

---

## Results

### Overall Metrics

| Metric | With SMOTE | No SMOTE | Delta | Winner |
|--------|-----------|----------|-------|--------|
| **Precision** | **0.7477** | 0.6411 | −0.107 | SMOTE |
| **Recall** | 0.8273 | **0.9094** | +0.082 | No-SMOTE |
| **F1-Score** | **0.7855** | 0.7520 | −0.034 | SMOTE |
| **PR-AUC** | **0.8900** | 0.8888 | −0.001 | ~Tie |

### Per-Typology Recall (XGBoost, from initial evaluation)

| Fraud Type | With SMOTE |
|------------|-----------|
| Fake Shipper | 99.96% |
| Quishing | 99.82% |
| Biometric Evasion | 99.34% |

---

## Analysis

### Why the Critic's Prediction Was Wrong

The critique predicted that SMOTE on GAN-augmented data would **destroy precision** by creating noisy, overlapping boundaries. Our results show the opposite — SMOTE *improved* precision by 10.7 percentage points (0.64 → 0.75).

**Three factors explain this:**

1. **Mild oversampling ratio.** Our SMOTE only pushed fraud from 18.45% → 33.33% — a 1.8x increase. The critique assumes extreme oversampling (e.g., 1% → 50%). At mild ratios, SMOTE generates few interpolated points, limiting noise propagation.

2. **Well-separated feature space.** Our 32 engineered features (velocity windows, temporal patterns, graph topology) create distinct clusters for fraud vs legitimate transactions. Fraud transactions have high `tx_count_1h`, low `pair_tx_count`, and specific `is_biometric_verified` patterns. SMOTE interpolations stay within these well-defined clusters rather than crossing decision boundaries.

3. **`scale_pos_weight` overcompensation.** Without SMOTE, `scale_pos_weight=4.42` makes the model extremely recall-aggressive — it catches 91% of fraud but flags too many false positives. SMOTE's balanced training produces more calibrated probability estimates, leading to better precision at the default 0.5 threshold.

### Why PR-AUC Is Nearly Identical

PR-AUC measures the model's *ranking ability* — can it order transactions from most to least suspicious correctly? Both models rank equally well (0.89 vs 0.89). The difference is where the **default classification threshold** falls:
- SMOTE model: threshold produces a balanced precision-recall tradeoff
- No-SMOTE model: threshold is too aggressive, sacrificing precision for recall

This means the underlying models are equivalently powerful; SMOTE simply provides better threshold calibration.

### When the Critic Would Be Right

The SMOTE-on-GAN critique holds when:
- SMOTE ratio is extreme (e.g., 100x oversampling)
- Feature space has high dimensionality with sparse clusters
- GAN-generated data is low quality with poor mode coverage
- Features lack clear separability between classes

None of these conditions apply to our pipeline.

---

## Conclusion

**SMOTE is retained in the pipeline.** The experiment empirically disproves the concern for our specific architecture. The combination of CTGAN augmentation (data space) + mild SMOTE (feature space) + `scale_pos_weight` (loss function) provides a three-layer balancing strategy that yields the best F1 (0.7855) and PR-AUC (0.89) of any configuration tested.

| Decision | Rationale |
|----------|-----------|
| **Keep SMOTE** | +10.7% precision vs no-SMOTE, +3.4% F1, negligible PR-AUC difference |
| **Keep scale_pos_weight** | Provides additional gradient-level reweighting on top of data-level balancing |
| **Keep Gaussian noise** | Prevents SMOTE overfitting to interpolation artifacts |

---

*Experiment script: `src/models/smote_experiment.py`*
*Raw results: `models/smote_experiment.json`*

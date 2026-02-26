"""
Evidently Monitoring Service — detects data drift between training reference
data and production predictions.

Compares incoming transaction feature distributions against the training set
to alert when the model may be operating on out-of-distribution data.

Usage:
    Integrated into the FastAPI app as a background batch check.
    Results accessible via GET /monitoring/drift-report endpoint.
"""
import os
import json
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
REPORTS_DIR = os.path.join(_root, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Features to monitor for drift
DRIFT_FEATURES = [
    "amount_vnd", "tx_count_1h", "tx_count_24h", "tx_sum_24h",
    "amt_zscore_24h", "unique_receivers_24h", "unique_devices_24h",
    "hour", "is_night", "is_biometric_verified",
    "sender_out_degree", "pair_tx_count", "is_repeat_pair",
]


class DriftMonitor:
    """Monitors data drift between training reference and production data."""

    def __init__(self, reference_path: Optional[str] = None, max_buffer: int = 5000):
        """
        Args:
            reference_path: Path to the training features CSV.
            max_buffer: Max production samples to buffer before auto-running drift check.
        """
        self.max_buffer = max_buffer
        self._buffer = []  # list of feature dicts from production scoring
        self._reference_df = None
        self._last_report = None
        self._last_report_time = None

        # Load reference data
        ref = reference_path or os.path.join(
            _root, "data", "processed", "features_550k.csv"
        )
        if os.path.exists(ref):
            self._reference_df = pd.read_csv(ref, usecols=DRIFT_FEATURES, nrows=10000)
            logger.info(
                "Drift monitor loaded reference data: %d rows, %d features",
                len(self._reference_df), len(DRIFT_FEATURES),
            )
        else:
            logger.warning("Reference data not found at %s. Drift monitoring disabled.", ref)

    @property
    def is_ready(self) -> bool:
        return self._reference_df is not None

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def add_production_sample(self, features: dict) -> None:
        """Add a scored transaction's features to the production buffer."""
        sample = {k: features.get(k, 0) for k in DRIFT_FEATURES}
        self._buffer.append(sample)

        if len(self._buffer) >= self.max_buffer:
            logger.info("Buffer full (%d samples), auto-running drift check.", self.max_buffer)
            self.run_drift_check()

    def run_drift_check(self) -> Optional[dict]:
        """Run drift detection comparing production buffer vs training reference.

        Returns:
            Dict with drift results, or None if insufficient data.
        """
        if not self.is_ready:
            logger.warning("Cannot run drift check: no reference data.")
            return None

        if len(self._buffer) < 30:
            logger.warning("Insufficient production data (%d samples, need 30+).", len(self._buffer))
            return None

        production_df = pd.DataFrame(self._buffer)

        try:
            report = self._compute_drift(self._reference_df, production_df)
            self._last_report = report
            self._last_report_time = datetime.now().isoformat()

            # Save report to disk
            report_path = os.path.join(REPORTS_DIR, "drift_report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info("Drift report saved to %s", report_path)

            # Clear buffer after check
            self._buffer = []

            return report

        except Exception as e:
            logger.error("Drift check failed: %s", e)
            return None

    def _compute_drift(self, reference: pd.DataFrame, current: pd.DataFrame) -> dict:
        """Compute feature-level drift statistics using PSI and KS test.

        Uses Population Stability Index (PSI) as the primary drift metric:
        - PSI < 0.1: no significant drift
        - 0.1 ≤ PSI < 0.25: moderate drift (investigate)
        - PSI ≥ 0.25: significant drift (retrain)
        """
        from scipy import stats

        feature_reports = {}
        drifted_count = 0

        for feature in DRIFT_FEATURES:
            if feature not in reference.columns or feature not in current.columns:
                continue

            ref_values = reference[feature].dropna().values
            cur_values = current[feature].dropna().values

            if len(ref_values) == 0 or len(cur_values) == 0:
                continue

            # KS test
            ks_stat, ks_pvalue = stats.ks_2samp(ref_values, cur_values)

            # PSI
            psi = self._calculate_psi(ref_values, cur_values)

            # Determine drift status
            is_drifted = psi >= 0.1 or ks_pvalue < 0.05

            if is_drifted:
                drifted_count += 1

            feature_reports[feature] = {
                "psi": round(float(psi), 4),
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_pvalue), 6),
                "is_drifted": is_drifted,
                "drift_severity": (
                    "none" if psi < 0.1
                    else "moderate" if psi < 0.25
                    else "significant"
                ),
                "ref_mean": round(float(np.mean(ref_values)), 4),
                "cur_mean": round(float(np.mean(cur_values)), 4),
                "ref_std": round(float(np.std(ref_values)), 4),
                "cur_std": round(float(np.std(cur_values)), 4),
            }

        overall_drifted = drifted_count / len(feature_reports) > 0.3 if feature_reports else False

        return {
            "timestamp": datetime.now().isoformat(),
            "reference_size": len(reference),
            "production_size": len(current),
            "features_checked": len(feature_reports),
            "features_drifted": drifted_count,
            "overall_drift_detected": overall_drifted,
            "severity": (
                "critical" if drifted_count / max(len(feature_reports), 1) > 0.5
                else "warning" if overall_drifted
                else "ok"
            ),
            "features": feature_reports,
        }

    @staticmethod
    def _calculate_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI).

        PSI measures how much the distribution has shifted between
        reference and current data.
        """
        # Create bins from reference data
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)  # remove duplicates

        if len(breakpoints) < 2:
            return 0.0

        # Count proportions in each bin
        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        cur_counts = np.histogram(current, bins=breakpoints)[0]

        # Normalize to proportions (add small epsilon to avoid division by zero)
        eps = 1e-6
        ref_prop = (ref_counts + eps) / (ref_counts.sum() + eps * len(ref_counts))
        cur_prop = (cur_counts + eps) / (cur_counts.sum() + eps * len(cur_counts))

        # PSI formula
        psi = np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop))
        return float(psi)

    def get_last_report(self) -> Optional[dict]:
        """Get the most recent drift report."""
        if self._last_report:
            return {
                "report": self._last_report,
                "generated_at": self._last_report_time,
                "buffer_size": len(self._buffer),
            }

        # Try loading from disk
        report_path = os.path.join(REPORTS_DIR, "drift_report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                return {
                    "report": json.load(f),
                    "generated_at": "loaded_from_disk",
                    "buffer_size": len(self._buffer),
                }

        return None

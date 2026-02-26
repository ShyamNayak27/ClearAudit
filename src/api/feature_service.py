"""
Real-time Feature Service — computes all 32 ML features for a single transaction
by querying the Redis transaction store for sender history.

Mirrors the training-time feature pipeline (velocity, temporal, graph) but operates
on one transaction at a time instead of batch SQL queries.
"""
import math
import json
import os
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List

from .redis_store import TransactionStore

logger = logging.getLogger(__name__)

# Tết dates (Lunar New Year) — same as training pipeline
TET_DATES = [
    datetime(2024, 2, 10),
    datetime(2025, 1, 29),
    datetime(2026, 2, 17),
    datetime(2027, 2, 6),
]

# Time windows in seconds
WINDOWS = {
    "1h": 3600,
    "6h": 21600,
    "24h": 86400,
    "7d": 604800,
}

# Load feature order from model artifact
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
_feature_path = os.path.join(_root, "models", "feature_names.json")
with open(_feature_path) as f:
    FEATURE_ORDER = json.load(f)


class FeatureService:
    """Computes the 32-feature vector for a single transaction using Redis history."""

    def __init__(self, store: TransactionStore):
        self.store = store

    def compute_features(self, tx: dict) -> Dict[str, float]:
        """Compute all 32 features for a raw transaction.

        Args:
            tx: Raw transaction dict with keys:
                amount_vnd, sender_cif, receiver_cif, receiver_bank_code,
                transaction_type, is_biometric_verified, device_mac_hash, timestamp

        Returns:
            Dict mapping feature name → value, in the order expected by the model.
        """
        ref_ts = TransactionStore._to_epoch(tx.get("timestamp"))

        features = {}

        # ─── 1. Raw features (4) ─────────────────────────────────────────
        features["amount_vnd"] = float(tx["amount_vnd"])
        features["receiver_bank_code"] = tx["receiver_bank_code"]   # encoded later
        features["transaction_type"] = tx["transaction_type"]       # encoded later
        features["is_biometric_verified"] = int(tx["is_biometric_verified"])

        # ─── 2. Velocity features (14) ───────────────────────────────────
        features.update(self._velocity_features(tx, ref_ts))

        # ─── 3. Temporal features (9) ────────────────────────────────────
        features.update(self._temporal_features(tx, ref_ts))

        # ─── 4. Graph features (5) ───────────────────────────────────────
        features.update(self._graph_features(tx))

        return features

    def features_to_vector(self, features: Dict[str, float]) -> List[float]:
        """Convert feature dict to ordered vector matching model's expected input."""
        return [features[name] for name in FEATURE_ORDER]

    # ─── Velocity Features ───────────────────────────────────────────────

    def _velocity_features(self, tx: dict, ref_ts: float) -> dict:
        """Compute velocity features from sender's transaction history."""
        sender = tx["sender_cif"]
        amount = float(tx["amount_vnd"])

        # Get history at each window level
        hist_1h = self.store.get_sender_history(sender, WINDOWS["1h"], ref_ts)
        hist_6h = self.store.get_sender_history(sender, WINDOWS["6h"], ref_ts)
        hist_24h = self.store.get_sender_history(sender, WINDOWS["24h"], ref_ts)
        hist_7d = self.store.get_sender_history(sender, WINDOWS["7d"], ref_ts)

        # Transaction counts
        tx_count_1h = len(hist_1h) + 1   # +1 for current transaction
        tx_count_6h = len(hist_6h) + 1
        tx_count_24h = len(hist_24h) + 1
        tx_count_7d = len(hist_7d) + 1

        # Amount sums (include current transaction)
        amounts_1h = [t["amount_vnd"] for t in hist_1h] + [amount]
        amounts_6h = [t["amount_vnd"] for t in hist_6h] + [amount]
        amounts_24h = [t["amount_vnd"] for t in hist_24h] + [amount]
        amounts_7d = [t["amount_vnd"] for t in hist_7d] + [amount]

        tx_sum_1h = sum(amounts_1h)
        tx_sum_6h = sum(amounts_6h)
        tx_sum_24h = sum(amounts_24h)
        tx_sum_7d = sum(amounts_7d)

        # Averages
        tx_avg_1h = tx_sum_1h / tx_count_1h
        tx_avg_24h = tx_sum_24h / tx_count_24h

        # Max in 24h
        tx_max_24h = max(amounts_24h)

        # Z-score: how unusual is this amount relative to 24h history?
        if len(amounts_24h) > 1:
            mean_24h = np.mean(amounts_24h)
            std_24h = np.std(amounts_24h)
            amt_zscore_24h = (amount - mean_24h) / std_24h if std_24h > 0 else 0.0
        else:
            amt_zscore_24h = 0.0

        # Unique receivers and devices in 24h
        unique_receivers_24h = self.store.get_unique_receivers(
            sender, WINDOWS["24h"], ref_ts
        )
        # +1 for the current receiver if not already counted
        # (we haven't stored this tx yet at computation time)
        unique_receivers_24h += 1  # current transaction adds a potential new receiver

        unique_devices_24h = self.store.get_unique_devices(
            sender, WINDOWS["24h"], ref_ts
        )
        unique_devices_24h += 1  # current device

        return {
            "tx_count_1h": tx_count_1h,
            "tx_count_6h": tx_count_6h,
            "tx_count_24h": tx_count_24h,
            "tx_count_7d": tx_count_7d,
            "tx_sum_1h": tx_sum_1h,
            "tx_sum_6h": tx_sum_6h,
            "tx_sum_24h": tx_sum_24h,
            "tx_sum_7d": tx_sum_7d,
            "tx_avg_1h": tx_avg_1h,
            "tx_avg_24h": tx_avg_24h,
            "tx_max_24h": tx_max_24h,
            "amt_zscore_24h": amt_zscore_24h,
            "unique_receivers_24h": unique_receivers_24h,
            "unique_devices_24h": unique_devices_24h,
        }

    # ─── Temporal Features ───────────────────────────────────────────────

    def _temporal_features(self, tx: dict, ref_ts: float) -> dict:
        """Compute temporal features from the transaction timestamp."""
        ts_str = tx.get("timestamp")
        if ts_str:
            try:
                dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                dt = datetime.fromtimestamp(ref_ts)
        else:
            dt = datetime.fromtimestamp(ref_ts)

        hour = dt.hour
        day_of_week = dt.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        is_night = 1 if hour in (0, 1, 2, 3, 4, 5, 22, 23) else 0

        # Tết proximity
        is_tet_period, days_to_tet = self._tet_features(dt)

        # Time since last transaction
        sender = tx["sender_cif"]
        hist = self.store.get_sender_history(sender, WINDOWS["7d"], ref_ts)
        if hist:
            last_ts = max(t["timestamp"] for t in hist)
            time_since_last = ref_ts - last_ts
        else:
            time_since_last = 0.0

        # Cyclical encoding
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)

        return {
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "is_night": is_night,
            "is_tet_period": int(is_tet_period),
            "days_to_tet": days_to_tet,
            "time_since_last_tx": time_since_last,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
        }

    @staticmethod
    def _tet_features(dt: datetime):
        """Check if date is within 7 days of Tết and compute distance."""
        min_dist = float("inf")
        for tet_date in TET_DATES:
            dist = abs((dt - tet_date).days)
            if dist < min_dist:
                min_dist = dist

        is_tet = min_dist <= 7
        return is_tet, min_dist

    # ─── Graph Features ──────────────────────────────────────────────────

    def _graph_features(self, tx: dict) -> dict:
        """Compute graph/network features from transaction relationships."""
        sender = tx["sender_cif"]
        receiver = tx["receiver_cif"]

        sender_out_degree = self.store.get_sender_out_degree(sender) + 1  # +1 current
        receiver_in_degree = self.store.get_receiver_in_degree(receiver) + 1
        sender_bank_diversity = self.store.get_sender_bank_diversity(sender) + 1
        pair_tx_count = self.store.get_pair_count(sender, receiver) + 1  # +1 current
        is_repeat_pair = 1 if pair_tx_count > 1 else 0

        return {
            "sender_out_degree": sender_out_degree,
            "receiver_in_degree": receiver_in_degree,
            "sender_bank_diversity": sender_bank_diversity,
            "pair_tx_count": pair_tx_count,
            "is_repeat_pair": is_repeat_pair,
        }

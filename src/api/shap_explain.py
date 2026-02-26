"""
SHAP Explanation Service — generates human-readable feature contribution
explanations for each fraud prediction using TreeExplainer.
"""
import random
import logging
import numpy as np
import xgboost as xgb
import shap

logger = logging.getLogger(__name__)

# Human-readable descriptions for feature contributions
FEATURE_DESCRIPTIONS = {
    "pair_tx_count": {
        "fraud": "first-time sender→receiver pair (no prior relationship)",
        "legit": "recurring sender→receiver pair (established relationship)",
    },
    "unique_devices_24h": {
        "fraud": "multiple devices used recently (unusual device switching)",
        "legit": "consistent device usage",
    },
    "is_biometric_verified": {
        "fraud": "biometric verification was bypassed",
        "legit": "biometric verification passed",
    },
    "amount_vnd": {
        "fraud": "unusual transaction amount",
        "legit": "normal transaction amount",
    },
    "transaction_type": {
        "fraud": "transaction channel commonly used in fraud",
        "legit": "normal transaction channel",
    },
    "tx_count_1h": {
        "fraud": "high transaction velocity in the last hour",
        "legit": "normal transaction frequency",
    },
    "tx_count_24h": {
        "fraud": "high transaction volume in the last 24 hours",
        "legit": "normal daily transaction volume",
    },
    "amt_zscore_24h": {
        "fraud": "amount is abnormally high vs sender's recent history",
        "legit": "amount is within sender's normal range",
    },
    "is_night": {
        "fraud": "transaction occurred during late-night hours",
        "legit": "transaction during normal business hours",
    },
    "hour": {
        "fraud": "unusual hour for this type of transaction",
        "legit": "typical hour for transactions",
    },
    "is_repeat_pair": {
        "fraud": "first transaction between this sender and receiver",
        "legit": "previously established sender→receiver pair",
    },
    "sender_out_degree": {
        "fraud": "sender transacts with unusually many recipients",
        "legit": "sender has a normal number of recipients",
    },
    "receiver_in_degree": {
        "fraud": "receiver has few incoming connections",
        "legit": "receiver has many incoming connections (established)",
    },
    "sender_bank_diversity": {
        "fraud": "sender uses many different bank channels",
        "legit": "sender uses consistent bank channels",
    },
    "time_since_last_tx": {
        "fraud": "very short time since sender's last transaction",
        "legit": "normal gap between transactions",
    },
    "is_tet_period": {
        "fraud": "transaction during Tết period (higher fraud risk)",
        "legit": "transaction outside high-risk holiday period",
    },
    "receiver_bank_code": {
        "fraud": "receiver bank associated with fraud patterns",
        "legit": "receiver bank is a standard institution",
    },
    "unique_receivers_24h": {
        "fraud": "sender sent to many different recipients today",
        "legit": "sender has few recipients today",
    },
    "tx_sum_24h": {
        "fraud": "high total amount sent in the last 24 hours",
        "legit": "normal total daily spending",
    },
    "tx_max_24h": {
        "fraud": "large single transaction in the last 24 hours",
        "legit": "no unusually large transactions recently",
    },
    "days_to_tet": {
        "fraud": "close to Tết holiday (seasonal fraud risk)",
        "legit": "far from high-risk holiday period",
    },
}

# Default descriptions for features not explicitly listed above
DEFAULT_DESC = {
    "fraud": "contributing to fraud signal",
    "legit": "within normal patterns",
}


class ShapExplainer:
    """Generates SHAP explanations for XGBoost predictions."""

    def __init__(self, xgb_model: xgb.Booster, feature_names: list):
        self.feature_names = feature_names or [f"feat_{i}" for i in range(32)]
        if xgb_model and hasattr(xgb_model, 'predict'):
            try:
                self.explainer = shap.TreeExplainer(xgb_model)
                logger.info("SHAP TreeExplainer initialized with %d features", len(self.feature_names))
            except Exception as e:
                logger.error("Failed to initialize SHAP TreeExplainer: %s. Using MOCK.", e)
                self.explainer = None
        else:
            self.explainer = None
            logger.info("SHAP initialized in MOCK mode (no model provided)")

    def explain(self, feature_vector: list, top_n: int = 5) -> list:
        """Generate top-N SHAP feature contributions."""
        if self.explainer is None:
            # Return random mock explanations for stability
            return [{
                "feature": self.feature_names[i],
                "value": float(feature_vector[i]),
                "contribution": float(np.random.uniform(-1, 1)),
                "direction": "fraud" if random.random() > 0.5 else "legit",
                "description": "mock contribution"
            } for i in range(min(top_n, len(self.feature_names)))]

        dmatrix = xgb.DMatrix([feature_vector], feature_names=self.feature_names)
        shap_values = self.explainer.shap_values(dmatrix)

        if isinstance(shap_values, list):
            # For multi-class, take the positive class
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif len(shap_values.shape) > 1:
            sv = shap_values[0]
        else:
            sv = shap_values

        # Rank by absolute SHAP value
        indices = np.argsort(np.abs(sv))[::-1][:top_n]

        explanations = []
        for idx in indices:
            name = self.feature_names[idx]
            value = float(feature_vector[idx])
            contribution = float(sv[idx])
            direction = "fraud" if contribution > 0 else "legit"

            # Get human-readable description
            descs = FEATURE_DESCRIPTIONS.get(name, DEFAULT_DESC)
            description = descs.get(direction, DEFAULT_DESC[direction])

            explanations.append({
                "feature": name,
                "value": round(value, 4),
                "contribution": round(contribution, 4),
                "direction": direction,
                "description": description,
            })

        return explanations

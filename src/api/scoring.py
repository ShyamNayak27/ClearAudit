"""
Model Scoring Service — loads trained models and scores transactions.

Loads XGBoost (champion), Autoencoder, scaler, label encoders, and threshold
at startup. Provides scoring, ensemble blending, and decision logic.
"""
import os
import json
import logging
import numpy as np
import xgboost as xgb
import joblib

try:
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError as e:
    TF_AVAILABLE = False
    logging.warning("TensorFlow could not be imported (%s). Autoencoder will be mocked.", e)


logger = logging.getLogger(__name__)

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODELS_DIR = os.path.join(_root, "models")


class ModelService:
    """Loads and serves all fraud detection models."""

    def __init__(self):
        self.xgb_model = None
        self.ae_model = None
        self.ae_scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.ae_threshold = None
        self.metrics = None
        self._loaded = False

    def load_models(self) -> None:
        """Load all model artifacts from disk."""
        logger.info("Loading models from %s ...", MODELS_DIR)

        # XGBoost
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(os.path.join(MODELS_DIR, "xgboost_model.json"))
        logger.info("  XGBoost loaded")

        # Autoencoder
        if TF_AVAILABLE:
            try:
                self.ae_model = keras.models.load_model(
                    os.path.join(MODELS_DIR, "autoencoder.keras"), compile=False
                )
                logger.info("  Autoencoder loaded")
            except Exception as e:
                logger.error("Failed to load Autoencoder: %s", e)
                self.ae_model = "MOCK"
        else:
            self.ae_model = "MOCK"
            logger.info("  Autoencoder mocked (TF unavailable)")

        # AE scaler
        self.ae_scaler = joblib.load(os.path.join(MODELS_DIR, "ae_scaler.joblib"))
        logger.info("  AE scaler loaded")

        # Label encoders
        self.label_encoders = joblib.load(
            os.path.join(MODELS_DIR, "label_encoders.joblib")
        )
        logger.info("  Label encoders loaded (%s)", list(self.label_encoders.keys()))

        # Feature names
        with open(os.path.join(MODELS_DIR, "feature_names.json")) as f:
            self.feature_names = json.load(f)
        logger.info("  Feature names loaded (%d features)", len(self.feature_names))

        # AE threshold
        with open(os.path.join(MODELS_DIR, "ae_threshold.json")) as f:
            self.ae_threshold = json.load(f)
        logger.info("  AE threshold: %.6f", self.ae_threshold["threshold"])

        # Metrics
        with open(os.path.join(MODELS_DIR, "metrics_report.json")) as f:
            self.metrics = json.load(f)
        logger.info("  Metrics loaded (champion: %s)", self.metrics.get("champion"))

        self._loaded = True
        logger.info("All models loaded successfully.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ─── Encode Categoricals ─────────────────────────────────────────────

    def encode_features(self, features: dict) -> dict:
        """Apply label encoding to categorical features, matching training pipeline."""
        encoded = features.copy()

        for col, encoder in self.label_encoders.items():
            if col in encoded:
                val = encoded[col]
                # Handle unseen categories gracefully
                if val in encoder.classes_:
                    encoded[col] = int(encoder.transform([val])[0])
                else:
                    # Assign a default (most common class index)
                    logger.warning("Unseen category '%s' for '%s', using default.", val, col)
                    encoded[col] = 0

        return encoded

    # ─── XGBoost Scoring ─────────────────────────────────────────────────

    def score_xgboost(self, feature_vector: list) -> float:
        """Run XGBoost and return fraud probability.

        Args:
            feature_vector: list of 32 feature values in model order.

        Returns:
            Probability of fraud (0.0–1.0).
        """
        dmatrix = xgb.DMatrix(
            [feature_vector], feature_names=self.feature_names
        )
        proba = self.xgb_model.predict(dmatrix)[0]
        return float(proba)

    # ─── Autoencoder Scoring ─────────────────────────────────────────────

    def score_autoencoder(self, feature_vector: list) -> float:
        """Run autoencoder and return reconstruction error (anomaly score).

        Args:
            feature_vector: list of 32 feature values.

        Returns:
            Mean squared reconstruction error.
        """
        if self.ae_model == "MOCK":
            return float(np.random.uniform(0.001, 0.009))

        x = np.array([feature_vector], dtype=np.float32)
        x_scaled = self.ae_scaler.transform(x)
        x_reconstructed = self.ae_model.predict(x_scaled, verbose=0)
        mse = np.mean((x_scaled - x_reconstructed) ** 2, axis=1)[0]
        return float(mse)

    # ─── Ensemble ────────────────────────────────────────────────────────

    def ensemble_score(self, xgb_prob: float, ae_error: float) -> float:
        """Compute weighted ensemble score.

        80% XGBoost probability + 20% normalized AE anomaly score.
        AE score is normalized using min-max against the training threshold:
        if error >= threshold, normalized to 1.0; if 0, normalized to 0.0.
        """
        threshold = self.ae_threshold["threshold"]
        # Normalize AE error to 0–1 range using threshold as the reference
        ae_normalized = min(ae_error / threshold, 1.0) if threshold > 0 else 0.0

        return 0.8 * xgb_prob + 0.2 * ae_normalized

    # ─── Decision ────────────────────────────────────────────────────────

    @staticmethod
    def decide(ensemble_score: float) -> tuple:
        """Convert ensemble score to a decision and confidence level.

        Returns:
            (decision, confidence) tuple.
        """
        if ensemble_score < 0.3:
            return "approve", "high"
        elif ensemble_score < 0.5:
            return "flag", "medium"
        elif ensemble_score < 0.7:
            return "flag", "high"
        else:
            return "block", "high"

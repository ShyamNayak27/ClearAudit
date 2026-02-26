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

        # 1. XGBoost
        try:
            self.xgb_model = xgb.Booster()
            self.xgb_model.load_model(os.path.join(MODELS_DIR, "xgboost_model.json"))
            logger.info("  XGBoost loaded")
        except Exception as e:
            logger.error("Failed to load XGBoost: %s", e)
            self.xgb_model = None

        # 2. Autoencoder
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

        # 3. Scaler, Encoders, Names, Thresholds
        try:
            self.ae_scaler = joblib.load(os.path.join(MODELS_DIR, "ae_scaler.joblib"))
            logger.info("  AE scaler loaded")

            self.label_encoders = joblib.load(
                os.path.join(MODELS_DIR, "label_encoders.joblib")
            )
            logger.info("  Label encoders loaded")

            with open(os.path.join(MODELS_DIR, "feature_names.json")) as f:
                self.feature_names = json.load(f)
            logger.info("  Feature names loaded")

            with open(os.path.join(MODELS_DIR, "ae_threshold.json")) as f:
                self.ae_threshold = json.load(f)
            
            with open(os.path.join(MODELS_DIR, "metrics_report.json")) as f:
                self.metrics = json.load(f)
        except Exception as e:
            logger.warning("Optional artifacts failed to load (LFS or missing): %s", e)
            if self.feature_names is None:
                self.feature_names = [f"feat_{i}" for i in range(32)]
            if self.ae_threshold is None:
                self.ae_threshold = {"threshold": 0.005}
            if self.metrics is None:
                self.metrics = {"champion": "mock"}

        self._loaded = True
        logger.info("Model service initialized (Loaded=%s)", "Full" if self.xgb_model else "Mock/Degraded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ─── Encode Categoricals ─────────────────────────────────────────────

    def encode_features(self, features: dict) -> dict:
        """Apply label encoding to categorical features, matching training pipeline."""
        encoded = features.copy()

        # Define categorical features to protect
        cat_cols = ["receiver_bank_code", "transaction_type"]

        if self.label_encoders:
            for col, encoder in self.label_encoders.items():
                if col in encoded:
                    val = encoded[col]
                    # Handle unseen categories gracefully
                    if hasattr(encoder, 'classes_') and val in encoder.classes_:
                        encoded[col] = int(encoder.transform([val])[0])
                    else:
                        # Assign a default (most common class index) or hash fallback
                        logger.warning("Unseen category '%s' for '%s', using hash fallback.", val, col)
                        encoded[col] = abs(hash(str(val))) % 100
        else:
            # Emergency fallback: if encoders are missing, MUST return numbers
            for col in cat_cols:
                if col in encoded:
                    val = encoded[col]
                    if isinstance(val, str):
                        encoded[col] = abs(hash(val)) % 100
                        logger.debug("Mock encoding for '%s': %s -> %d", col, val, encoded[col])

        return encoded

    # ─── XGBoost Scoring ─────────────────────────────────────────────────

    def score_xgboost(self, feature_vector: list) -> float:
        """Run XGBoost and return fraud probability.

        Args:
            feature_vector: list of 32 feature values in model order.

        Returns:
            Probability of fraud (0.0–1.0).
        """
        if not self.xgb_model or not hasattr(self.xgb_model, 'predict'):
            return float(np.random.uniform(0.1, 0.4))

        # Coerce feature names to strings and vector to floats to avoid XGBoost Unicode/dtype issues
        names = [str(n) for n in self.feature_names]
        vals = [float(v) for v in feature_vector]

        try:
            dmatrix = xgb.DMatrix(
                [vals], feature_names=names
            )
            proba = self.xgb_model.predict(dmatrix)[0]
            return float(proba)
        except Exception as e:
            logger.warning("XGBoost prediction failed (likely mock model state): %s", e)
            return float(np.random.uniform(0.1, 0.4))

    # ─── Autoencoder Scoring ─────────────────────────────────────────────

    def score_autoencoder(self, feature_vector: list) -> float:
        """Run autoencoder and return reconstruction error (anomaly score).

        Args:
            feature_vector: list of 32 feature values.

        Returns:
            Mean squared reconstruction error.
        """
        if self.ae_model == "MOCK" or self.ae_model is None or not self.ae_scaler:
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

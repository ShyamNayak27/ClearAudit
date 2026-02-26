"""
Vietnam Fraud Detector — FastAPI Application.

Provides real-time fraud scoring for Vietnamese banking transactions using
an XGBoost + Autoencoder ensemble with Redis-powered feature computation
and SHAP-based explanations.

Usage:
    cd <project_root>
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    TransactionRequest,
    ScoringResponse,
    HealthResponse,
    ModelInfoResponse,
    FeatureContribution,
)
from .redis_store import TransactionStore
from .feature_service import FeatureService
from .scoring import ModelService
from .shap_explain import ShapExplainer
from .monitoring import DriftMonitor

# ─── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("fraud_api")

# ─── Global Services ─────────────────────────────────────────────────────
store = TransactionStore(redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"))
feature_service = FeatureService(store)
model_service = ModelService()
shap_explainer: ShapExplainer = None  # initialized after model load
drift_monitor = DriftMonitor()


def setup_app_services():
    """Manually initialize all global services (used by unified servers)."""
    global shap_explainer
    
    if not model_service.is_loaded:
        model_service.load_models()
    
    if shap_explainer is None:
        shap_explainer = ShapExplainer(model_service.xgb_model, model_service.feature_names)
    
    logger.info("Global services initialized. Redis: %s", 
                "connected" if store.is_redis_connected else "in-memory fallback")

# ─── Lifespan ────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, cleanup at shutdown."""
    logger.info("=" * 60)
    logger.info("  VIETNAM FRAUD DETECTOR — API Starting")
    logger.info("=" * 60)

    setup_app_services()

    logger.info("=" * 60)
    yield
    logger.info("API shutting down.")


# ─── FastAPI App ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Vietnam Fraud Detector",
    description=(
        "Real-time fraud scoring API for Vietnamese banking transactions. "
        "Accepts raw transactions and returns fraud probability, decision, "
        "and SHAP-based explanations using an XGBoost + Autoencoder ensemble."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ───────────────────────────────────────────────────────────

@app.post("/score", response_model=ScoringResponse)
async def score_transaction(tx: TransactionRequest):
    """Score a transaction for fraud.

    Accepts a raw transaction, computes all 32 features from Redis history,
    runs XGBoost + Autoencoder ensemble, and returns a decision with SHAP explanations.
    """
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    tx_dict = tx.model_dump()
    tx_id = f"TXN_{uuid.uuid4().hex[:12].upper()}"

    # 1. Compute features from Redis history
    raw_features = feature_service.compute_features(tx_dict)

    # 2. Encode categoricals (receiver_bank_code, transaction_type)
    encoded_features = model_service.encode_features(raw_features)

    # 3. Build ordered feature vector
    feature_vector = feature_service.features_to_vector(encoded_features)

    # 4. Score with XGBoost
    xgb_prob = model_service.score_xgboost(feature_vector)

    # 5. Score with Autoencoder
    ae_error = model_service.score_autoencoder(feature_vector)

    # 6. Ensemble
    ensemble = model_service.ensemble_score(xgb_prob, ae_error)

    # 7. Decision
    decision, confidence = model_service.decide(ensemble)

    # 8. SHAP explanation
    shap_raw = shap_explainer.explain(feature_vector, top_n=5)
    top_features = [FeatureContribution(**s) for s in shap_raw]

    # 9. Store transaction in Redis for future history
    store.store_transaction(tx_dict)

    # 10. Feed production sample to drift monitor
    drift_monitor.add_production_sample(encoded_features)

    # 11. Populate Diagnostics (for analyst comparison)
    diagnostics = {
        "tx_1h": int(raw_features["tx_count_1h"]),
        "tx_24h": int(raw_features["tx_count_24h"]),
        "receivers_24h": int(raw_features["unique_receivers_24h"]),
        "devices_24h": int(raw_features["unique_devices_24h"]),
        "bank_diversity": int(raw_features["sender_bank_diversity"]),
        "pair_count": int(raw_features["pair_tx_count"]),
        "time_since_last": round(raw_features["time_since_last_tx"], 1),
        "is_night": bool(raw_features["is_night"]),
    }

    logger.info(
        "Scored %s: ensemble=%.3f decision=%s (XGB=%.3f, AE=%.4f)",
        tx_id, ensemble, decision, xgb_prob, ae_error,
    )

    return ScoringResponse(
        transaction_id=tx_id,
        fraud_score=round(ensemble, 4),
        xgboost_score=round(xgb_prob, 4),
        ae_anomaly_score=round(ae_error, 6),
        decision=decision,
        confidence=confidence,
        diagnostics=diagnostics,
        top_features=top_features,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health — models loaded, Redis connected, key metrics."""
    metrics = model_service.metrics or {}
    xgb_metrics = metrics.get("models", {}).get("xgboost", {})
    ensemble_metrics = metrics.get("models", {}).get("ensemble", {})

    return HealthResponse(
        status="healthy" if model_service.is_loaded else "degraded",
        models_loaded=model_service.is_loaded,
        redis_connected=store.is_redis_connected,
        xgboost_pr_auc=xgb_metrics.get("pr_auc"),
        ensemble_f1=ensemble_metrics.get("f1"),
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get model metadata — champion, features, metrics, thresholds."""
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    return ModelInfoResponse(
        champion=model_service.metrics.get("champion", "unknown"),
        feature_count=len(model_service.feature_names),
        feature_names=model_service.feature_names,
        metrics=model_service.metrics.get("models", {}),
        ae_threshold=model_service.ae_threshold["threshold"],
        ensemble_weights={"xgboost": 0.8, "autoencoder": 0.2},
    )


@app.get("/monitoring/drift-report")
async def get_drift_report():
    """Get the latest drift monitoring report."""
    report = drift_monitor.get_last_report()
    if not report:
        return {
            "status": "no_report",
            "message": "No drift report available yet. Need at least 30 scored transactions.",
            "buffer_size": drift_monitor.buffer_size,
        }
    return report


@app.post("/monitoring/run-check")
async def run_drift_check():
    """Manually trigger a drift check against the current production buffer."""
    result = drift_monitor.run_drift_check()
    if result is None:
        return {
            "status": "insufficient_data",
            "message": f"Need at least 30 transactions. Current buffer: {drift_monitor.buffer_size}",
        }
    return {"status": "completed", "report": result}

"""
Pydantic models for the fraud detection API request/response contracts.
"""
from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class TransactionRequest(BaseModel):
    """Raw transaction submitted for fraud scoring."""

    amount_vnd: float = Field(..., gt=0, description="Transaction amount in VND")
    sender_cif: str = Field(..., description="Sender customer ID (e.g., VN_12345678)")
    receiver_cif: str = Field(..., description="Receiver customer ID")
    receiver_bank_code: str = Field(
        ...,
        description="Receiver bank code (VCB, TCB, BIDV, VPB, MBB, ACB, MOMO, ZALOPAY, VNPAY)"
    )
    transaction_type: str = Field(
        ...,
        description="Transaction type (P2P_TRANSFER, E_COMMERCE, QR_PAYMENT, BILL_PAYMENT)"
    )
    is_biometric_verified: bool = Field(
        ..., description="Whether biometric verification was passed"
    )
    device_mac_hash: str = Field(..., description="Hashed MAC address of the device")
    timestamp: Optional[str] = Field(
        None,
        description="ISO timestamp (YYYY-MM-DD HH:MM:SS). Defaults to current time if omitted."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "amount_vnd": 35000,
                    "sender_cif": "VN_99887766",
                    "receiver_cif": "VN_11223344",
                    "receiver_bank_code": "MOMO",
                    "transaction_type": "P2P_TRANSFER",
                    "is_biometric_verified": False,
                    "device_mac_hash": "a1b2c3d4e5f6",
                    "timestamp": "2025-03-15 02:30:00"
                }
            ]
        }
    }


class Decision(str, Enum):
    """Fraud decision categories."""
    APPROVE = "approve"
    FLAG = "flag"
    BLOCK = "block"


class FeatureContribution(BaseModel):
    """A single SHAP feature contribution."""
    feature: str
    value: float = Field(..., description="The feature's actual value for this transaction")
    contribution: float = Field(..., description="SHAP value — positive pushes toward fraud")
    direction: str = Field(..., description="'fraud' or 'legit'")
    description: str = Field(..., description="Human-readable explanation")


class ScoringResponse(BaseModel):
    """Full fraud scoring response."""
    transaction_id: str = Field(..., description="Generated transaction ID for tracking")
    fraud_score: float = Field(..., ge=0, le=1, description="Ensemble fraud probability (0–1)")
    xgboost_score: float = Field(..., ge=0, le=1, description="XGBoost probability")
    ae_anomaly_score: float = Field(..., ge=0, description="Autoencoder reconstruction error")
    decision: Decision = Field(..., description="approve / flag / block")
    confidence: str = Field(..., description="low / medium / high")
    diagnostics: dict = Field(..., description="Key feature values for manual comparison")
    top_features: List[FeatureContribution] = Field(
        ..., description="Top 5 SHAP feature contributions"
    )


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    models_loaded: bool
    redis_connected: bool
    xgboost_pr_auc: Optional[float] = None
    ensemble_f1: Optional[float] = None


class ModelInfoResponse(BaseModel):
    """Model metadata response."""
    champion: str
    feature_count: int
    feature_names: List[str]
    metrics: dict
    ae_threshold: float
    ensemble_weights: dict

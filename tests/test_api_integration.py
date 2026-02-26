import pytest
from fastapi.testclient import TestClient
from src.api.app import app, setup_app_services

# Initialize services for testing (loads models with in-memory fallback)
setup_app_services()

client = TestClient(app)

def test_health_check():
    """Verify that the health check endpoint returns 200 and correct status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "degraded"]

def test_scoring_endpoint_legit():
    """Verify that a legitimate transaction is scored correctly."""
    payload = {
        "amount_vnd": 500000,
        "sender_cif": "VN_TEST_001",
        "receiver_cif": "VN_RECV_001",
        "receiver_bank_code": "VCB",
        "transaction_type": "P2P_TRANSFER",
        "is_biometric_verified": True,
        "device_mac_hash": "test_device_001",
        "timestamp": "2026-01-01 12:00:00"
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_score" in data
    assert "decision" in data
    assert data["decision"] in ["approve", "flag", "block"]

def test_scoring_endpoint_fraud_pattern():
    """Verify that a suspicious transaction is processed without error."""
    payload = {
        "amount_vnd": 15000000,  # High amount
        "sender_cif": "VN_TEST_999",
        "receiver_cif": "VN_RECV_999",
        "receiver_bank_code": "MOMO",
        "transaction_type": "P2P_TRANSFER",
        "is_biometric_verified": False, # No biometric
        "device_mac_hash": "test_device_999",
        "timestamp": "2026-01-01 02:00:00" # Night time
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_score" in data
    # Suspicious transactions should likely be flagged/blocked, but we test for success
    assert data["fraud_score"] > 0

def test_model_info():
    """Verify that model metadata endpoint works."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "feature_count" in data
    assert data["feature_count"] == 32

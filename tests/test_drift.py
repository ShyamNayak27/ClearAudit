"""Batch test — sends 40 random transactions to test drift monitoring."""
import requests
import json
import random

BASE = "http://127.0.0.1:8000"
BANKS = ["VCB", "TCB", "BIDV", "VPB", "MBB", "ACB", "MOMO", "ZALOPAY", "VNPAY"]
TX_TYPES = ["P2P_TRANSFER", "E_COMMERCE", "QR_PAYMENT", "BILL_PAYMENT"]

def run_drift_test():
    print("Sending 40 transactions to fill drift buffer...")
    for i in range(40):
        tx = {
            "amount_vnd": random.choice([35000, 199000, 500000, 2000000, 9500000]),
            "sender_cif": f"VN_BATCH_{random.randint(1, 20):04d}",
            "receiver_cif": f"VN_RECV_{random.randint(1, 50):04d}",
            "receiver_bank_code": random.choice(BANKS),
            "transaction_type": random.choice(TX_TYPES),
            "is_biometric_verified": random.random() > 0.3,
            "device_mac_hash": f"dev_{random.randint(1, 10)}",
            "timestamp": f"2025-03-15 {random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00",
        }
        r = requests.post(f"{BASE}/score", json=tx)
        d = r.json()["decision"]
        print(f"  [{i+1:2d}] amount={tx['amount_vnd']:>10,}  biometric={str(tx['is_biometric_verified']):5s} -> {d}")

    print("\nTriggering drift check...")
    r = requests.post(f"{BASE}/monitoring/run-check")
    result = r.json()
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    run_drift_test()

""" 
Refined API test — sends 10-20 random transactions with realistic and fraud-labeled data 
to verify API robustness and scoring logic.
"""
import requests
import json
import random
import time

BASE = "http://127.0.0.1:8000"
BANKS = ["VCB", "TCB", "BIDV", "VPB", "MBB", "ACB", "MOMO", "ZALOPAY", "VNPAY"]
TX_TYPES = ["P2P_TRANSFER", "E_COMMERCE", "QR_PAYMENT", "BILL_PAYMENT"]

def generate_random_transaction():
    """Generates a random transaction, occasionally injecting known fraud patterns."""
    rand_val = random.random()
    
    # 20% chance of a "suspicious" high amount/low velocity/no biometric case
    if rand_val < 0.2:
        return {
            "amount_vnd": random.randint(9_000_000, 15_000_000), # Large single tx
            "sender_cif": f"VN_{random.randint(100, 200)}",
            "receiver_cif": f"VN_{random.randint(500, 600)}",
            "receiver_bank_code": random.choice(BANKS),
            "transaction_type": "P2P_TRANSFER",
            "is_biometric_verified": False, # Often flagged
            "device_mac_hash": f"dev_{random.randint(1000, 9999)}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    # 20% chance of a "small P2P" (Fake Shipper style)
    elif rand_val < 0.4:
         return {
            "amount_vnd": random.randint(30_000, 50_000), 
            "sender_cif": f"VN_{random.randint(201, 300)}",
            "receiver_cif": f"VN_{random.randint(601, 700)}",
            "receiver_bank_code": random.choice(["MOMO", "ZALOPAY"]),
            "transaction_type": "P2P_TRANSFER",
            "is_biometric_verified": False,
            "device_mac_hash": f"dev_{random.randint(1000, 9999)}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    # Default legitimate-looking transaction
    else:
        return {
            "amount_vnd": random.randint(50_000, 1_000_000),
            "sender_cif": f"VN_{random.randint(301, 500)}",
            "receiver_cif": f"VN_{random.randint(701, 999)}",
            "receiver_bank_code": random.choice(BANKS),
            "transaction_type": random.choice(TX_TYPES),
            "is_biometric_verified": True,
            "device_mac_hash": f"dev_{random.randint(1000, 9999)}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

def run_tests():
    num_tests = random.randint(10, 20)
    print(f"=== RUNNING {num_tests} RANDOM TRANSACTION TESTS ===")
    
    try:
        # Check health first
        requests.get(f"{BASE}/health", timeout=2)
    except:
        print(f"Error: API is not running at {BASE}")
        return

    results = []
    for i in range(num_tests):
        tx = generate_random_transaction()
        r = requests.post(f"{BASE}/score", json=tx)
        if r.status_code == 200:
            res = r.json()
            decision = res['decision']
            score = res['fraud_score']
            results.append((decision, score))
            print(f"Test {i+1:02d}: Amount={tx['amount_vnd']:>10,} | Decision={decision:7s} | Score={score:.4f}")
            
            # Show key signals for comparison (from diagnostics)
            diag = res.get("diagnostics", {})
            signals = []
            if "tx_1h" in diag: signals.append(f"1h_tx={diag['tx_1h']}")
            if "devices_24h" in diag: signals.append(f"devices={diag['devices_24h']}")
            if "pair_count" in diag: signals.append(f"pair={diag['pair_count']}")
            if "is_night" in diag: signals.append(f"night={diag['is_night']}")
            
            print(f"    Signals: {', '.join(signals)} | Type={tx['transaction_type']}")

            # Show explanations for non-approved transactions
            if decision != "approve":
                print("    Reasons:")
                for feat in res.get("top_features", []):
                    # Only show fraud-contributing features for flags/blocks
                    if feat["direction"] == "fraud":
                        icon = "🔴"
                        # Include the actual value in the output
                        val = feat.get('value', 'N/A')
                        print(f"      {icon} {feat['feature']:20s} (val={val}) : {feat['description']}")
                print("-" * 30)
        else:
            print(f"Test {i+1:02d}: FAILED with status {r.status_code}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Total Tests: {len(results)}")
    print(f"Approve: {sum(1 for d, s in results if d == 'approve')}")
    print(f"Flag:    {sum(1 for d, s in results if d == 'flag')}")
    print(f"Block:   {sum(1 for d, s in results if d == 'block')}")

if __name__ == "__main__":
    run_tests()

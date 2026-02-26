import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import uuid
import hashlib
import os

# Set seed for reproducibility
np.random.seed(42)
fake = Faker('vi_VN') # Use Vietnamese locale

# ----------------- CONFIGURATIONS -----------------
NUM_USERS = 5000
NUM_TRANSACTIONS = 100000 
START_DATE = datetime(2026, 2, 1)
END_DATE = datetime(2026, 2, 28)

# Tết 2026 exact dates
TET_START = datetime(2026, 2, 14)
TET_END = datetime(2026, 2, 22)

# Specific to the Vietnamese Financial Ecosystem (From PDF Page 7)
BANKS = ['VCB', 'TCB', 'BIDV', 'VPB', 'ACB', 'MBB', 'TPB', 'STB', 'MOMO', 'ZALOPAY', 'VNPAY']
TX_TYPES = ['P2P_TRANSFER', 'BILL_PAYMENT', 'QR_PAYMENT', 'LI_XI', 'E_COMMERCE']
LUCKY_AMOUNTS = [50000, 100000, 200000, 500000, 1000000, 2000000, 5000000]

# ----------------- HELPER FUNCTIONS -----------------
def generate_users(num_users):
    """Generate a pool of CIFs (Customer Information Files) and MAC hashes."""
    users =[]
    for _ in range(num_users):
        cif = f"CIF{fake.random_number(digits=8, fix_len=True)}"
        mac = hashlib.sha256(fake.mac_address().encode()).hexdigest()
        users.append({'cif': cif, 'mac': mac})
    return users

def get_transaction_amount(tx_type, is_tet):
    """Determine transaction amount based on type and seasonality."""
    if tx_type == 'LI_XI':
        return np.random.choice(LUCKY_AMOUNTS)
    
    # Base log-normal distribution for normal payments
    amount = np.random.lognormal(mean=12.5, sigma=1.5) 
    
    # Inject Biometric Evasion Fraud (Smurfing just below 10M VND)
    if np.random.rand() < 0.02:  # 2% chance of evasion attempt
        amount = np.random.uniform(9000000, 9999999)
        
    # Inject Fake Shipper scam (30k - 50k VND)
    if np.random.rand() < 0.03: 
        amount = np.random.uniform(30000, 50000)

    # Inject Quishing amount pattern (200k - 2M VND, typical phishing lure)
    if np.random.rand() < 0.02:
        amount = np.random.choice([199000, 299000, 499000, 999000, 1999000])

    return round(amount, 0)

# ----------------- SIMULATION ENGINE -----------------
def simulate_vietnam_transactions():
    print("Generating user pool...")
    users = generate_users(NUM_USERS)
    
    transactions =[]
    current_time = START_DATE
    
    print(f"Simulating {NUM_TRANSACTIONS} transactions...")
    for i in range(NUM_TRANSACTIONS):
        # 1. Temporal Seasonality Injection (HMM proxy)
        if TET_START <= current_time <= TET_END:
            time_step = timedelta(seconds=np.random.randint(1, 30))
            is_tet = True
            # During Tet: 40% Li Xi, drop in Bill Payments
            tx_type_weights = [0.20, 0.05, 0.15, 0.40, 0.20]
        else:
            time_step = timedelta(seconds=np.random.randint(10, 300))
            is_tet = False
            # Normal Days: Very rare Li Xi
            tx_type_weights = [0.30, 0.25, 0.25, 0.02, 0.18]
            
        current_time += time_step
        if current_time > END_DATE:
             break 
             
        # 2. Assign attributes
        sender = np.random.choice(users)
        receiver = np.random.choice(users)
        while sender == receiver: 
            receiver = np.random.choice(users)
            
        tx_type = np.random.choice(TX_TYPES, p=tx_type_weights)
        amount = get_transaction_amount(tx_type, is_tet)
        
        # Determine Bank (Force E-wallets for Li Xi as per PDF page 2)
        if tx_type == 'LI_XI':
            receiver_bank = np.random.choice(
                ['MOMO', 'ZALOPAY', 'VNPAY'],
                p=[0.50, 0.30, 0.20]
            )
        else:
            receiver_bank = np.random.choice(BANKS)

        # 3. Regulatory Logic: SBV Decision 2345
        is_biometric_verified = bool(amount > 10000000)
        
        # 4. Target Variable (Fraud labels + fraud_type)
        is_fraud = 0
        fraud_type = 'none'
        
        # Fake Shipper Scams (Late night, specific amount, P2P)
        if 0 <= current_time.hour <= 5 and 30000 <= amount <= 50000 and tx_type == 'P2P_TRANSFER':
            is_fraud = 1
            fraud_type = 'fake_shipper'
            
        # Quishing (QR phishing: QR_PAYMENT, late-night, lure amounts)
        elif tx_type == 'QR_PAYMENT' and 0 <= current_time.hour <= 6 and amount in [199000, 299000, 499000, 999000, 1999000]:
            is_fraud = 1
            fraud_type = 'quishing'
            
        # Biometric Evasion / Smurfing (Amounts strictly between 9M and 9.99M)
        elif 9000000 <= amount <= 9999999 and np.random.rand() < 0.15:
            is_fraud = 1
            fraud_type = 'biometric_evasion'
            
        # 5. Append record matching the schema on PDF Page 6/7
        transactions.append({
            'transaction_id': str(uuid.uuid4()),
            'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'amount_vnd': float(amount),
            'sender_cif': sender['cif'],
            'receiver_cif': receiver['cif'],
            'receiver_bank_code': receiver_bank,
            'transaction_type': tx_type,
            'is_biometric_verified': is_biometric_verified,
            'device_mac_hash': sender['mac'],
            'is_fraud': is_fraud,
            'fraud_type': fraud_type
        })

    # Convert to DataFrame
    df = pd.DataFrame(transactions)
    
    # Save to data/raw
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '../../'))
    raw_data_dir = os.path.join(project_root, 'data', 'raw')
    os.makedirs(raw_data_dir, exist_ok=True)
    
    output_path = os.path.join(raw_data_dir, 'vietnam_transactions_base.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Successfully generated {len(df)} base transactions.")
    print(f"Total Fraud Cases Injected: {df['is_fraud'].sum()}")
    print(f"File saved to: {output_path}")

if __name__ == "__main__":
    simulate_vietnam_transactions()
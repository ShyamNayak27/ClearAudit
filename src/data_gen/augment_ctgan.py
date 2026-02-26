import pandas as pd
import numpy as np
import os
import uuid
import hashlib
import warnings
from datetime import datetime, timedelta
from faker import Faker
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

# Suppress verbose warnings from SDV/PyTorch
warnings.filterwarnings('ignore')

fake = Faker('vi_VN')
np.random.seed(42)

# ----------------- CONFIGURATION -----------------
TARGET_ROWS = 500000
EPOCHS = 15  # Kept at 15 for local execution time (Production would be 300)
BATCH_SIZE = 500

# Fraud injection targets (proportion of TOTAL synthetic rows)
FRAUD_INJECTION = {
    'fake_shipper': 0.05,      # 5% — P2P, 30k-50k VND, late night
    'quishing': 0.03,          # 3% — QR_PAYMENT, lure amounts, nighttime
    'biometric_evasion': 0.04, # 4% — 9M-9.99M VND, biometric=False
}

BANKS = ['VCB', 'TCB', 'BIDV', 'VPB', 'ACB', 'MBB', 'TPB', 'STB', 'MOMO', 'ZALOPAY', 'VNPAY']
QUISHING_AMOUNTS = [199000, 299000, 499000, 999000, 1999000]


def generate_synthetic_ids(num_rows):
    """Generate synthetic CIFs and MACs for the newly created GAN rows."""
    print(f"Generating IDs for {num_rows} new transactions...")

    # Generate a pool of 20,000 new users to simulate expanding network
    users = []
    for _ in range(20000):
        cif = f"CIF{fake.random_number(digits=8, fix_len=True)}"
        mac = hashlib.sha256(fake.mac_address().encode()).hexdigest()
        users.append({'cif': cif, 'mac': mac})

    senders = np.random.choice(users, size=num_rows)
    receivers = np.random.choice(users, size=num_rows)

    # Extract correctly using list comprehensions
    tx_ids = [str(uuid.uuid4()) for _ in range(num_rows)]
    sender_cifs = [u['cif'] for u in senders]
    sender_macs = [u['mac'] for u in senders]
    receiver_cifs = [u['cif'] for u in receivers]

    # Ensure no self-transfers
    for i in range(num_rows):
        if sender_cifs[i] == receiver_cifs[i]:
            receiver_cifs[i] = f"CIF{fake.random_number(digits=8, fix_len=True)}"

    return tx_ids, sender_cifs, receiver_cifs, sender_macs


def inject_targeted_fraud(df, num_rows):
    """
    Post-generation step: overwrite a portion of synthetic rows with
    specific fraud typologies to ensure heavy representation.
    """
    print("Injecting targeted fraud typologies into synthetic data...")

    indices = df.index.tolist()
    np.random.shuffle(indices)
    pointer = 0

    # --- 1. Fake Shipper Injection ---
    n_fake_shipper = int(num_rows * FRAUD_INJECTION['fake_shipper'])
    fs_indices = indices[pointer:pointer + n_fake_shipper]
    pointer += n_fake_shipper

    for idx in fs_indices:
        hour = np.random.randint(0, 6)  # Late night 00:00 - 05:59
        base_date = datetime(2026, 2, np.random.randint(1, 29))
        ts = base_date.replace(hour=hour, minute=np.random.randint(0, 60),
                               second=np.random.randint(0, 60))
        df.at[idx, 'timestamp'] = ts.strftime('%Y-%m-%d %H:%M:%S')
        df.at[idx, 'amount_vnd'] = round(float(np.random.uniform(30000, 50000)), 0)
        df.at[idx, 'transaction_type'] = 'P2P_TRANSFER'
        df.at[idx, 'receiver_bank_code'] = np.random.choice(BANKS)
        df.at[idx, 'is_biometric_verified'] = False
        df.at[idx, 'is_fraud'] = 1
        df.at[idx, 'fraud_type'] = 'fake_shipper'

    print(f"  Injected {n_fake_shipper} Fake Shipper fraud records")

    # --- 2. Quishing Injection ---
    n_quishing = int(num_rows * FRAUD_INJECTION['quishing'])
    qs_indices = indices[pointer:pointer + n_quishing]
    pointer += n_quishing

    for idx in qs_indices:
        hour = np.random.randint(0, 7)  # Late night 00:00 - 06:59
        base_date = datetime(2026, 2, np.random.randint(1, 29))
        ts = base_date.replace(hour=hour, minute=np.random.randint(0, 60),
                               second=np.random.randint(0, 60))
        df.at[idx, 'timestamp'] = ts.strftime('%Y-%m-%d %H:%M:%S')
        df.at[idx, 'amount_vnd'] = float(np.random.choice(QUISHING_AMOUNTS))
        df.at[idx, 'transaction_type'] = 'QR_PAYMENT'
        df.at[idx, 'receiver_bank_code'] = np.random.choice(['MOMO', 'ZALOPAY', 'VNPAY'])
        df.at[idx, 'is_biometric_verified'] = False
        df.at[idx, 'is_fraud'] = 1
        df.at[idx, 'fraud_type'] = 'quishing'

    print(f"  Injected {n_quishing} Quishing fraud records")

    # --- 3. Biometric Evasion Injection ---
    n_bio = int(num_rows * FRAUD_INJECTION['biometric_evasion'])
    bio_indices = indices[pointer:pointer + n_bio]
    pointer += n_bio

    for idx in bio_indices:
        hour = np.random.randint(0, 24)
        base_date = datetime(2026, 2, np.random.randint(1, 29))
        ts = base_date.replace(hour=hour, minute=np.random.randint(0, 60),
                               second=np.random.randint(0, 60))
        df.at[idx, 'timestamp'] = ts.strftime('%Y-%m-%d %H:%M:%S')
        df.at[idx, 'amount_vnd'] = round(float(np.random.uniform(9000000, 9999999)), 0)
        df.at[idx, 'transaction_type'] = np.random.choice(['P2P_TRANSFER', 'E_COMMERCE'])
        df.at[idx, 'receiver_bank_code'] = np.random.choice(BANKS)
        df.at[idx, 'is_biometric_verified'] = False  # Evasion = bypassing biometric
        df.at[idx, 'is_fraud'] = 1
        df.at[idx, 'fraud_type'] = 'biometric_evasion'

    print(f"  Injected {n_bio} Biometric Evasion fraud records")

    # Mark remaining rows as non-fraud (if CTGAN didn't already)
    remaining = indices[pointer:]
    df.loc[remaining, 'fraud_type'] = df.loc[remaining, 'fraud_type'].fillna('none')

    return df


def augment_data():
    # 1. Paths
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '../../'))
    base_data_path = os.path.join(project_root, 'data', 'raw', 'vietnam_transactions_base.csv')
    synthetic_dir = os.path.join(project_root, 'data', 'synthetic')
    os.makedirs(synthetic_dir, exist_ok=True)

    # 2. Load Base Data
    print("Loading base transaction data...")
    df_base = pd.read_csv(base_data_path)
    current_rows = len(df_base)
    rows_to_generate = TARGET_ROWS - current_rows

    # Add fraud_type column to base data if missing
    if 'fraud_type' not in df_base.columns:
        df_base['fraud_type'] = 'none'
        df_base.loc[df_base['is_fraud'] == 1, 'fraud_type'] = 'legacy_fraud'

    print(f"Base rows: {current_rows}. Need to generate: {rows_to_generate} rows via CTGAN.")

    # 3. Prepare Data for GAN (Drop IDs — GANs can't model unique identifiers)
    train_columns = ['timestamp', 'amount_vnd', 'receiver_bank_code',
                     'transaction_type', 'is_biometric_verified', 'is_fraud']
    train_data = df_base[train_columns].copy()

    # SDV requires datetime objects, not strings
    train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])

    # 4. SDV Metadata Construction
    print("Constructing SingleTableMetadata...")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_data)

    # Enforce strict data types to prevent float generation for categorical classes
    metadata.update_column(column_name='is_fraud', sdtype='categorical')
    metadata.update_column(column_name='is_biometric_verified', sdtype='boolean')

    # 5. Initialize and Train CTGAN
    print(f"Initializing CTGAN Model (Epochs: {EPOCHS})...")
    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=EPOCHS,
        verbose=True
    )

    print("Training CTGAN on Vietnamese financial behaviors... (This may take 2-5 minutes)")
    synthesizer.fit(train_data)

    # 6. Generate New Data (Batched to save memory)
    print(f"Training complete. Generating {rows_to_generate} synthetic rows...")

    chunk_size = 100000
    synthetic_chunks = []
    generated_count = 0

    while generated_count < rows_to_generate:
        to_gen = min(chunk_size, rows_to_generate - generated_count)
        print(f"  Generating chunk of {to_gen} rows...")
        chunk = synthesizer.sample(num_rows=to_gen)
        synthetic_chunks.append(chunk)
        generated_count += to_gen

    df_synthetic = pd.concat(synthetic_chunks, ignore_index=True)

    # 7. Reconstruct Schema (Add back IDs)
    print("Reconstructing schema identifiers...")
    tx_ids, senders_list, receivers_list, macs_list = generate_synthetic_ids(rows_to_generate)

    df_synthetic['transaction_id'] = tx_ids
    df_synthetic['sender_cif'] = senders_list
    df_synthetic['receiver_cif'] = receivers_list
    df_synthetic['device_mac_hash'] = macs_list

    # Initialize fraud_type for synthetic rows
    df_synthetic['fraud_type'] = 'none'

    # 8. *** TARGETED FRAUD INJECTION ***
    df_synthetic = inject_targeted_fraud(df_synthetic, rows_to_generate)

    # Reorder columns to match original schema
    ordered_columns = ['transaction_id', 'timestamp', 'amount_vnd', 'sender_cif',
                       'receiver_cif', 'receiver_bank_code', 'transaction_type',
                       'is_biometric_verified', 'device_mac_hash', 'is_fraud', 'fraud_type']
    df_synthetic = df_synthetic[ordered_columns]
    df_base = df_base[ordered_columns]

    # 9. Combine and Save
    print("Merging base and synthetic datasets...")
    df_final = pd.concat([df_base, df_synthetic], ignore_index=True)

    output_path = os.path.join(synthetic_dir, 'vietnam_transactions_500k.csv')
    df_final.to_csv(output_path, index=False)

    # 10. Report
    print("\n" + "=" * 60)
    print(f"SUCCESS! Dataset saved to: {output_path}")
    print(f"Total Rows: {len(df_final)}")
    print(f"Total Fraud Cases: {df_final['is_fraud'].sum()}")
    print(f"\nFraud Type Distribution:")
    print(df_final['fraud_type'].value_counts().to_string())
    print(f"\nFraud Rate: {df_final['is_fraud'].mean():.2%}")
    print("=" * 60)


if __name__ == "__main__":
    augment_data()
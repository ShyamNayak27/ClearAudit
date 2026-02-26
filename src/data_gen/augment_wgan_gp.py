"""
WGAN-GP (Wasserstein GAN with Gradient Penalty) for Fraud-Focused Oversampling.

This script trains a WGAN-GP exclusively on FRAUD records from the base dataset
to learn the underlying distribution of fraudulent transactions. It then generates
high-quality synthetic fraud records with explicit typology labels.

Output: data/synthetic/vietnam_fraud_wgan_gp.csv
"""
import pandas as pd
import numpy as np
import os
import uuid
import hashlib
import warnings
from datetime import datetime
from faker import Faker

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')
fake = Faker('vi_VN')
np.random.seed(42)
torch.manual_seed(42)

# ----------------- CONFIGURATION -----------------
LATENT_DIM = 128         # Noise vector dimension
HIDDEN_DIM = 256         # Hidden layer width
NUM_EPOCHS = 300         # WGAN-GP training epochs
BATCH_SIZE = 256
CRITIC_ITERS = 5         # Critic updates per generator update
LAMBDA_GP = 10           # Gradient penalty coefficient
LR = 1e-4
NUM_FRAUD_TO_GENERATE = 50000  # Synthetic fraud records to produce

# Fraud typology injection ratios (within generated fraud)
FRAUD_TYPE_RATIOS = {
    'fake_shipper': 0.40,
    'quishing': 0.25,
    'biometric_evasion': 0.35,
}

BANKS = ['VCB', 'TCB', 'BIDV', 'VPB', 'ACB', 'MBB', 'TPB', 'STB', 'MOMO', 'ZALOPAY', 'VNPAY']
QUISHING_AMOUNTS = [199000, 299000, 499000, 999000, 1999000]


# ================== WGAN-GP Architecture ==================

class Generator(nn.Module):
    """Generator network: maps latent noise -> synthetic tabular row."""
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2),
            nn.BatchNorm1d(HIDDEN_DIM * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class Critic(nn.Module):
    """Critic (Discriminator) network: scores realness of tabular rows."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


def compute_gradient_penalty(critic, real_data, fake_data, device):
    """Compute gradient penalty for WGAN-GP."""
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real_data)

    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)

    critic_interp = critic(interpolated)

    gradients = torch.autograd.grad(
        outputs=critic_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interp),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ================== Data Preparation ==================

def prepare_fraud_data(df_base):
    """Extract fraud-only records and encode for neural network input."""
    # Filter to fraud-only rows
    df_fraud = df_base[df_base['is_fraud'] == 1].copy()
    print(f"Found {len(df_fraud)} fraud records in base data for WGAN-GP training.")

    if len(df_fraud) < 50:
        raise ValueError(
            f"Only {len(df_fraud)} fraud records found. Need at least 50 to train WGAN-GP. "
            "Run simulate_paysim_vn.py first to generate base data with fraud."
        )

    # Select numeric/encodable features for GAN training
    feature_cols = ['amount_vnd', 'receiver_bank_code', 'transaction_type',
                    'is_biometric_verified']
    # Extract hour from timestamp for temporal pattern learning
    df_fraud['hour'] = pd.to_datetime(df_fraud['timestamp']).dt.hour
    feature_cols.append('hour')

    # Encode categoricals
    encoders = {}
    for col in ['receiver_bank_code', 'transaction_type']:
        le = LabelEncoder()
        df_fraud[col] = le.fit_transform(df_fraud[col])
        encoders[col] = le

    df_fraud['is_biometric_verified'] = df_fraud['is_biometric_verified'].astype(int)

    # Scale continuous features
    scaler = StandardScaler()
    data_array = df_fraud[feature_cols].values.astype(np.float32)
    data_scaled = scaler.fit_transform(data_array)

    return data_scaled, feature_cols, encoders, scaler


# ================== Training Loop ==================

def train_wgan_gp(real_data, device):
    """Train WGAN-GP on fraud data."""
    data_dim = real_data.shape[1]
    dataset = TensorDataset(torch.FloatTensor(real_data))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    generator = Generator(LATENT_DIM, data_dim).to(device)
    critic = Critic(data_dim).to(device)

    opt_g = optim.Adam(generator.parameters(), lr=LR, betas=(0.0, 0.9))
    opt_c = optim.Adam(critic.parameters(), lr=LR, betas=(0.0, 0.9))

    print(f"Training WGAN-GP for {NUM_EPOCHS} epochs on {len(real_data)} fraud samples...")
    print(f"  Architecture: Generator({LATENT_DIM} -> {data_dim}), Critic({data_dim} -> 1)")

    for epoch in range(NUM_EPOCHS):
        g_loss_epoch = 0
        c_loss_epoch = 0
        n_batches = 0

        for (real_batch,) in dataloader:
            real_batch = real_batch.to(device)
            batch_size = real_batch.size(0)

            # --- Train Critic ---
            for _ in range(CRITIC_ITERS):
                z = torch.randn(batch_size, LATENT_DIM, device=device)
                fake_batch = generator(z).detach()

                c_real = critic(real_batch).mean()
                c_fake = critic(fake_batch).mean()
                gp = compute_gradient_penalty(critic, real_batch, fake_batch, device)

                c_loss = c_fake - c_real + LAMBDA_GP * gp

                opt_c.zero_grad()
                c_loss.backward()
                opt_c.step()

            # --- Train Generator ---
            z = torch.randn(batch_size, LATENT_DIM, device=device)
            fake_batch = generator(z)
            g_loss = -critic(fake_batch).mean()

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            g_loss_epoch += g_loss.item()
            c_loss_epoch += c_loss.item()
            n_batches += 1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            avg_g = g_loss_epoch / max(n_batches, 1)
            avg_c = c_loss_epoch / max(n_batches, 1)
            print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | G_loss: {avg_g:.4f} | C_loss: {avg_c:.4f}")

    print("WGAN-GP training complete.")
    return generator


# ================== Generation & Post-Processing ==================

def generate_fraud_records(generator, scaler, encoders, feature_cols, device):
    """Generate synthetic fraud records and decode them back to original format."""
    print(f"Generating {NUM_FRAUD_TO_GENERATE} synthetic fraud records via WGAN-GP...")

    generator.eval()
    with torch.no_grad():
        z = torch.randn(NUM_FRAUD_TO_GENERATE, LATENT_DIM, device=device)
        synthetic_scaled = generator(z).cpu().numpy()

    # Inverse-scale
    synthetic_raw = scaler.inverse_transform(synthetic_scaled)
    df_syn = pd.DataFrame(synthetic_raw, columns=feature_cols)

    # Round and clip encoded categorical columns
    for col in ['receiver_bank_code', 'transaction_type']:
        le = encoders[col]
        n_classes = len(le.classes_)
        df_syn[col] = df_syn[col].round(0).astype(int).clip(0, n_classes - 1)
        df_syn[col] = le.inverse_transform(df_syn[col])

    # Post-process numeric columns
    df_syn['amount_vnd'] = df_syn['amount_vnd'].abs().round(0).astype(float)
    df_syn['is_biometric_verified'] = df_syn['is_biometric_verified'].round(0).astype(int).clip(0, 1).astype(bool)
    df_syn['hour'] = df_syn['hour'].round(0).astype(int).clip(0, 23)

    # Generate timestamps from hours
    timestamps = []
    for _, row in df_syn.iterrows():
        day = np.random.randint(1, 29)
        ts = datetime(2026, 2, day, int(row['hour']),
                      np.random.randint(0, 60), np.random.randint(0, 60))
        timestamps.append(ts.strftime('%Y-%m-%d %H:%M:%S'))
    df_syn['timestamp'] = timestamps
    df_syn.drop(columns=['hour'], inplace=True)

    # All records are fraud
    df_syn['is_fraud'] = 1

    # Assign fraud typology labels based on configured ratios
    n = len(df_syn)
    fraud_types = []
    for ftype, ratio in FRAUD_TYPE_RATIOS.items():
        fraud_types.extend([ftype] * int(n * ratio))
    # Fill remainder
    while len(fraud_types) < n:
        fraud_types.append(np.random.choice(list(FRAUD_TYPE_RATIOS.keys())))
    np.random.shuffle(fraud_types)
    df_syn['fraud_type'] = fraud_types[:n]

    # Apply typology-specific overrides to ensure realistic patterns
    _apply_typology_overrides(df_syn)

    # Generate IDs
    print("Generating identifiers for WGAN-GP records...")
    users = []
    for _ in range(5000):
        cif = f"CIF{fake.random_number(digits=8, fix_len=True)}"
        mac = hashlib.sha256(fake.mac_address().encode()).hexdigest()
        users.append({'cif': cif, 'mac': mac})

    df_syn['transaction_id'] = [str(uuid.uuid4()) for _ in range(n)]
    senders = np.random.choice(users, size=n)
    receivers = np.random.choice(users, size=n)
    df_syn['sender_cif'] = [u['cif'] for u in senders]
    df_syn['receiver_cif'] = [u['cif'] for u in receivers]
    df_syn['device_mac_hash'] = [u['mac'] for u in senders]

    # Reorder columns
    ordered = ['transaction_id', 'timestamp', 'amount_vnd', 'sender_cif',
               'receiver_cif', 'receiver_bank_code', 'transaction_type',
               'is_biometric_verified', 'device_mac_hash', 'is_fraud', 'fraud_type']
    df_syn = df_syn[ordered]

    return df_syn


def _apply_typology_overrides(df):
    """Enforce realistic patterns for each fraud typology."""
    # Fake Shipper: P2P, 30k-50k VND, late night
    mask_fs = df['fraud_type'] == 'fake_shipper'
    df.loc[mask_fs, 'transaction_type'] = 'P2P_TRANSFER'
    df.loc[mask_fs, 'amount_vnd'] = df.loc[mask_fs].apply(
        lambda _: round(float(np.random.uniform(30000, 50000)), 0), axis=1)
    df.loc[mask_fs, 'is_biometric_verified'] = False

    # Quishing: QR_PAYMENT, specific lure amounts
    mask_qs = df['fraud_type'] == 'quishing'
    df.loc[mask_qs, 'transaction_type'] = 'QR_PAYMENT'
    df.loc[mask_qs, 'amount_vnd'] = df.loc[mask_qs].apply(
        lambda _: float(np.random.choice(QUISHING_AMOUNTS)), axis=1)
    df.loc[mask_qs, 'receiver_bank_code'] = df.loc[mask_qs].apply(
        lambda _: np.random.choice(['MOMO', 'ZALOPAY', 'VNPAY']), axis=1)
    df.loc[mask_qs, 'is_biometric_verified'] = False

    # Biometric Evasion: 9M-9.99M VND, biometric bypassed
    mask_bio = df['fraud_type'] == 'biometric_evasion'
    df.loc[mask_bio, 'amount_vnd'] = df.loc[mask_bio].apply(
        lambda _: round(float(np.random.uniform(9000000, 9999999)), 0), axis=1)
    df.loc[mask_bio, 'transaction_type'] = df.loc[mask_bio].apply(
        lambda _: np.random.choice(['P2P_TRANSFER', 'E_COMMERCE']), axis=1)
    df.loc[mask_bio, 'is_biometric_verified'] = False


# ================== Main ==================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load base data
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '../../'))
    base_path = os.path.join(project_root, 'data', 'raw', 'vietnam_transactions_base.csv')

    if not os.path.exists(base_path):
        raise FileNotFoundError(
            f"Base data not found at {base_path}. Run simulate_paysim_vn.py first."
        )

    df_base = pd.read_csv(base_path)

    # 2. Prepare fraud-only training data
    real_data, feature_cols, encoders, scaler = prepare_fraud_data(df_base)

    # 3. Train WGAN-GP
    generator = train_wgan_gp(real_data, device)

    # 4. Generate synthetic fraud
    df_fraud_synthetic = generate_fraud_records(generator, scaler, encoders, feature_cols, device)

    # 5. Save
    synthetic_dir = os.path.join(project_root, 'data', 'synthetic')
    os.makedirs(synthetic_dir, exist_ok=True)
    output_path = os.path.join(synthetic_dir, 'vietnam_fraud_wgan_gp.csv')
    df_fraud_synthetic.to_csv(output_path, index=False)

    # 6. Report
    print("\n" + "=" * 60)
    print(f"WGAN-GP Fraud Generation Complete!")
    print(f"Output: {output_path}")
    print(f"Total synthetic fraud records: {len(df_fraud_synthetic)}")
    print(f"\nFraud Typology Distribution:")
    print(df_fraud_synthetic['fraud_type'].value_counts().to_string())
    print(f"\nAmount Statistics:")
    print(df_fraud_synthetic['amount_vnd'].describe().to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()

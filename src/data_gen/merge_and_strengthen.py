"""
Step A: Merge WGAN-GP fraud data + Apply realistic fraud timestamp distributions.

1. Loads the CTGAN 500k dataset
2. Appends the WGAN-GP 50k fraud records
3. Overwrites fraud timestamps with REALISTIC per-typology distributions:
   - Fake Shipper: 30% late night (ATO), 70% daytime (lunch/afternoon — when
     people expect packages and COD deliveries)
   - Quishing: 80% daytime/evening (shopping hours when people scan QR codes),
     20% late night
   - Biometric Evasion: Uniform 24/7 (smurfing bots run continuously)
4. Injects burst velocity patterns for a subset of fraud senders
5. Regenerates transaction_ids and saves merged 550k dataset
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)


def _pick_fake_shipper_hour():
    """Fake Shipper: 30% late night (ATO attack), 70% daytime (COD delivery scam).
    
    Daytime hours weighted toward lunch (11-13) and afternoon (14-17)
    when package deliveries are most common and victims expect them.
    """
    if np.random.random() < 0.30:
        # 30% — Late night ATO (Account Takeover) variant
        return np.random.choice([0, 1, 2, 3, 4, 5])
    else:
        # 70% — Daytime delivery scam
        # Weighted: lunch (11-13) and afternoon (14-17) most common
        return np.random.choice(
            [9, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 18]
        )


def _pick_quishing_hour():
    """Quishing: 80% daytime/evening (shopping hours), 20% late night.
    
    QR phishing targets victims who are actively shopping — in stores,
    food stalls, or browsing online in the evening.
    """
    if np.random.random() < 0.80:
        # 80% — Active shopping hours (10am–10pm, weighted toward evening)
        return np.random.choice(
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22]
        )
    else:
        # 20% — Late night (automated/opportunistic)
        return np.random.choice([23, 0, 1, 2, 3, 4])


def _pick_biometric_evasion_hour():
    """Biometric Evasion: Uniform 24/7 (smurfing bots run continuously).
    
    Automated scripts break large amounts into sub-threshold (< 10M VND)
    transactions. Bots don't sleep — they run at any hour.
    """
    return np.random.randint(0, 24)


def main():
    print("=" * 60)
    print("STEP A: MERGE WGAN-GP + REALISTIC FRAUD TIMESTAMPS")
    print("=" * 60)

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    # ─── 1. Load both datasets ───
    ctgan_path = os.path.join(root, 'data', 'synthetic', 'vietnam_transactions_500k.csv')
    wgan_path = os.path.join(root, 'data', 'synthetic', 'vietnam_fraud_wgan_gp.csv')

    print("\nLoading datasets...")
    df_ctgan = pd.read_csv(ctgan_path)
    df_wgan = pd.read_csv(wgan_path)

    print(f"  CTGAN:  {len(df_ctgan):,} rows ({df_ctgan['is_fraud'].sum():,} fraud)")
    print(f"  WGAN-GP: {len(df_wgan):,} rows ({df_wgan['is_fraud'].sum():,} fraud)")

    # ─── 2. Merge datasets ───
    print("\nMerging datasets...")
    df = pd.concat([df_ctgan, df_wgan], ignore_index=True)
    print(f"  Merged: {len(df):,} rows ({df['is_fraud'].sum():,} fraud, {df['is_fraud'].mean():.2%})")

    # ─── 3. Regenerate unique transaction IDs ───
    df['transaction_id'] = [f"TXN_{i:08d}" for i in range(len(df))]

    # ─── 4. Get timestamp range from legitimate data ───
    ts = pd.to_datetime(df['timestamp'])
    date_min = ts.min().date()
    date_max = ts.max().date()
    date_range = (date_max - date_min).days
    print(f"  Date range: {date_min} to {date_max} ({date_range} days)")

    # ─── 5. Apply realistic per-typology timestamp distributions ───
    print("\nApplying realistic fraud timestamp distributions...")
    print("  Fake Shipper:      30% night (ATO) / 70% daytime (COD scam)")
    print("  Quishing:          80% daytime-evening (shopping) / 20% night")
    print("  Biometric Evasion: Uniform 24/7 (bots)")

    fraud_mask = df['is_fraud'] == 1
    n_fraud = fraud_mask.sum()
    print(f"  Fraud transactions to fix: {n_fraud:,}")

    fraud_types_series = df.loc[fraud_mask, 'fraud_type']

    new_timestamps = []
    for idx, ftype in fraud_types_series.items():
        # Random date within the dataset range
        random_day = date_min + timedelta(days=np.random.randint(0, max(date_range, 1)))

        if ftype == 'fake_shipper':
            hour = _pick_fake_shipper_hour()
        elif ftype == 'quishing':
            hour = _pick_quishing_hour()
        elif ftype == 'biometric_evasion':
            hour = _pick_biometric_evasion_hour()
        else:
            # Legacy/unknown fraud — mixed distribution
            hour = np.random.choice([0, 1, 2, 3, 10, 11, 14, 15, 20, 21, 22, 23])

        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)

        ts_new = datetime(random_day.year, random_day.month, random_day.day,
                          hour, minute, second)
        new_timestamps.append(ts_new.strftime('%Y-%m-%d %H:%M:%S'))

    df.loc[fraud_mask, 'timestamp'] = new_timestamps

    # ─── 6. Create burst patterns for fraud (rapid transactions from same sender) ───
    print("  Injecting burst velocity patterns for fraud senders...")

    # Group fraud by sender and make their timestamps close together (within minutes)
    fraud_senders = df.loc[fraud_mask, 'sender_cif'].unique()

    # For ~20% of fraud senders, create burst patterns (multiple tx within 5 minutes)
    burst_senders = np.random.choice(fraud_senders,
                                      size=min(len(fraud_senders) // 5, 2000),
                                      replace=False)

    for sender in burst_senders:
        sender_fraud_idx = df[(df['sender_cif'] == sender) & (df['is_fraud'] == 1)].index
        if len(sender_fraud_idx) <= 1:
            continue

        # Set a base timestamp, then make subsequent ones 10-120 seconds apart
        base_ts = pd.to_datetime(df.loc[sender_fraud_idx[0], 'timestamp'])
        for i, idx in enumerate(sender_fraud_idx[1:], 1):
            offset = timedelta(seconds=np.random.randint(10, 120) * i)
            new_ts = base_ts + offset
            df.loc[idx, 'timestamp'] = new_ts.strftime('%Y-%m-%d %H:%M:%S')

    # ─── 7. Sort by timestamp ───
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['transaction_id'] = [f"TXN_{i:08d}" for i in range(len(df))]
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # ─── 8. Verify distributions ───
    print("\nVerifying fraud signal quality...")
    ts_check = pd.to_datetime(df['timestamp'])
    fraud_hours = ts_check[df['is_fraud'] == 1].dt.hour
    legit_hours = ts_check[df['is_fraud'] == 0].dt.hour

    print(f"  Fraud hour distribution (top 5):")
    for hour, count in fraud_hours.value_counts().head(5).items():
        print(f"    Hour {hour:2d}: {count:,} ({count/len(fraud_hours):.1%})")

    print(f"  Legit hour distribution (top 5):")
    for hour, count in legit_hours.value_counts().head(5).items():
        print(f"    Hour {hour:2d}: {count:,} ({count/len(legit_hours):.1%})")

    # Night ratio comparison
    fraud_night = ((fraud_hours >= 0) & (fraud_hours <= 6)).mean()
    legit_night = ((legit_hours >= 0) & (legit_hours <= 6)).mean()
    print(f"\n  Night (00-06) ratio — Fraud: {fraud_night:.1%}  Legit: {legit_night:.1%}  Separation: {fraud_night/max(legit_night,0.001):.1f}x")

    # ─── 9. Save ───
    out_path = os.path.join(root, 'data', 'synthetic', 'vietnam_transactions_550k.csv')
    print(f"\nSaving merged dataset to: {out_path}")
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"  Total rows:    {len(df):,}")
    print(f"  Fraud rows:    {df['is_fraud'].sum():,} ({df['is_fraud'].mean():.2%})")
    print(f"  Fraud types:   {df[df['is_fraud']==1]['fraud_type'].value_counts().to_dict()}")
    print(f"  Output file:   {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

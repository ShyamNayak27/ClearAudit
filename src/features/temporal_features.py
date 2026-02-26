"""
Temporal Features — Pandas-based time and seasonality feature extraction.

Computes hour-of-day, day-of-week, weekend flags, Tết proximity,
time-since-last-transaction, and cyclical hour encodings.
"""
import pandas as pd
import numpy as np
from datetime import datetime


# Tết 2026 dates
TET_START = datetime(2026, 2, 14)
TET_END = datetime(2026, 2, 22)


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute temporal and seasonality features from the timestamp column.
    
    Args:
        df: DataFrame with at least 'transaction_id', 'timestamp', 'sender_cif'.
            Must be sorted by timestamp.
    
    Returns:
        DataFrame with transaction_id + all temporal feature columns.
    """
    print("  [Temporal] Computing time-based features...")

    result = pd.DataFrame()
    result['transaction_id'] = df['transaction_id']

    ts = pd.to_datetime(df['timestamp'])

    # --- Basic temporal ---
    result['hour'] = ts.dt.hour
    result['day_of_week'] = ts.dt.dayofweek            # 0=Monday, 6=Sunday
    result['is_weekend'] = (ts.dt.dayofweek >= 5).astype(int)
    result['is_night'] = ((ts.dt.hour >= 0) & (ts.dt.hour <= 6)).astype(int)

    # --- Tết seasonality ---
    result['is_tet_period'] = (
        (ts >= TET_START) & (ts <= TET_END)
    ).astype(int)

    # Days to Tết start (negative = after Tết started, positive = before)
    result['days_to_tet'] = (TET_START - ts).dt.total_seconds() / 86400.0
    result['days_to_tet'] = result['days_to_tet'].round(2)

    # --- Time since last transaction (per sender) ---
    df_temp = df[['sender_cif']].copy()
    df_temp['ts'] = ts
    df_temp['time_since_last_tx'] = (
        df_temp.groupby('sender_cif')['ts']
        .diff()
        .dt.total_seconds()
        .fillna(0)
    )
    result['time_since_last_tx'] = df_temp['time_since_last_tx'].values

    # --- Cyclical hour encoding (sin/cos for continuity at midnight) ---
    result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24).round(6)
    result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24).round(6)

    feature_count = len(result.columns) - 1  # exclude transaction_id
    print(f"  [Temporal] Computed {feature_count} temporal features.")
    return result

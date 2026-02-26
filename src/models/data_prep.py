"""
Step 2.1 — Data Preparation & Balancing.

Loads the feature-engineered dataset, encodes categoricals, performs a stratified
train/test split, and balances the training set using SMOTE + Gaussian noise
injection for robust fraud detection model training.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib


# ─── Configuration ───
TEST_SIZE = 0.20
RANDOM_STATE = 42
SMOTE_SAMPLING_STRATEGY = 0.5  # Bring minority to 50% of majority class
GAUSSIAN_NOISE_STD = 0.01      # Std dev of noise added to SMOTE samples

# Columns to DROP (identifiers, targets, metadata)
DROP_COLUMNS = [
    'transaction_id',
    'timestamp',
    'sender_cif',
    'receiver_cif',
    'device_mac_hash',
    'fraud_type',  # Kept separately for per-typology evaluation
]

# Target column
TARGET = 'is_fraud'

# Columns to label-encode
CATEGORICAL_COLUMNS = ['receiver_bank_code', 'transaction_type']


def load_feature_data():
    """Load the feature-engineered dataset."""
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '../../'))
    input_path = os.path.join(project_root, 'data', 'processed', 'features_550k.csv')

    print(f"Loading features from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Shape: {df.shape}")
    print(f"  Fraud distribution: {df[TARGET].value_counts().to_dict()}")
    return df, project_root


def encode_categoricals(df, encoders=None, fit=True):
    """
    Label-encode categorical columns.
    
    Args:
        df: DataFrame to encode.
        encoders: Dict of pre-fitted LabelEncoders (for transform-only on test set).
        fit: If True, fit new encoders. If False, use provided encoders.
    
    Returns:
        df: Encoded DataFrame.
        encoders: Dict of fitted LabelEncoders.
    """
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen categories gracefully
            df[col] = df[col].astype(str).map(
                lambda x, _le=le: _le.transform([x])[0] if x in _le.classes_
                else -1
            )

    return df, encoders


def prepare_data():
    """Main data preparation pipeline."""
    print("=" * 60)
    print("STEP 2.1: DATA PREPARATION & BALANCING")
    print("=" * 60)

    # ─── 1. Load ───
    df, project_root = load_feature_data()

    # ─── 2. Preserve fraud_type for evaluation ───
    fraud_types = df['fraud_type'].copy()

    # ─── 3. Separate features and target ───
    y = df[TARGET].copy()
    X = df.drop(columns=DROP_COLUMNS + [TARGET], errors='ignore')

    print(f"\n  Features before encoding: {list(X.columns)}")
    print(f"  Feature count: {len(X.columns)}")

    # ─── 4. Encode categoricals ───
    print("\nEncoding categorical columns...")
    X, encoders = encode_categoricals(X, fit=True)

    # Convert boolean to int
    if 'is_biometric_verified' in X.columns:
        X['is_biometric_verified'] = X['is_biometric_verified'].astype(int)

    print(f"  Encoded columns: {CATEGORICAL_COLUMNS}")
    print(f"  Final feature dtypes:\n{X.dtypes.value_counts().to_string()}")

    # ─── 5. Train/Test split (stratified) ───
    print(f"\nStratified train/test split ({1-TEST_SIZE:.0%} / {TEST_SIZE:.0%})...")
    X_train, X_test, y_train, y_test, ft_train, ft_test = train_test_split(
        X, y, fraud_types,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"  Train: {X_train.shape[0]:,} rows ({y_train.mean():.2%} fraud)")
    print(f"  Test:  {X_test.shape[0]:,} rows ({y_test.mean():.2%} fraud)")

    # ─── 6. SMOTE on training set only ───
    print(f"\nApplying SMOTE (sampling_strategy={SMOTE_SAMPLING_STRATEGY})...")

    pre_smote_fraud = y_train.sum()
    pre_smote_legit = (y_train == 0).sum()

    smote = SMOTE(
        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        random_state=RANDOM_STATE,
        k_neighbors=5,
    )
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    post_smote_fraud = y_train_balanced.sum()
    smote_new_samples = post_smote_fraud - pre_smote_fraud

    print(f"  Before SMOTE: {pre_smote_legit:,} legit / {pre_smote_fraud:,} fraud")
    print(f"  After SMOTE:  {(y_train_balanced == 0).sum():,} legit / {post_smote_fraud:,} fraud")
    print(f"  New synthetic fraud samples: {smote_new_samples:,}")

    # ─── 7. Gaussian noise injection on SMOTE-generated samples ───
    print(f"\nInjecting Gaussian noise (std={GAUSSIAN_NOISE_STD}) into SMOTE samples...")

    # The SMOTE samples are appended at the end of the array
    n_original = len(X_train)
    smote_indices = range(n_original, len(X_train_balanced))

    # Only add noise to numeric columns (skip encoded categoricals which must stay integer)
    numeric_mask = np.ones(X_train_balanced.shape[1], dtype=bool)
    for col in CATEGORICAL_COLUMNS:
        if col in X_train_balanced.columns:
            col_idx = X_train_balanced.columns.get_loc(col)
            numeric_mask[col_idx] = False

    # Also skip binary columns
    for col in ['is_biometric_verified', 'is_weekend', 'is_night', 'is_tet_period', 'is_repeat_pair']:
        if col in X_train_balanced.columns:
            col_idx = X_train_balanced.columns.get_loc(col)
            numeric_mask[col_idx] = False

    noise = np.zeros_like(X_train_balanced.iloc[list(smote_indices)].values)
    noise[:, numeric_mask] = np.random.normal(
        0, GAUSSIAN_NOISE_STD, size=(len(smote_indices), numeric_mask.sum())
    )
    X_train_balanced.iloc[list(smote_indices)] += noise

    print(f"  Noise applied to {numeric_mask.sum()} continuous feature columns")
    print(f"  Skipped {(~numeric_mask).sum()} categorical/binary columns")

    # ─── 8. Save everything ───
    output_dir = os.path.join(project_root, 'data', 'processed')
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print(f"\nSaving prepared data to: {output_dir}")

    X_train_balanced.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    pd.Series(y_train_balanced, name=TARGET).to_csv(
        os.path.join(output_dir, 'y_train.csv'), index=False)
    pd.Series(y_test, name=TARGET).to_csv(
        os.path.join(output_dir, 'y_test.csv'), index=False)

    # Save fraud_type for test set (for per-typology evaluation in Step 2.4)
    ft_test.to_csv(os.path.join(output_dir, 'test_fraud_types.csv'), index=False)

    # Save label encoders for inference pipeline
    joblib.dump(encoders, os.path.join(models_dir, 'label_encoders.joblib'))

    # Save feature column names (for consistent ordering during inference)
    feature_names = list(X_train_balanced.columns)
    with open(os.path.join(models_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f, indent=2)

    # ─── 9. Summary report ───
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Feature count:       {len(feature_names)}")
    print(f"Train rows (balanced): {len(X_train_balanced):,}")
    print(f"Test rows (original):  {len(X_test):,}")
    print(f"Train fraud ratio:   {y_train_balanced.mean():.2%}")
    print(f"Test fraud ratio:    {y_test.mean():.2%} (untouched)")
    print(f"\nFiles saved:")
    print(f"  {output_dir}/X_train.csv")
    print(f"  {output_dir}/X_test.csv")
    print(f"  {output_dir}/y_train.csv")
    print(f"  {output_dir}/y_test.csv")
    print(f"  {output_dir}/test_fraud_types.csv")
    print(f"  {models_dir}/label_encoders.joblib")
    print(f"  {models_dir}/feature_names.json")
    print("=" * 60)


if __name__ == "__main__":
    prepare_data()

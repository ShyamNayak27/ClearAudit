"""
Step 2.3 — Unsupervised Anomaly Detection: TensorFlow Autoencoder.

Trains an autoencoder ONLY on legitimate transactions so it learns what
"normal" looks like. At inference, high reconstruction error → anomaly → fraud.
This catches zero-day attacks that supervised models have never seen.

Outputs:
    models/autoencoder.keras
    models/ae_scaler.joblib
    models/ae_threshold.json
"""
import os
import json
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, f1_score, precision_recall_curve, auc
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info logs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ─── Configuration ───
ENCODING_DIM = 16         # Bottleneck size
EPOCHS = 50
BATCH_SIZE = 512
VALIDATION_SPLIT = 0.1
THRESHOLD_PERCENTILE = 95  # Reconstruction error percentile for anomaly cutoff
RANDOM_SEED = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def build_autoencoder(input_dim):
    """
    Build encoder-decoder architecture.
    
    Architecture: input(N) → 64 → 32 → 16 (bottleneck) → 32 → 64 → output(N)
    """
    # Encoder
    encoder_input = keras.Input(shape=(input_dim,), name='encoder_input')
    x = layers.Dense(64, activation='relu', name='enc_dense_1')(encoder_input)
    x = layers.BatchNormalization(name='enc_bn_1')(x)
    x = layers.Dropout(0.2, name='enc_dropout_1')(x)
    x = layers.Dense(32, activation='relu', name='enc_dense_2')(x)
    x = layers.BatchNormalization(name='enc_bn_2')(x)
    bottleneck = layers.Dense(ENCODING_DIM, activation='relu', name='bottleneck')(x)

    # Decoder
    x = layers.Dense(32, activation='relu', name='dec_dense_1')(bottleneck)
    x = layers.BatchNormalization(name='dec_bn_1')(x)
    x = layers.Dropout(0.2, name='dec_dropout_1')(x)
    x = layers.Dense(64, activation='relu', name='dec_dense_2')(x)
    x = layers.BatchNormalization(name='dec_bn_2')(x)
    decoder_output = layers.Dense(input_dim, activation='linear', name='decoder_output')(x)

    autoencoder = keras.Model(encoder_input, decoder_output, name='fraud_autoencoder')

    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
    )

    return autoencoder


def compute_reconstruction_error(model, X):
    """Compute per-sample MSE reconstruction error."""
    X_pred = model.predict(X, verbose=0)
    mse = np.mean(np.square(X - X_pred), axis=1)
    return mse


def main():
    print("=" * 60)
    print("STEP 2.3: AUTOENCODER TRAINING (ANOMALY DETECTION)")
    print("=" * 60)

    # ─── 1. Load data ───
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '../../'))
    processed_dir = os.path.join(project_root, 'data', 'processed')
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)

    print("\nLoading data...")
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv'))['is_fraud']
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv'))['is_fraud']

    print(f"  Train total:  {len(X_train):,}")
    print(f"  Test total:   {len(X_test):,}")

    # ─── 2. Filter to legitimate-only for training ───
    X_train_legit = X_train[y_train == 0].copy()
    print(f"  Train legit only: {len(X_train_legit):,} (fraud removed for autoencoder)")

    # ─── 3. Scale features (neural nets need normalization) ───
    print("\nScaling features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_legit)
    X_test_scaled = scaler.transform(X_test)

    input_dim = X_train_scaled.shape[1]
    print(f"  Input dimension: {input_dim}")

    # ─── 4. Build and train autoencoder ───
    print(f"\nBuilding autoencoder: {input_dim} → 64 → 32 → {ENCODING_DIM} → 32 → 64 → {input_dim}")
    model = build_autoencoder(input_dim)
    model.summary()

    print(f"\nTraining for {EPOCHS} epochs (batch_size={BATCH_SIZE})...")
    t0 = time.time()

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    history = model.fit(
        X_train_scaled, X_train_scaled,  # Input = target (reconstruction)
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    elapsed = time.time() - t0
    final_loss = history.history['val_loss'][-1]
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Final val_loss: {final_loss:.6f}")

    # ─── 5. Compute reconstruction errors and set threshold ───
    print(f"\nComputing reconstruction errors...")

    # Errors on training data (legitimate only) to set the threshold
    train_errors = compute_reconstruction_error(model, X_train_scaled)

    # Errors on test data (mixed legit + fraud)
    test_errors = compute_reconstruction_error(model, X_test_scaled)

    # Set threshold at the configured percentile of training errors
    threshold = float(np.percentile(train_errors, THRESHOLD_PERCENTILE))
    print(f"  Train error — mean: {train_errors.mean():.6f}, std: {train_errors.std():.6f}")
    print(f"  Threshold (p{THRESHOLD_PERCENTILE}): {threshold:.6f}")

    # ─── 6. Evaluate on the test set ───
    print(f"\nEvaluating on test set...")
    y_pred_ae = (test_errors > threshold).astype(int)
    f1_ae = f1_score(y_test, y_pred_ae)

    # PR-AUC using reconstruction error as the anomaly score
    prec, rec, _ = precision_recall_curve(y_test, test_errors)
    pr_auc_ae = auc(rec, prec)

    print(f"  PR-AUC: {pr_auc_ae:.4f}")
    print(f"  F1:     {f1_ae:.4f}")
    print(classification_report(y_test, y_pred_ae, target_names=['Legit', 'Fraud']))

    # Error distribution comparison
    legit_errors = test_errors[y_test == 0]
    fraud_errors = test_errors[y_test == 1]
    print(f"  Legit test error  — mean: {legit_errors.mean():.6f}, std: {legit_errors.std():.6f}")
    print(f"  Fraud test error  — mean: {fraud_errors.mean():.6f}, std: {fraud_errors.std():.6f}")
    print(f"  Separation ratio: {fraud_errors.mean() / legit_errors.mean():.2f}x")

    # ─── 7. Save everything ───
    print("\nSaving model artifacts...")

    model_path = os.path.join(models_dir, 'autoencoder.keras')
    model.save(model_path)
    print(f"  Model: {model_path}")

    scaler_path = os.path.join(models_dir, 'ae_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler: {scaler_path}")

    threshold_info = {
        'threshold': threshold,
        'percentile': THRESHOLD_PERCENTILE,
        'train_error_mean': float(train_errors.mean()),
        'train_error_std': float(train_errors.std()),
        'pr_auc': float(pr_auc_ae),
        'f1': float(f1_ae),
        'epochs_trained': len(history.history['loss']),
        'final_val_loss': float(final_loss),
    }
    threshold_path = os.path.join(models_dir, 'ae_threshold.json')
    with open(threshold_path, 'w') as f:
        json.dump(threshold_info, f, indent=2)
    print(f"  Threshold: {threshold_path}")

    # ─── 8. Summary ───
    print("\n" + "=" * 60)
    print("AUTOENCODER TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Architecture:     {input_dim} → 64 → 32 → {ENCODING_DIM} → 32 → 64 → {input_dim}")
    print(f"  Trained on:       {len(X_train_legit):,} legitimate-only samples")
    print(f"  Epochs:           {len(history.history['loss'])} (early stopping)")
    print(f"  Threshold (p{THRESHOLD_PERCENTILE}): {threshold:.6f}")
    print(f"  PR-AUC:           {pr_auc_ae:.4f}")
    print(f"  F1:               {f1_ae:.4f}")
    print(f"  Fraud error is    {fraud_errors.mean() / legit_errors.mean():.2f}x higher than legit")
    print("=" * 60)


if __name__ == "__main__":
    main()

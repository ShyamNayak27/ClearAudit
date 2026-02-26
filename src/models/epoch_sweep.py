"""
Epoch Sweep: Find the optimal number of epochs for the autoencoder.
Trains for 200 epochs and logs F1 + PR-AUC at every epoch.
"""
import os, json, numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_curve, auc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
np.random.seed(42)

ENCODING_DIM = 16
MAX_EPOCHS = 200
BATCH_SIZE = 512
THRESHOLD_PERCENTILE = 95

# Load data
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
pdir = os.path.join(root, 'data', 'processed')
X_train = pd.read_csv(os.path.join(pdir, 'X_train.csv'))
y_train = pd.read_csv(os.path.join(pdir, 'y_train.csv'))['is_fraud']
X_test = pd.read_csv(os.path.join(pdir, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(pdir, 'y_test.csv'))['is_fraud']

X_legit = X_train[y_train == 0].copy()
scaler = StandardScaler()
X_legit_s = scaler.fit_transform(X_legit)
X_test_s = scaler.transform(X_test)

dim = X_legit_s.shape[1]

# Build model
inp = keras.Input(shape=(dim,))
x = layers.Dense(64, activation='relu')(inp)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.BatchNormalization()(x)
bn = layers.Dense(ENCODING_DIM, activation='relu')(x)
x = layers.Dense(32, activation='relu')(bn)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.BatchNormalization()(x)
out = layers.Dense(dim, activation='linear')(x)
model = keras.Model(inp, out)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')

# Custom callback to evaluate F1 every 10 epochs
class F1Logger(keras.callbacks.Callback):
    def __init__(self):
        self.results = []
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Train errors for threshold
            train_pred = self.model.predict(X_legit_s, verbose=0)
            train_err = np.mean(np.square(X_legit_s - train_pred), axis=1)
            thresh = float(np.percentile(train_err, THRESHOLD_PERCENTILE))
            # Test errors
            test_pred = self.model.predict(X_test_s, verbose=0)
            test_err = np.mean(np.square(X_test_s - test_pred), axis=1)
            y_pred = (test_err > thresh).astype(int)
            f1 = f1_score(y_test, y_pred)
            prec, rec, _ = precision_recall_curve(y_test, test_err)
            prauc = auc(rec, prec)
            self.results.append({
                'epoch': epoch + 1, 'f1': round(f1, 4),
                'pr_auc': round(prauc, 4), 'threshold': round(thresh, 6),
                'val_loss': round(logs.get('val_loss', 0), 6)
            })
            print(f"  >>> Epoch {epoch+1:3d} | F1: {f1:.4f} | PR-AUC: {prauc:.4f} | thresh: {thresh:.6f} | val_loss: {logs.get('val_loss',0):.6f}")

logger = F1Logger()

print(f"Training {MAX_EPOCHS} epochs, evaluating every 10...")
model.fit(
    X_legit_s, X_legit_s,
    epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
    validation_split=0.1, callbacks=[logger], verbose=0
)

# Print summary table
print("\n" + "=" * 70)
print(f"{'Epoch':>6} | {'F1':>7} | {'PR-AUC':>7} | {'Val Loss':>10} | {'Threshold':>10}")
print("-" * 70)
best_f1 = 0
best_epoch = 0
for r in logger.results:
    marker = ""
    if r['f1'] > best_f1:
        best_f1 = r['f1']
        best_epoch = r['epoch']
        marker = " <-- BEST"
    print(f"{r['epoch']:>6} | {r['f1']:>7.4f} | {r['pr_auc']:>7.4f} | {r['val_loss']:>10.6f} | {r['threshold']:>10.6f}{marker}")

print("=" * 70)
print(f"\nBEST F1: {best_f1:.4f} at epoch {best_epoch}")

# Save results
with open(os.path.join(root, 'models', 'epoch_sweep.json'), 'w') as f:
    json.dump({'results': logger.results, 'best_epoch': best_epoch, 'best_f1': best_f1}, f, indent=2)

import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import joblib
from utils.landmarks import extract_landmarks

DATA_DIR       = "data/dynamic"
SEQUENCE_LENGTH = 40
MODEL_OUT      = "models/dynamic_model.keras"
ENCODER_OUT    = "models/dynamic_label_encoder.pkl"

# ── Load data ─────────────────────────────────────────────────────
print("📂 Loading dynamic sign data...")
X, y = [], []

signs = sorted(os.listdir(DATA_DIR))
print(f"   Found signs: {signs}\n")

for sign in signs:
    sign_dir = os.path.join(DATA_DIR, sign)
    files    = [f for f in os.listdir(sign_dir) if f.endswith(".npy")]
    
    if len(files) == 0:
        print(f"   ⚠️  Skipping '{sign}' — no data found")
        continue

    loaded = 0
    for f in files:
        path = os.path.join(sign_dir, f)
        try:
            seq = np.load(path, allow_pickle=False)
            if seq.shape == (SEQUENCE_LENGTH, 63):
                X.append(seq)
                y.append(sign)
                loaded += 1
        except Exception as e:
            print(f"   ⚠️  Skipping {f}: {e}")
    
    print(f"   ✅ {sign}: {loaded} sequences loaded")
    sign_dir = os.path.join(DATA_DIR, sign)
    files    = [f for f in os.listdir(sign_dir) if f.endswith(".npy")]
    for f in files:
        seq = np.load(os.path.join(sign_dir, f))
        if seq.shape == (SEQUENCE_LENGTH, 63):
            X.append(seq)
            y.append(sign)

X = np.array(X)   # shape: (total_samples, 30, 63)
y = np.array(y)

print(f"✅ Loaded {len(X)} sequences across {len(signs)} signs")
print(f"   Input shape: {X.shape}\n")

# ── Encode labels ─────────────────────────────────────────────────
le = LabelEncoder()
y_encoded    = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# ── Train / test split ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"📊 Train: {len(X_train)} | Test: {len(X_test)}\n")

# ── Build LSTM model ──────────────────────────────────────────────
print("🧠 Building LSTM model...")
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 63)),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dropout(0.2),

    Dense(len(signs), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ── Callbacks ─────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=15,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=7, verbose=1, min_lr=1e-6)
]

# ── Train ─────────────────────────────────────────────────────────
print("\n🚀 Training...\n")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    callbacks=callbacks
)

# ── Evaluate ──────────────────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🎯 Test Accuracy: {acc*100:.2f}%")

# ── Save ──────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
model.save(MODEL_OUT)
joblib.dump(le, ENCODER_OUT)

print(f"\n💾 Model saved to {MODEL_OUT}")
print(f"💾 Label encoder saved to {ENCODER_OUT}")
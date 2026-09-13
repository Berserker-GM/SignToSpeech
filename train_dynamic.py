"""
Train LSTM on dynamic sign sequences (.npy).

Pipeline:
  load (N, 40, 63) → velocity concat → (N, 40, 126)
  split → augment TRAIN only → class-weighted fit → report + confusion matrix
"""
from __future__ import annotations

import os

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from utils.sequence_features import add_velocity

DATA_DIR = "data/dynamic"
SEQUENCE_LENGTH = 40
POSE_DIM = 63
FEATURE_DIM = 126  # pose (63) + velocity (63)
MODEL_OUT = "models/dynamic_model.keras"
ENCODER_OUT = "models/dynamic_label_encoder.pkl"
AUGMENT_MULTIPLIER = 2  # extra synthetic copies per real train sample (0 = off)


def augment_sequence(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Lightweight aug on a (T, 63) pose sequence (velocity added later).
    - xy jitter (scale / small translate / small rotation in xy)
    - light time warp (drop/dup frames → re-pad to T)
    """
    out = seq.astype(np.float32).copy()
    t, d = out.shape
    assert d == POSE_DIM

    # Reshape to (T, 21, 3)
    xyz = out.reshape(t, 21, 3)

    scale = float(rng.uniform(0.92, 1.08))
    tx = float(rng.uniform(-0.04, 0.04))
    ty = float(rng.uniform(-0.04, 0.04))
    angle = float(rng.uniform(-12.0, 12.0)) * np.pi / 180.0
    c, s = np.cos(angle), np.sin(angle)

    xy = xyz[:, :, :2] * scale
    x = xy[:, :, 0] * c - xy[:, :, 1] * s + tx
    y = xy[:, :, 0] * s + xy[:, :, 1] * c + ty
    xyz[:, :, 0] = x
    xyz[:, :, 1] = y
    # z untouched except tiny noise
    xyz[:, :, 2] += rng.normal(0.0, 0.005, size=xyz[:, :, 2].shape)

    warped = xyz.reshape(t, POSE_DIM)

    # Time warp: randomly drop or duplicate ~10% of frames
    n_ops = max(1, t // 10)
    idx = list(range(t))
    for _ in range(n_ops):
        if rng.random() < 0.5 and len(idx) > t // 2:
            del idx[int(rng.integers(0, len(idx)))]
        else:
            j = int(rng.integers(0, len(idx)))
            idx.insert(j, idx[j])

    warped = warped[np.array(idx)]
    if len(warped) >= t:
        warped = warped[:t]
    else:
        pad = np.repeat(warped[-1:], t - len(warped), axis=0)
        warped = np.concatenate([warped, pad], axis=0)

    return warped.astype(np.float32)


def load_sequences():
    print("📂 Loading dynamic sign data...")
    X, y = [], []

    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f"Missing {DATA_DIR}/ — run collect_dynamic.py first.")

    signs = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    )
    print(f"   Found signs: {signs}\n")

    for sign in signs:
        sign_dir = os.path.join(DATA_DIR, sign)
        files = [f for f in os.listdir(sign_dir) if f.endswith(".npy")]

        if len(files) == 0:
            print(f"   ⚠️  Skipping '{sign}' — no data found")
            continue

        loaded = 0
        for f in files:
            path = os.path.join(sign_dir, f)
            try:
                seq = np.load(path, allow_pickle=False)
                if seq.shape == (SEQUENCE_LENGTH, POSE_DIM):
                    X.append(seq.astype(np.float32))
                    y.append(sign)
                    loaded += 1
            except Exception as e:
                print(f"   ⚠️  Skipping {f}: {e}")

        print(f"   ✅ {sign}: {loaded} sequences loaded")

    if not X:
        raise SystemExit("No valid sequences found.")

    X = np.stack(X, axis=0)  # (N, T, 63)
    y = np.array(y)
    print(f"\n✅ Loaded {len(X)} sequences across {len(set(y))} signs")
    print(f"   Pose shape: {X.shape}")
    return X, y


def main():
    X_pose, y = load_sequences()

    # Velocity features → (N, T, 126)
    print("➕ Adding frame-to-frame velocity features...")
    X = np.stack([add_velocity(seq) for seq in X_pose], axis=0)
    print(f"   Feature shape: {X.shape}  (last dim {FEATURE_DIM} = pose+vel)\n")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    n_classes = len(le.classes_)

    X_train, X_test, y_train, y_test, y_tr_idx, y_te_idx = train_test_split(
        X,
        y_categorical,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    # Augment TRAIN only (operate on pose half, then re-add velocity)
    if AUGMENT_MULTIPLIER > 0:
        print(f"🎛️  Augmenting train set ×{AUGMENT_MULTIPLIER} (no test leakage)...")
        rng = np.random.default_rng(42)
        aug_x, aug_y = [], []
        for seq, label_oh in zip(X_train, y_train):
            pose = seq[:, :POSE_DIM]
            for _ in range(AUGMENT_MULTIPLIER):
                aug_pose = augment_sequence(pose, rng)
                aug_x.append(add_velocity(aug_pose))
                aug_y.append(label_oh)
        if aug_x:
            X_train = np.concatenate([X_train, np.stack(aug_x)], axis=0)
            y_train = np.concatenate([y_train, np.stack(aug_y)], axis=0)
            y_tr_idx = np.argmax(y_train, axis=1)
        print(f"   Train after aug: {len(X_train)} | Test: {len(X_test)}\n")
    else:
        print(f"📊 Train: {len(X_train)} | Test: {len(X_test)}\n")

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_tr_idx), y=y_tr_idx
    )
    class_weight = {int(c): float(w) for c, w in zip(np.unique(y_tr_idx), class_weights)}
    print(f"⚖️  Class weights: { {le.classes_[k]: round(v, 2) for k, v in class_weight.items()} }\n")

    print("🧠 Building LSTM model...")
    model = Sequential(
        [
            LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, FEATURE_DIM)),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, verbose=1, min_lr=1e-6
        ),
    ]

    print("\n🚀 Training...\n")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=16,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n🎯 Test Accuracy: {acc * 100:.2f}%")

    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\n📋 Per-sign classification report:")
    print(
        classification_report(
            y_true, y_pred, target_names=list(le.classes_), zero_division=0
        )
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    print("🧩 Confusion matrix (rows=true, cols=pred):")
    header = "true\\pred".ljust(12) + "".join(f"{c[:7]:>8}" for c in le.classes_)
    print(header)
    for i, name in enumerate(le.classes_):
        row = f"{name[:11]:<12}" + "".join(f"{cm[i, j]:8d}" for j in range(n_classes))
        print(row)

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_OUT)
    joblib.dump(le, ENCODER_OUT)

    print(f"\n💾 Model saved to {MODEL_OUT}")
    print(f"💾 Label encoder saved to {ENCODER_OUT}")
    print(
        f"\n📌 SUMMARY: input_shape=({SEQUENCE_LENGTH}, {FEATURE_DIM}) "
        f"| AUGMENT_MULTIPLIER={AUGMENT_MULTIPLIER} | class_weight=balanced"
    )
    print("   Live inference must append velocity the same way (see api/inference.py).")


if __name__ == "__main__":
    main()

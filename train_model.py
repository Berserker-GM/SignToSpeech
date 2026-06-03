import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from utils.landmarks import extract_landmarks

print("📂 Loading data...")
df = pd.read_csv("data/landmarks.csv")

X = df.drop("label", axis=1).values
y = df["label"].values

# Encode labels to numbers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"✅ Loaded {len(X)} samples across {len(le.classes_)} signs")
print(f"   Signs: {list(le.classes_)}\n")

# ── Train / test split ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"📊 Train: {len(X_train)} samples | Test: {len(X_test)} samples\n")

# ── Train Random Forest ───────────────────────────────────────────
print("🌲 Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1       # use all CPU cores
)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean() * 100

print(f"\n🎯 Accuracy: {accuracy:.2f}%\n")
print("📋 Per-sign breakdown:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Save model + label encoder ────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sign_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")

print("💾 Model saved to models/sign_model.pkl")
print("💾 Label encoder saved to models/label_encoder.pkl")
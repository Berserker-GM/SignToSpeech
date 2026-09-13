"""
Train static RandomForest on data/landmarks.csv.

Expects columns: x0,y0,z0,...,x20,y20,z20, session, label
Falls back if session is missing (warns; uses row-block groups).

If each sign was collected in a single session (common after one SPACE press
per sign), GroupShuffleSplit would put whole signs only in train or only in
test → ~0% accuracy. In that case we fall back to a stratified split.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
)
from sklearn.preprocessing import LabelEncoder

print("Loading data...")
df = pd.read_csv("data/landmarks.csv")

if "label" not in df.columns:
    raise SystemExit("CSV missing 'label' column")

if "session" not in df.columns:
    print("⚠️  No 'session' column — inventing groups of 10 consecutive rows (recollect recommended)")
    df["session"] = np.arange(len(df)) // 10

feature_cols = [c for c in df.columns if c not in ("label", "session")]
X = df[feature_cols].values
y = df["label"].values
groups = df["session"].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)
labels_all = list(range(len(le.classes_)))

print(f"✅ Loaded {len(X)} samples across {len(le.classes_)} signs")
print(f"   Feature cols: {len(feature_cols)} (session excluded from X)")
print(f"   Sessions: {len(np.unique(groups))}")
print(f"   Signs: {list(le.classes_)}\n")

# ── Detect degenerate sessions (1 session ≈ 1 sign) ───────────────
sess_to_labels = (
    df.groupby("session")["label"].nunique().rename("n_labels")
)
labels_per_session = int(sess_to_labels.max()) if len(sess_to_labels) else 0
sessions_per_label = df.groupby("label")["session"].nunique()
min_sessions_per_label = int(sessions_per_label.min()) if len(sessions_per_label) else 0

use_grouped = min_sessions_per_label >= 2 and labels_per_session >= 1
if not use_grouped:
    print(
        "⚠️  Each sign has only ONE session (you pressed SPACE once per sign).\n"
        "   Grouped split would train on some signs and test on others → 0% accuracy.\n"
        "   Using stratified split instead so every sign appears in train AND test.\n"
        "   Tip: for real session validation, collect each sign 2+ times (SPACE → fill → next).\n"
    )

# ── Train / test split ────────────────────────────────────────────
if use_grouped:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y_encoded, groups))
    split_name = "grouped (session)"
else:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y_encoded))
    split_name = "stratified"

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

print(f"📊 Train: {len(X_train)} samples | Test: {len(X_test)} samples  [{split_name}]")
if use_grouped:
    print(
        f"   Train sessions: {len(np.unique(groups[train_idx]))} | "
        f"Test sessions: {len(np.unique(groups[test_idx]))}"
    )
print(
    f"   Train signs: {sorted(le.inverse_transform(np.unique(y_train)))}\n"
    f"   Test signs:  {sorted(le.inverse_transform(np.unique(y_test)))}\n"
)

# ── Cross-validation ──────────────────────────────────────────────
if use_grouped:
    n_unique_groups = len(np.unique(groups))
    n_splits = min(5, n_unique_groups)
    if n_splits >= 2:
        print(f"🔁 Grouped CV ({n_splits}-fold by session)...")
        cv = GroupKFold(n_splits=n_splits)
        cv_scores = cross_val_score(
            RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            ),
            X,
            y_encoded,
            cv=cv,
            groups=groups,
            scoring="accuracy",
            n_jobs=-1,
        )
        print(f"   CV accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
        print(f"   Fold scores: {[round(s * 100, 2) for s in cv_scores]}\n")
else:
    n_splits = min(5, int(np.min(np.bincount(y_encoded))))
    n_splits = max(2, n_splits) if np.min(np.bincount(y_encoded)) >= 2 else 0
    if n_splits >= 2:
        print(f"🔁 Stratified CV ({n_splits}-fold)...")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            ),
            X,
            y_encoded,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        print(f"   CV accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
        print(f"   Fold scores: {[round(s * 100, 2) for s in cv_scores]}\n")
    else:
        print("⚠️  Not enough samples per class for CV — skipping\n")

# ── Train Random Forest ───────────────────────────────────────────
print("🌲 Training Random Forest (class_weight='balanced')...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean() * 100

print(f"\n🎯 Hold-out accuracy ({split_name}): {accuracy:.2f}%\n")
print("📋 Per-sign breakdown:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=labels_all,
        target_names=list(le.classes_),
        zero_division=0,
    )
)

cm = confusion_matrix(y_test, y_pred, labels=labels_all)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print("🧩 Confusion matrix (rows=true, cols=pred):")
with pd.option_context("display.max_rows", 100, "display.max_columns", 100, "display.width", 200):
    print(cm_df.to_string())

# ── Save ──────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
# Retrain on ALL data for the shipped model (after reporting hold-out score)
final = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)
final.fit(X, y_encoded)
joblib.dump(final, "models/sign_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")

print("\n💾 Model saved to models/sign_model.pkl (trained on all samples)")
print("💾 Label encoder saved to models/label_encoder.pkl")
print(
    f"\n📌 SUMMARY: 63-d landmarks | split={split_name} | "
    "class_weight=balanced | final model fits all rows"
)
if not use_grouped:
    print(
        "   Recollect tip: for each sign, press SPACE → collect → finish, "
        "then run that sign again later for a 2nd session."
    )

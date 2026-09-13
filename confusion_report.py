"""
Confusion-matrix report for static signs — find which pairs need more samples.

Usage:
  python confusion_report.py
  python confusion_report.py --top 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "landmarks.csv"
MODEL = ROOT / "models" / "sign_model.pkl"
ENCODER = ROOT / "models" / "label_encoder.pkl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=12, help="Top confused pairs to print")
    args = parser.parse_args()

    if not DATA.is_file() or not MODEL.is_file():
        print("Need data/landmarks.csv and models/sign_model.pkl")
        return 1

    df = pd.read_csv(DATA)
    X = df.drop("label", axis=1).values
    y = df["label"].values
    le = joblib.load(ENCODER)
    model = joblib.load(MODEL)

    mask = np.isin(y, le.classes_)
    X, y = X[mask], y[mask]
    y_enc = le.transform(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
    )
    pred = model.predict(X_te)
    names = list(le.classes_)

    print(classification_report(y_te, pred, target_names=names, zero_division=0))

    cm = confusion_matrix(y_te, pred, labels=list(range(len(names))))
    pairs: list[tuple[int, str, str, int]] = []
    for i, true_name in enumerate(names):
        for j, pred_name in enumerate(names):
            if i == j:
                continue
            n = int(cm[i, j])
            if n > 0:
                pairs.append((n, true_name, pred_name, i))

    pairs.sort(reverse=True)
    print(f"=== Top {args.top} confused pairs (true → predicted) ===")
    for n, t, p, _ in pairs[: args.top]:
        print(f"  {n:4d}×  {t:12s} → {p}")

    print("\nCollect 40–80 extra samples for EACH side of the top pairs,")
    print("varying distance, angle (±20°), lighting, and hand (left/right if used).")
    print("Then: python train_model.py && python confusion_report.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

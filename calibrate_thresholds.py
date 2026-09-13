"""
Data-driven threshold suggestion for the static RandomForest.

Usage (from repo root, venv active):
  python calibrate_thresholds.py

Reads data/landmarks.csv + models/sign_model.pkl, sweeps thresholds,
prints the confidence where correct vs incorrect distributions cross,
and the F1-maximizing STATIC_THRESHOLD.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "landmarks.csv"
MODEL = ROOT / "models" / "sign_model.pkl"
ENCODER = ROOT / "models" / "label_encoder.pkl"


def main() -> int:
    if not DATA.is_file():
        print(f"Missing {DATA} — collect static samples first.")
        return 1
    if not MODEL.is_file():
        print(f"Missing {MODEL} — run train_model.py first.")
        return 1

    df = pd.read_csv(DATA)
    X = df.drop("label", axis=1).values
    y = df["label"].values
    le = joblib.load(ENCODER)
    model = joblib.load(MODEL)

    # Align labels with encoder
    mask = np.isin(y, le.classes_)
    X, y = X[mask], y[mask]
    y_enc = le.transform(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
    )
    # Use held-out set only for calibration (avoid optimistic bias)
    proba = model.predict_proba(X_te)
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    correct = pred == y_te

    correct_conf = conf[correct]
    wrong_conf = conf[~correct]

    print("=== Static confidence histograms (held-out) ===")
    print(f"  samples: {len(conf)}  correct: {correct.sum()}  wrong: {(~correct).sum()}")
    if len(correct_conf):
        print(
            f"  correct conf  p50={np.median(correct_conf):.3f}  "
            f"p10={np.percentile(correct_conf, 10):.3f}  "
            f"p90={np.percentile(correct_conf, 90):.3f}"
        )
    if len(wrong_conf):
        print(
            f"  wrong   conf  p50={np.median(wrong_conf):.3f}  "
            f"p90={np.percentile(wrong_conf, 90):.3f}"
        )
    else:
        print("  (no wrong predictions on this split)")

    # Crossover heuristic: threshold above most wrongs, below most corrects
    if len(wrong_conf) and len(correct_conf):
        crossover = 0.5 * (
            float(np.percentile(wrong_conf, 90)) + float(np.percentile(correct_conf, 10))
        )
        print(f"\n  crossover heuristic (wrong_p90 + correct_p10)/2 = {crossover:.3f}")
    else:
        crossover = 0.65
        print(f"\n  crossover fallback = {crossover:.3f}")

    best_t, best_f1 = 0.5, -1.0
    print("\n=== Threshold sweep (precision/recall via accept-if-conf>=t) ===")
    print(f"{'t':>6}  {'accept%':>8}  {'acc|acc':>8}  {'F1*':>8}")
    for t in np.linspace(0.50, 0.95, 19):
        take = conf >= t
        if take.sum() == 0:
            continue
        # Treat rejected frames as a separate "none" — F1 on accepted only is optimistic;
        # use a soft score: accuracy on accepted * coverage
        acc = (pred[take] == y_te[take]).mean()
        coverage = take.mean()
        # Macro-ish: reward both accuracy and not rejecting everything
        score = f1_score(y_te[take], pred[take], average="macro", zero_division=0)
        blended = 0.7 * score + 0.3 * coverage
        print(f"{t:6.2f}  {coverage*100:7.1f}%  {acc*100:7.1f}%  {score:8.3f}")
        if blended > best_f1:
            best_f1 = blended
            best_t = float(t)

    print("\n=== Suggestion for params.py ===")
    print(f"  STATIC_THRESHOLD = {best_t:.2f}   # best blended F1/coverage")
    print(f"  # or try crossover ≈ {crossover:.2f}")
    print("  DYNAMIC_THRESHOLD ≈ STATIC_THRESHOLD - 0.03  (LSTM scores run a bit softer)")
    print("\nRe-run after collecting more data; thresholds drift with class balance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

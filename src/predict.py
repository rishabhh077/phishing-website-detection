"""
predict.py
----------
Command-line tool: give it one or more URLs, it prints a verdict.

Usage:
    python src/predict.py "http://paypal-secure.verify-login.xyz/update"
    python src/predict.py "https://www.google.com" "http://192.168.1.1/login"
"""

import sys
import json
from pathlib import Path

import joblib
import pandas as pd

from features import extract_features, FEATURE_NAMES

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def load_artifacts():
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")  # may be None
    with open(MODELS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    with open(MODELS_DIR / "feature_importance.json") as f:
        importance_map = json.load(f)
    return model, scaler, feature_names, importance_map


def predict_url(url: str, model, scaler, feature_names, importance_map):
    feats = extract_features(url)
    X = pd.DataFrame([feats])[feature_names]
    X_in = scaler.transform(X) if scaler is not None else X

    proba_phishing = model.predict_proba(X_in)[0][1]
    label = "PHISHING" if proba_phishing >= 0.5 else "legit"

    # Top reasons: the features with the highest raw values among the
    # ones the model weighs most heavily, by precomputed permutation
    # importance (model-agnostic, saved once by train.py).
    reasons = sorted(
        ((name, importance_map.get(name, 0.0), X.iloc[0][name]) for name in feature_names),
        key=lambda t: t[1],
        reverse=True,
    )[:5]

    return label, proba_phishing, feats, reasons


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    model, scaler, feature_names, importance_map = load_artifacts()

    for url in sys.argv[1:]:
        label, proba, feats, reasons = predict_url(
            url, model, scaler, feature_names, importance_map
        )
        print(f"\nURL:      {url}")
        print(f"Verdict:  {label}  (phishing probability: {proba:.1%})")
        print("Top signals the model looked at:")
        for name, importance, value in reasons:
            print(f"  - {name:<22} value={value}")


if __name__ == "__main__":
    main()

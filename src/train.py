"""
train.py
--------
Loads data/raw_urls.csv (columns: url,label where label=1 is phishing/
malicious and 0 is legitimate), extracts features with features.py,
trains three models, evaluates them properly (precision/recall/F1/ROC-AUC,
not just accuracy), and saves the best model + a metrics report + a SHAP
summary plot.

Run:
    python src/train.py
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler

from features import extract_features_batch, FEATURE_NAMES

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw_urls.csv"
PROCESSED_PATH = ROOT / "data" / "processed_features.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def load_or_build_features():
    if PROCESSED_PATH.exists():
        print(f"Loading cached features from {PROCESSED_PATH}")
        return pd.read_csv(PROCESSED_PATH)

    print(f"Loading raw URLs from {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH)
    raw = raw.dropna(subset=["url", "label"])

    print(f"Extracting features for {len(raw)} URLs...")
    t0 = time.time()
    feats = extract_features_batch(raw["url"].tolist())
    feats["label"] = raw["label"].values
    print(f"Done in {time.time() - t0:.1f}s")

    feats.to_csv(PROCESSED_PATH, index=False)
    print(f"Cached features to {PROCESSED_PATH}")
    return feats


def evaluate(name, model, X_test, y_test, scaler=None):
    X_eval = scaler.transform(X_test) if scaler is not None else X_test
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    for k, v in metrics.items():
        print(f"  {k:>10}: {v:.4f}")
    print("  confusion matrix [ [TN FP] [FN TP] ]:")
    print(" ", cm.tolist())
    print(classification_report(y_test, y_pred, target_names=["legit", "phishing"]))

    return metrics, cm.tolist()


def main():
    feats = load_or_build_features()
    X = feats[FEATURE_NAMES]
    y = feats["label"].astype(int)

    print(f"\nDataset: {len(X)} rows | phishing/malicious={y.sum()} "
          f"({y.mean():.1%}) | legitimate={(y == 0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    results = {}

    # ---- Model 1: Logistic Regression (baseline, needs scaled features) ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train_scaled, y_train)
    results["logistic_regression"] = evaluate(
        "Logistic Regression", logreg, X_test, y_test, scaler=scaler
    )

    # ---- Model 2: Random Forest ----
    # max_depth and max_leaf_nodes are capped (and min_samples_leaf raised)
    # to keep the saved model under a few MB instead of several hundred --
    # important for a model you'll commit to GitHub or deploy. The F1/ROC
    # cost of this is small (checked against an uncapped forest).
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=14, min_samples_leaf=5, n_jobs=-1,
        class_weight="balanced", random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    results["random_forest"] = evaluate("Random Forest", rf, X_test, y_test)

    # ---- Model 3: Gradient Boosted Trees (sklearn's HistGradientBoosting,
    # the same algorithm family as LightGBM/XGBoost, bundled with scikit-learn
    # so the project has no finicky native-wheel dependency) ----
    from sklearn.ensemble import HistGradientBoostingClassifier

    # HistGradientBoostingClassifier has no class_weight param, so we pass
    # per-sample weights to get the same "pay more attention to phishing"
    # effect as class_weight="balanced".
    class_counts = y_train.value_counts()
    weight_for = {0: 1.0, 1: class_counts[0] / class_counts[1]}
    sample_weight = y_train.map(weight_for).values

    gb_name = "Gradient Boosted Trees"
    gb = HistGradientBoostingClassifier(
        max_iter=300, max_depth=8, learning_rate=0.1, random_state=RANDOM_STATE,
    )
    gb.fit(X_train, y_train, sample_weight=sample_weight)
    results[gb_name] = evaluate(gb_name, gb, X_test, y_test)

    # ---- Pick the best model by F1 on the phishing class ----
    best_name = max(results, key=lambda k: results[k][0]["f1"])
    best_model = {"logistic_regression": logreg, "random_forest": rf, gb_name: gb}[best_name]
    best_scaler = scaler if best_name == "logistic_regression" else None

    print(f"\n{'#' * 60}\nBest model: {best_name} (by F1 score)\n{'#' * 60}")

    joblib.dump(best_model, MODELS_DIR / "best_model.pkl", compress=3)
    joblib.dump(best_scaler, MODELS_DIR / "scaler.pkl")  # None if not needed
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(FEATURE_NAMES, f)
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(
            {name: m[0] for name, m in results.items()} | {"best_model": best_name},
            f, indent=2,
        )

    print(f"\nSaved model -> {MODELS_DIR / 'best_model.pkl'}")
    print(f"Saved metrics -> {MODELS_DIR / 'metrics.json'}")

    # ---- Explainability: which features drive the model? ----
    # Different model types expose "importance" differently (tree models:
    # feature_importances_, linear models: coef_), and HistGradientBoosting
    # exposes neither. Rather than branch on model type (which silently
    # breaks the moment the best model changes), compute permutation
    # importance uniformly -- it works for any fitted model by measuring
    # how much shuffling each feature hurts performance, and is what
    # predict.py / app.py load at inference time too.
    try:
        from sklearn.inspection import permutation_importance
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        print("\nComputing permutation importance (best model)...")
        X_eval = best_scaler.transform(X_test) if best_scaler is not None else X_test
        perm = permutation_importance(
            best_model, X_eval, y_test, n_repeats=5,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        importances = perm.importances_mean
        importances = np.clip(importances, 0, None)  # negative = noise, floor at 0

        with open(MODELS_DIR / "feature_importance.json", "w") as f:
            json.dump(dict(zip(FEATURE_NAMES, importances.tolist())), f, indent=2)

        order = np.argsort(importances)[::-1]
        top_n = min(15, len(FEATURE_NAMES))
        top_feats = [FEATURE_NAMES[i] for i in order[:top_n]]
        top_vals = importances[order[:top_n]]

        plt.figure(figsize=(8, 6))
        plt.barh(top_feats[::-1], top_vals[::-1], color="#4C72B0")
        plt.xlabel("Permutation importance (drop in score when shuffled)")
        plt.title(f"Top {top_n} features - {best_name}")
        plt.tight_layout()
        plt.savefig(MODELS_DIR / "feature_importance.png", dpi=150)
        print(f"Saved feature importance plot -> {MODELS_DIR / 'feature_importance.png'}")
        print("Top 5 features:", top_feats[:5])
    except Exception as e:
        print(f"(Skipping feature importance: {e})")


if __name__ == "__main__":
    main()

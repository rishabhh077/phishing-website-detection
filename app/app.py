"""
app.py
------
Streamlit demo for the phishing URL detector.

Run:
    streamlit run app/app.py
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import extract_features  # noqa: E402

MODELS_DIR = ROOT / "models"

st.set_page_config(page_title="Phishing URL Detector", page_icon="🛡️", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    with open(MODELS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    with open(MODELS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    with open(MODELS_DIR / "feature_importance.json") as f:
        importance_map = json.load(f)
    return model, scaler, feature_names, metrics, importance_map


model, scaler, feature_names, metrics, importance_map = load_artifacts()
best_name = metrics.get("best_model", "model")
best_metrics = metrics.get(best_name, {})

st.title("🛡️ Phishing Website Detector")
st.caption(
    "Paste a URL below. The model looks only at the URL's structure "
    "(length, special characters, suspicious keywords, TLD, etc.) — "
    "it never visits the page."
)

with st.sidebar:
    st.subheader("Model info")
    st.write(f"**Best model:** {best_name.replace('_', ' ').title()}")
    if best_metrics:
        st.metric("ROC-AUC", f"{best_metrics['roc_auc']:.3f}")
        st.metric("Recall (phishing caught)", f"{best_metrics['recall']:.1%}")
        st.metric("Precision", f"{best_metrics['precision']:.1%}")
    st.caption(
        "Trained on ~415k real-world URLs. See the project README for "
        "dataset details and known limitations."
    )

url = st.text_input("URL to check", placeholder="e.g. https://www.example.com/login")

col1, col2 = st.columns([1, 1])
check_clicked = col1.button("Check URL", type="primary", use_container_width=True)
example_clicked = col2.button("Try a phishing-style example", use_container_width=True)

if example_clicked:
    url = "http://paypal-secure-login.verify-account.xyz/update/billing"
    check_clicked = True

if check_clicked and url.strip():
    feats = extract_features(url)
    X = pd.DataFrame([feats])[feature_names]
    X_in = scaler.transform(X) if scaler is not None else X
    proba = model.predict_proba(X_in)[0][1]
    is_phishing = proba >= 0.5

    if is_phishing:
        st.error(f"⚠️ Likely PHISHING — confidence {proba:.1%}")
    else:
        st.success(f"✅ Looks legitimate — phishing confidence only {proba:.1%}")

    st.progress(min(max(proba, 0.0), 1.0))

    # Explain the top contributing features for this prediction, using the
    # permutation importances computed once at training time (works the
    # same regardless of which model type ended up best).
    top = sorted(
        ((name, importance_map.get(name, 0.0), X.iloc[0][name]) for name in feature_names),
        key=lambda t: t[1],
        reverse=True,
    )[:8]

    st.subheader("What the model looked at")
    explain_df = pd.DataFrame(top, columns=["feature", "model_weight", "value_for_this_url"])
    st.dataframe(explain_df, hide_index=True, use_container_width=True)

    with st.expander("Full feature vector"):
        st.json(feats)

elif check_clicked:
    st.warning("Please enter a URL first.")

st.divider()
st.caption(
    "⚠️ Educational project — not a production security product. "
    "Lexical URL features alone cannot catch every phishing site "
    "(e.g. a compromised legitimate domain), and can occasionally "
    "flag unusual-but-safe URLs. Always verify suspicious links through "
    "a trusted source."
)

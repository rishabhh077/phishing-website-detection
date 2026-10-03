# Phishing Website Detection using Machine Leing

A machine-learning system that classifies a URL as **phishing** or
**legitimate** using only the URL's own structure — no page visit, no
WHOIS lookup needed to run it live. Built for a college cybersecurity
project / portfolio piece.

## Results

Trained and evaluated on a held-out 20% test split of ~415,000 URLs.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 80.6% | 0.436 | 0.718 | 0.543 | 0.854 |
| Random Forest | 90.5% | 0.658 | 0.844 | 0.740 | 0.954 |
| **Gradient Boosted Trees (best)** | **90.7%** | **0.656** | **0.889** | **0.755** | **0.966** |

**Gradient Boosted Trees** (scikit-learn's `HistGradientBoostingClassifier`)
was selected (highest F1 on the phishing class). It catches **88.9% of
phishing URLs** (recall) while keeping precision at 66%, and separates
the two classes well by probability (ROC-AUC 0.966). Recall is
prioritized over precision here deliberately — missing a phishing site
is worse than one extra warning. It's also under 500 KB on disk, so it
loads instantly and is easy to commit to a repo or deploy.

Full numbers: `models/metrics.json`. Feature importance chart:
`models/feature_importance.png`.

## How it works

1. **`src/features.py`** turns one URL string into 26 numeric features —
   length stats, counts of `.`/`-`/`@`/digits, Shannon entropy of the
   hostname, suspicious keywords (`login`, `verify`, `secure`, ...),
   suspicious TLDs, IP-address hosts, URL shorteners, etc. This same
   function is used both to build the training set and to score a URL
   typed into the app, so there's no train/serve mismatch.
2. **`src/build_dataset.py`** builds `data/raw_urls.csv` from the
   [faizann24/Using-machine-learning-to-detect-malicious-URLs](https://github.com/faizann24/Using-machine-learning-to-detect-malicious-URLs)
   dataset (~415k real URLs labeled good/bad), plus a curated set of
   well-known legitimate root domains. See **Known limitations** below
   for why that second part exists — it fixes a real bias found in the
   raw data.
3. **`src/train.py`** extracts features for every URL, trains Logistic
   Regression / Random Forest / Gradient Boosted Trees, evaluates all
   three properly (precision/recall/F1/ROC-AUC, not just accuracy, plus
   a confusion matrix), and saves the best one to `models/`.
4. **`src/predict.py`** is a CLI to check URLs from the terminal.
5. **`app/app.py`** is a Streamlit web demo: paste a URL, get a verdict,
   confidence score, and the top features that drove the decision.

## Project structure

```
phishing-detector/
├── data/
│   ├── ml_urls/              # cloned source dataset (git-ignored)
│   ├── raw_urls.csv          # combined + augmented url,label dataset (git-ignored, regenerate with build_dataset.py)
│   └── processed_features.csv# cached extracted features (git-ignored, regenerate with train.py)
├── src/
│   ├── features.py           # URL -> feature vector (the core logic)
│   ├── build_dataset.py      # builds data/raw_urls.csv
│   ├── train.py               # trains + evaluates + saves the model
│   └── predict.py            # CLI predictions
├── app/
│   └── app.py                 # Streamlit demo
├── models/                    # saved model, scaler, metrics, plot (generated)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce from scratch

```bash
# 1. Get the source dataset
git clone --depth 1 https://github.com/faizann24/Using-machine-learning-to-detect-malicious-URLs.git data/ml_urls

# 2. Build the combined, bias-corrected dataset
python src/build_dataset.py

# 3. Train all three models, save the best one + metrics + importance plot
python src/train.py
```

## Use it

```bash
# Command line
python src/predict.py "http://paypal-secure-login.verify-account.xyz/update"

# Web demo
streamlit run app/app.py
```

## Known limitations (be upfront about these — graders and interviewers like it)

- **Lexical features only.** The model never visits the page, so it
  can't see login forms, brand logos, or where a form actually submits
  to. A compromised *legitimate* domain hosting a phishing page would
  likely be missed. Adding content-based features (external form
  actions, favicon mismatch) and WHOIS domain age is the natural next
  step — see **Extending this project** below.
- **Training-data bias, partially corrected.** The source dataset's
  "legitimate" URLs are almost all scraped content pages (forum
  threads, articles) and essentially none are bare root domains —
  while ~3% of its "bad" URLs are. Trained naively, the model learned
  "no path → phishing" and flagged `https://www.google.com` as
  phishing with 93% confidence. `build_dataset.py` patches this by
  adding real bare-domain examples for ~100 well-known legitimate
  sites. This fixed the obvious cases, but domains and patterns outside
  that curated list (e.g. `.in`-ccTLD sites not in the list) are still
  under-represented and can produce borderline false positives — this
  is a good illustration of why *real* production systems need
  continuous data auditing, not just one offline training run.
- **"Phishing" label is really "malicious."** The underlying dataset
  labels spam/malware-hosting URLs as "bad" too, not only credential-
  phishing pages. For a narrower phishing-only model, retrain on
  phishing-specific sources like [PhishTank](https://phishtank.org/) or
  the [PhiUSIIL dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset).
- **No SHAP.** This environment's package mirror didn't have `shap` or
  `xgboost` available, so explainability uses scikit-learn's
  model-agnostic `permutation_importance` instead (computed once in
  `train.py`, saved to `models/feature_importance.json`, and reused by
  both the CLI and the app). If you have a normal internet connection,
  `pip install shap xgboost` and swapping them in for per-prediction
  explanations is a reasonable upgrade.

## Extending this project

- Add host-based features: domain age/expiry via `python-whois`, DNS
  record checks.
- Add content-based features: fetch the page with `requests`, parse
  with `BeautifulSoup` for external-domain form actions, password
  fields, iframe count, favicon mismatch.
- Try a character-level model (TF-IDF n-grams or a small CNN) directly
  on the raw URL string and compare it against the hand-crafted
  features.
- Add SHAP for per-prediction explanations in the app.
- Package the model behind a small API and build a browser extension
  that checks the current tab's URL automatically.
- Evaluate on **fresh** PhishTank URLs (not just a held-out split of
  the same dataset) to see how well it generalizes to phishing
  campaigns created after the training data was collected.

## Dataset credit

Base dataset: [faizann24/Using-machine-learning-to-detect-malicious-URLs](https://github.com/faizann24/Using-machine-learning-to-detect-malicious-URLs)
(MIT-style educational dataset, ~410k labeled URLs).

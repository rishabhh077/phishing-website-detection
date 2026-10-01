"""
build_dataset.py
-----------------
Builds data/raw_urls.csv (the single source of truth for training) from:

  1. The faizann24/Using-machine-learning-to-detect-malicious-URLs dataset
     (data.csv + data2.csv) -- ~411k real-world URLs labeled good/bad.

  2. A curated list of well-known legitimate root domains (google.com,
     github.com, wikipedia.org, ...), each expanded with a few common path
     patterns (bare domain, "/", "/login", "/search?q=...", etc.) and both
     http/https schemes.

Step 2 exists to fix a real bias discovered in step 1's data: in that
dataset, 0% of "good" URLs are bare root domains (every legitimate sample
has some path, because the dataset was built by scraping actual content
pages). Only ~3% of "bad" URLs are bare domains. A model trained on step 1
alone learns "no path => phishing" and misclassifies perfectly ordinary
URLs like "https://www.google.com". Step 2 patches that gap with real
examples of what a legitimate bare-domain visit looks like, which is
exactly the case a demo/portfolio app will be tested on most.

Run:
    python src/build_dataset.py
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "data" / "ml_urls" / "data"
OUT_PATH = ROOT / "data" / "raw_urls.csv"

# A broad, well-known set of legitimate domains across categories (search,
# social, shopping, banking, government, news, dev tools, Indian services
# too since that's a common user base for this kind of project). These are
# just well-known public domain names, used only as plain label=0 (legit)
# training examples -- nothing is fetched from them.
LEGIT_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "instagram.com", "x.com",
    "twitter.com", "linkedin.com", "wikipedia.org", "reddit.com", "amazon.com",
    "netflix.com", "microsoft.com", "apple.com", "github.com", "gitlab.com",
    "stackoverflow.com", "stackexchange.com", "cloudflare.com", "yahoo.com",
    "bing.com", "duckduckgo.com", "zoom.us", "salesforce.com", "adobe.com",
    "paypal.com", "ebay.com", "walmart.com", "target.com", "chase.com",
    "bankofamerica.com", "wellsfargo.com", "hdfcbank.com", "icicibank.com",
    "sbi.co.in", "irctc.co.in", "flipkart.com", "myntra.com", "zomato.com",
    "swiggy.com", "indiatimes.com", "ndtv.com", "bbc.com", "cnn.com",
    "nytimes.com", "theguardian.com", "reuters.com", "bloomberg.com",
    "dropbox.com", "slack.com", "notion.so", "trello.com", "atlassian.com",
    "spotify.com", "soundcloud.com", "twitch.tv", "pinterest.com", "quora.com",
    "medium.com", "wordpress.com", "shopify.com", "etsy.com", "airbnb.com",
    "booking.com", "expedia.com", "uber.com", "lyft.com", "doordash.com",
    "coursera.org", "udemy.com", "edx.org", "khanacademy.org", "mit.edu",
    "stanford.edu", "harvard.edu", "nasa.gov", "whitehouse.gov", "usa.gov",
    "india.gov.in", "uidai.gov.in", "incometax.gov.in", "nic.in",
    "w3.org", "mozilla.org", "python.org", "nodejs.org", "npmjs.com",
    "docker.com", "kubernetes.io", "aws.amazon.com", "azure.microsoft.com",
    "cloud.google.com", "digitalocean.com", "heroku.com", "vercel.com",
    "netlify.com", "wikipedia.com", "imdb.com", "rottentomatoes.com",
    "espn.com", "cricbuzz.com", "nseindia.com", "bseindia.com", "zerodha.com",
    "groww.in", "paytm.com", "phonepe.com", "razorpay.com", "stripe.com",
    "wordreference.com", "grammarly.com", "canva.com", "figma.com", "trello.com",
    "asana.com", "monday.com", "zendesk.com", "hubspot.com", "mailchimp.com",
    "wordpress.org", "github.io", "readthedocs.org", "arxiv.org", "nature.com",
    "sciencedirect.com", "springer.com", "ieee.org", "acm.org", "who.int",
]

PATH_VARIANTS = [
    "",                      # bare domain, e.g. "google.com"
    "/",
    "/login",
    "/about",
    "/search?q=weather+today",
    "/index.html",
    "/en/home",
]


def build_legit_augmentation():
    rows = []
    for domain in LEGIT_DOMAINS:
        for scheme in ("https://", "http://"):
            for path in PATH_VARIANTS:
                rows.append({"url": f"{scheme}{domain}{path}", "label": 0})
                rows.append({"url": f"{scheme}www.{domain}{path}", "label": 0})
    return pd.DataFrame(rows).drop_duplicates(subset="url")


def main():
    df1 = pd.read_csv(SOURCE_DIR / "data.csv")  # columns: url,label (good/bad)
    df2 = pd.read_csv(
        SOURCE_DIR / "data2.csv", header=None, names=["url", "label"]
    )  # all bad

    combined = pd.concat([df1, df2], ignore_index=True)
    combined["url"] = combined["url"].astype(str).str.strip()
    combined = combined[combined["url"].str.len() > 3]
    combined["label"] = combined["label"].map({"good": 0, "bad": 1})
    combined = combined.dropna(subset=["label"])
    combined["label"] = combined["label"].astype(int)

    legit_aug = build_legit_augmentation()
    print(f"Base dataset:        {len(combined)} rows")
    print(f"Legit augmentation:  {len(legit_aug)} rows (fixes the bare-domain bias)")

    full = pd.concat([combined, legit_aug], ignore_index=True)
    full = full.drop_duplicates(subset="url")

    print(f"\nFinal label distribution:\n{full['label'].value_counts()}")
    full.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH} ({len(full)} rows)")


if __name__ == "__main__":
    main()

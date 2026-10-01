"""
features.py
------------
Turns a raw URL string into a numeric feature vector for the phishing
classifier. Every feature here is computed ONLY from the URL string itself
(no network calls), so the exact same function is used:
  1. offline, in bulk, to build the training dataset, and
  2. online, in the Streamlit app, to score a URL a user just typed in.

That consistency matters: a lot of student projects train on one feature
set and score on another, which quietly breaks the model.
"""

import re
import math
from urllib.parse import urlparse

# Common URL-shortener domains (phishers love these to hide the real target)
SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly", "shorte.st", "rebrand.ly", "tiny.cc",
}

# TLDs that show up disproportionately often in phishing campaigns
SUSPICIOUS_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "work", "click", "link",
    "loan", "win", "review", "country", "science", "kim", "party",
}

# Keywords phishing pages commonly stuff into the URL to look legitimate
SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "secure", "account", "update", "bank",
    "confirm", "webscr", "password", "wallet", "billing", "suspend",
    "limited", "security", "support", "ebayisapi", "paypal",
]

IP_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)$"
)

FEATURE_NAMES = [
    "url_length", "hostname_length", "path_length", "num_dots",
    "num_hyphens", "num_at", "num_question", "num_equals", "num_slashes",
    "num_digits", "num_percent", "num_underscore", "num_ampersand",
    "digit_ratio", "is_ip_host", "num_subdomains", "has_https",
    "has_port", "shannon_entropy_host", "num_suspicious_words",
    "has_suspicious_tld", "is_shortened", "has_double_slash_redirect",
    "hostname_has_hyphen", "tld_length", "has_www",
]


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def extract_features(url: str) -> dict:
    """Return an ordered dict of features for a single URL string."""
    url = url.strip()
    # Make sure urlparse can find a scheme/host even if the user pasted a
    # bare domain like "paypal-secure-login.com".
    parse_target = url if "://" in url else "http://" + url

    # Malicious/malformed URLs (stray "[", "]", bad IPv6-looking netlocs,
    # control characters, etc.) can make urlparse raise instead of just
    # parsing leniently. That's exactly the kind of input this project
    # needs to survive, so fall back to an empty parse rather than crashing.
    try:
        parsed = urlparse(parse_target)
        hostname = parsed.hostname or ""
        path = parsed.path or ""
        port = parsed.port
        scheme = parsed.scheme
    except ValueError:
        hostname, path, port, scheme = "", "", None, ""

    labels = hostname.split(".") if hostname else []
    tld = labels[-1].lower() if len(labels) > 1 else ""

    feats = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_at": url.count("@"),
        "num_question": url.count("?"),
        "num_equals": url.count("="),
        "num_slashes": url.count("/"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_percent": url.count("%"),
        "num_underscore": url.count("_"),
        "num_ampersand": url.count("&"),
        "digit_ratio": (sum(c.isdigit() for c in url) / len(url)) if url else 0.0,
        "is_ip_host": int(bool(IP_PATTERN.match(hostname))),
        "num_subdomains": max(len(labels) - 2, 0),
        "has_https": int(scheme == "https"),
        "has_port": int(port is not None),
        "shannon_entropy_host": _shannon_entropy(hostname),
        "num_suspicious_words": sum(
            1 for w in SUSPICIOUS_WORDS if w in url.lower()
        ),
        "has_suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "is_shortened": int(hostname.lower() in SHORTENERS),
        "has_double_slash_redirect": int(url.rfind("//") > 7),
        "hostname_has_hyphen": int("-" in hostname),
        "tld_length": len(tld),
        "has_www": int(hostname.lower().startswith("www.")),
    }
    return feats


def extract_features_batch(urls):
    """Vectorized helper: list[str] -> pandas.DataFrame of features."""
    import pandas as pd
    rows = [extract_features(u) for u in urls]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


if __name__ == "__main__":
    # Quick manual sanity check
    samples = [
        "https://www.google.com",
        "http://paypal-secure-login.com.verify-account.xyz/signin",
        "http://192.168.1.1/update/account@paypal.com",
        "https://bit.ly/3xyzAbc",
    ]
    for u in samples:
        print(u)
        print(extract_features(u))
        print()

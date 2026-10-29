"""
Phishing Link Detector
Cyber Security and Laws - S.E. Information Technology (SPPU)

Detects phishing URLs using lexical/host-based features and a Random Forest
classifier. Works fully offline: the URL is never opened or fetched.

Usage:
    python phishing_detector.py train                 # train on built-in sample data
    python phishing_detector.py train --csv data.csv  # train on CSV (columns: url,label ; 1=phishing)
    python phishing_detector.py check "http://paypa1-secure-login.xyz/verify?id=1"
    python phishing_detector.py                       # interactive mode
"""
import argparse
import math
import random
import re
import string
import sys
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

MODEL_PATH = "phishing_model.joblib"

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "update", "secure", "account", "banking",
    "confirm", "password", "suspend", "wallet", "paypal", "invoice", "free",
    "bonus", "alert", "unlock", "billing", "support", "kyc", "refund",
]
SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "cutt.ly", "rb.gy"}
RISKY_TLDS = {"xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "work", "support", "zip", "icu", "buzz"}
IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

FEATURE_NAMES = [
    "url_length", "host_length", "path_length", "num_dots", "num_hyphens_host",
    "num_digits", "digit_ratio", "num_subdomains", "has_at", "has_ip_host",
    "is_https", "has_port", "has_punycode", "double_slash_in_path", "num_query_params",
    "suspicious_word_count", "is_shortener", "risky_tld", "host_entropy", "special_char_count",
]


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    probs = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in probs)


def extract_features(url: str) -> list:
    """Convert a URL into a numeric feature vector (see FEATURE_NAMES)."""
    url = url.strip()
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    parts = host.split(".")
    tld = parts[-1] if parts else ""
    low = url.lower()
    digits = sum(c.isdigit() for c in url)
    return [
        len(url),
        len(host),
        len(path),
        url.count("."),
        host.count("-"),
        digits,
        digits / max(len(url), 1),
        max(len(parts) - 2, 0),
        int("@" in url),
        int(bool(IP_RE.match(host))),
        int(parsed.scheme == "https"),
        int(parsed.port is not None),
        int("xn--" in host),
        int("//" in path),
        len([q for q in parsed.query.split("&") if q]),
        sum(w in low for w in SUSPICIOUS_WORDS),
        int(host in SHORTENERS),
        int(tld in RISKY_TLDS),
        _entropy(host),
        sum(c in "%$!*,;=~#" for c in url),
    ]


# --------------------------------------------------------------------------
# Built-in sample data generator (used when no CSV dataset is supplied).
# For real evaluation use PhishTank / OpenPhish (phishing) + Tranco (benign).
# --------------------------------------------------------------------------
_BRANDS = ["paypal", "amazon", "google", "microsoft", "apple", "sbi", "hdfcbank", "icicibank", "netflix", "facebook"]
_LEGIT = ["wikipedia.org", "github.com", "python.org", "stackoverflow.com", "bbc.com", "nytimes.com",
          "coursera.org", "mit.edu", "unipune.ac.in", "irctc.co.in", "flipkart.com", "amazon.in",
          "microsoft.com", "google.com", "apple.com", "paypal.com", "hdfcbank.com", "sbi.co.in"]
_WORDS = ["news", "docs", "blog", "shop", "learn", "course", "article", "product", "help", "about", "search"]


def _rand(n, alphabet=string.ascii_lowercase + string.digits):
    return "".join(random.choice(alphabet) for _ in range(n))


def _legit_url():
    d = random.choice(_LEGIT)
    sub = random.choice(["", "www.", "docs.", "en.", "blog.", "accounts."])
    path = "/".join(random.choice(_WORDS) for _ in range(random.randint(0, 3)))
    url = f"{random.choice(['https', 'https', 'https', 'http'])}://{sub}{d}/{path}"
    if random.random() < 0.15:  # benign URLs may legitimately carry login/query words
        url += f"{'/' if not url.endswith('/') else ''}login?next=/home&lang=en"
    return url


def _phish_url():
    brand = random.choice(_BRANDS)
    word = random.choice(SUSPICIOUS_WORDS)
    kind = random.randint(0, 5)
    tld = random.choice(["xyz", "top", "tk", "click", "com", "net", "info", "in"])
    if kind == 0:  # brand-in-subdomain
        host = f"{brand}.{word}-{_rand(5)}.{tld}"
    elif kind == 1:  # hyphenated typosquat
        host = f"{brand.replace('a', '4', 1)}-{word}-{_rand(3)}.{tld}"
    elif kind == 2:  # raw IP
        host = ".".join(str(random.randint(1, 254)) for _ in range(4))
    elif kind == 3:  # random-looking host
        host = f"{_rand(random.randint(10, 18))}.{tld}"
    elif kind == 4:  # '@' trick
        host = f"{brand}.com@{_rand(8)}.{tld}"
    else:  # clean-looking but malicious path
        host = f"{_rand(6)}{brand}.{tld}"
    path = f"/{word}/{_rand(random.randint(4, 12))}.php?id={_rand(6)}&session={_rand(10)}"
    scheme = "https" if random.random() < 0.4 else "http"  # many phishing sites now use HTTPS
    return f"{scheme}://{host}{path}"


def make_sample_dataset(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    rows = [(_legit_url(), 0) for _ in range(n // 2)] + [(_phish_url(), 1) for _ in range(n // 2)]
    random.shuffle(rows)
    return pd.DataFrame(rows, columns=["url", "label"])


# --------------------------------------------------------------------------
def train(csv_path=None):
    df = pd.read_csv(csv_path) if csv_path else make_sample_dataset()
    X = np.array([extract_features(u) for u in df["url"]])
    y = df["label"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(f"Samples: {len(df)}  (train {len(X_tr)}, test {len(X_te)})")
    print(f"Accuracy: {accuracy_score(y_te, pred):.4f}")
    print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_te, pred))
    print(classification_report(y_te, pred, target_names=["legitimate", "phishing"]))
    imp = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda t: -t[1])[:8]
    print("Top features:", ", ".join(f"{n}={v:.3f}" for n, v in imp))
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def _rule_reasons(url: str) -> list:
    f = dict(zip(FEATURE_NAMES, extract_features(url)))
    reasons = []
    if f["has_ip_host"]: reasons.append("uses an IP address instead of a domain name")
    if f["has_at"]: reasons.append("contains '@' (hides the real destination)")
    if f["is_shortener"]: reasons.append("uses a URL shortener")
    if f["risky_tld"]: reasons.append("uses a high-risk top-level domain")
    if f["has_punycode"]: reasons.append("contains punycode (possible look-alike characters)")
    if f["suspicious_word_count"] >= 2: reasons.append("has several credential/urgency keywords")
    if f["num_hyphens_host"] >= 2: reasons.append("domain contains many hyphens")
    if f["num_subdomains"] >= 3: reasons.append("has an unusually deep subdomain chain")
    if not f["is_https"]: reasons.append("does not use HTTPS")
    return reasons


def check(url: str, model=None) -> dict:
    if model is None:
        model = joblib.load(MODEL_PATH)
    prob = float(model.predict_proba([extract_features(url)])[0][1])
    label = "PHISHING" if prob >= 0.5 else "LEGITIMATE"
    return {"url": url, "verdict": label, "phishing_probability": round(prob, 3), "reasons": _rule_reasons(url)}


def show(result: dict):
    print(f"\nURL      : {result['url']}")
    print(f"Verdict  : {result['verdict']}  (phishing probability {result['phishing_probability']:.1%})")
    for r in result["reasons"]:
        print(f"  - {r}")


def main():
    ap = argparse.ArgumentParser(description="Phishing Link Detector")
    sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("train"); t.add_argument("--csv")
    c = sub.add_parser("check"); c.add_argument("url")
    args = ap.parse_args()
    if args.cmd == "train":
        train(args.csv)
    elif args.cmd == "check":
        show(check(args.url))
    else:
        model = joblib.load(MODEL_PATH)
        print("Phishing Link Detector - type a URL (or 'q' to quit)")
        while True:
            u = input("URL> ").strip()
            if u.lower() in {"q", "quit", "exit"}:
                break
            if u:
                show(check(u, model))


if __name__ == "__main__":
    main()

"""
Phishing Link Detector - web interface
Run:  python app.py   ->  http://127.0.0.1:5000
The URL is analysed as text only; it is never opened and never stored.
"""
import os
import joblib
from flask import Flask, jsonify, render_template, request

import phishing_detector as pd_
from phishing_detector import FEATURE_NAMES, MODEL_PATH, check, extract_features

app = Flask(__name__)

# Train automatically on first run so the app never crashes on a missing model
if not os.path.exists(MODEL_PATH):
    print("Model not found - training on sample data (one-time)...")
    pd_.train()
model = joblib.load(MODEL_PATH)

EXAMPLES = [
    "https://www.wikipedia.org/wiki/Cyber_security",
    "https://github.com/python/cpython",
    "http://paypa1-secure-login.xyz/verify/a8x2k.php?id=12&session=ab34",
    "http://192.168.4.7/login.php?id=1",
]


def analyse(url: str) -> dict:
    result = check(url, model)
    p = result["phishing_probability"]
    result["risk"] = "high" if p >= 0.7 else "medium" if p >= 0.4 else "low"
    result["percent"] = round(p * 100, 1)
    result["features"] = list(zip(FEATURE_NAMES, [round(v, 3) for v in extract_features(url)]))
    return result


@app.route("/", methods=["GET", "POST"])
def index():
    url, result, error = "", None, None
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        if not url:
            error = "Please enter a URL to check."
        elif len(url) > 2000:
            error = "That URL is too long (max 2000 characters)."
        elif " " in url:
            error = "A URL cannot contain spaces."
        else:
            result = analyse(url)
    return render_template("index.html", url=url, result=result, error=error, examples=EXAMPLES)


@app.route("/api/check")
def api_check():
    """JSON endpoint:  /api/check?url=http://example.com"""
    url = request.args.get("url", "").strip()
    if not url:
        return jsonify(error="missing 'url' parameter"), 400
    r = analyse(url)
    r.pop("features")
    return jsonify(r)


if __name__ == "__main__":
    app.run(debug=False)

import re
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from textblob import TextBlob

# ---- Copy of your helper functions (unchanged) ----
def extract_url_if_exists(text):
    if text is None:
        return ""
    url_regex = r'(https?://[^\s]+|www\.[^\s]+)'
    m = re.search(url_regex, str(text))
    return m.group(0) if m else ""

def detect_urgent_phrases(text):
    phrases = [
        "verify your account", "act now", "urgent", "immediate action", "click here",
        "update your information", "suspend", "limited time", "reset your password", "login now"
    ]
    t = (text or "").lower()
    return int(any(p in t for p in phrases))

def detect_suspicious_anchor_text(html):
    # If you pass plain text, it will just return 0 (safe).
    if not html:
        return 0
    anchor_regex = r'<a\s+href="([^"]+)">([^<]+)</a>'
    matches = re.findall(anchor_regex, html)
    return int(any(href and anchor and anchor not in href for href, anchor in matches))

def extract_sentiment_score(text):
    try:
        return TextBlob(text or "").sentiment.polarity
    except Exception:
        return 0.0

def extract_features_from_url(url):
    u = str(url or "")
    return {
        "url_length": len(u),
        "num_dots": u.count("."),
        "has_https": int("https" in u.lower()),
        "has_at_symbol": int("@" in u),
        "has_hyphen": int("-" in u),
        "has_ip_address": int(bool(re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", u))),
        "subdomain_count": len(u.split("//")[-1].split("/")[0].split(".")) - 2 if "." in u else 0,
    }

# ---- Master builder: must match your training stacking order ----
def build_feature_matrix(subject, body_text, body_html, urls_list, tfidf_vectorizer):
    """
    Reconstruct the SAME combined features you used in training:
      X_combined = hstack([X_url_sparse, X_text_sparse, X_tfidf])
    """
    subject = subject or ""
    body_text = body_text or ""
    body_html = body_html or ""

    # 1) Text to feed TF-IDF (you trained on df['text'], so concatenate subject + body_text)
    text_for_tfidf = f"{subject} {body_text}".strip()
    X_tfidf = tfidf_vectorizer.transform([text_for_tfidf])

    # 2) URL features: if no urls provided, try to extract one from text
    url = urls_list[0] if urls_list else extract_url_if_exists(f"{subject} {body_text} {body_html}")
    url_feats = pd.DataFrame([extract_features_from_url(url)])
    X_url_sparse = csr_matrix(url_feats.values)

    # 3) Custom text features (urgent phrases, anchor mismatch, sentiment)
    urgent_phrase = detect_urgent_phrases(body_text)
    suspicious_anchor = detect_suspicious_anchor_text(body_html)
    sentiment_score = extract_sentiment_score(body_text)
    X_text_custom = csr_matrix(np.array([[urgent_phrase, suspicious_anchor, sentiment_score]], dtype=float))

    # 4) Combine in the SAME ORDER as training
    X_combined = hstack([X_url_sparse, X_text_custom, X_tfidf])

    return X_combined

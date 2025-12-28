# app.py — PhishGuard (PSL-aware + brand rules + edit distance + robust UI)

import os, re, csv, json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import tldextract  # PSL / eTLD+1 parsing

# ───────────────────────── Page config ─────────────────────────
st.set_page_config(page_title="PhishGuard", page_icon="🛡️", layout="centered")

# ───────────────────────── Files / paths ───────────────────────
MODEL_FILE = "models/phishing_model.joblib"
META_FILE  = "models/phishing_meta.json"   # optional: contains {"threshold": ...}

HISTORY_LOG    = Path("history_log.jsonl")
FEEDBACK_LOG   = Path("feedback_log.tsv")
USER_ALLOWLIST = Path("trusted_userlist.txt")
SUGGEST_QUEUE  = Path("allowlist_suggestions.txt")
ADMIN_PIN      = os.getenv("ADMIN_PIN", "")  # optional admin PIN for the Admin page

# Built-in safe (global) domains. User allowlist is merged on top.
SAFE_DOMAINS = {
    "google.com","amazon.com","wikipedia.org","microsoft.com","github.com",
    "facebook.com","apple.com","linkedin.com","youtube.com","instagram.com",
    "x.com","paypal.com","chase.com","bankofamerica.com","wellsfargo.com",
    "bestbuy.com","target.com","walmart.com","flipkart.com","myntra.com"
}

# ───────────────────────── Brand rules / helpers ───────────────
# Reserved TLDs (RFC 2606)
RESERVED_TLDS = (".test", ".example", ".invalid", ".localhost")

# Domains we allow for brand tokens
BRAND_ALLOW = {
    "google.com","paypal.com","microsoft.com","amazon.com",
    "apple.com","wellsfargo.com","bankofamerica.com","walmart.com"
}

# Words to detect in SLD/host/path (normalized)
BRAND_TOKENS = {
    "google","paypal","microsoft","amazon","apple",
    "wellsfargo","bankofamerica","walmart"
}

# Basic leetspeak map
LEET_MAP = str.maketrans({
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t",
    "8": "b", "$": "s", "@": "a", "!": "i"
})

def de_leet(s: str) -> str:
    return s.translate(LEET_MAP)

# Confusable pairs that often mimic characters visually
CONFUSABLE_PAIRS = [
    ("rn", "m"),
    ("vv", "w"),
    ("cl", "d"),
]

def normalize_confusables(s: str) -> str:
    out = s
    for a, b in CONFUSABLE_PAIRS:
        out = out.replace(a, b)
    return out

def levenshtein(a: str, b: str) -> int:
    a = a.lower(); b = b.lower()
    if a == b: return 0
    if not a:   return len(b)
    if not b:   return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins  = prev[j] + 1
            dele = cur[j-1] + 1
            sub  = prev[j-1] + (ca != cb)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]

def brand_distance_hit(sld_norm: str, max_dist: int = 1):
    """
    Returns (hit, closest_token, distance) if sld_norm is within max_dist
    edits of any BRAND_TOKENS. Minimum length gate to reduce false positives.
    """
    if len(sld_norm) < 5:
        return False, "", 99
    best_token, best_d = "", 99
    for tok in BRAND_TOKENS:
        d = levenshtein(sld_norm, tok)
        if d < best_d:
            best_token, best_d = tok, d
    return (best_d <= max_dist), best_token, best_d

# ───────────────────────── Tiny URL helpers ────────────────────
def ensure_scheme(u: str) -> str:
    return u if u.startswith(("http://","https://")) else "https://" + u

def clean_url(u: str) -> str:
    u = (u or "").strip()
    u = re.sub(r"#.*$", "", u)  # strip fragment
    u = u.replace(" ", "")
    u = ensure_scheme(u)
    return u.lower()

def extract_host(u: str) -> str:
    return (urlparse(u).hostname or "").lower()

def base_domain_from_host(host: str) -> str:
    ext = tldextract.extract(host or "")
    return f"{ext.domain}.{ext.suffix}".lower() if ext.suffix else (host or "").lower()

def subdomain_depth_psl(host: str, registrable: str) -> int:
    if not host or not registrable or registrable not in host:
        return host.count(".")
    return max(0, host.count(".") - registrable.count("."))

def sigmoid(x: float) -> float:
    x = max(min(float(x), 20.0), -20.0)  # prevent overflow
    return 1.0 / (1.0 + np.exp(-x))

# ───────────────────────── Logs ────────────────────────────────
def log_history(row: dict):
    try:
        with HISTORY_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass  # never crash UI on logging

def read_last_history(n=10):
    if not HISTORY_LOG.exists():
        return []
    try:
        lines = HISTORY_LOG.read_text(encoding="utf-8").splitlines()
        items = [json.loads(x) for x in lines[-n:]] if lines else []
        return list(reversed(items))
    except Exception:
        return []

def log_feedback(row: dict):
    new = not FEEDBACK_LOG.exists()
    try:
        with FEEDBACK_LOG.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["ts","url","host","decision","score","threshold","user_label","notes"]
            )
            if new: w.writeheader()
            w.writerow(row)
    except Exception:
        pass

# ───────────────────────── Allowlist helpers ───────────────────
def load_user_allowlist() -> set[str]:
    try:
        if USER_ALLOWLIST.exists():
            return {ln.strip().lower()
                    for ln in USER_ALLOWLIST.read_text(encoding="utf-8").splitlines()
                    if ln.strip()}
    except Exception:
        pass
    return set()

def save_user_allowlist(domains: set[str]):
    try:
        USER_ALLOWLIST.write_text("\n".join(sorted(domains)) + "\n", encoding="utf-8")
    except Exception:
        pass

def load_suggestions() -> list[str]:
    try:
        if SUGGEST_QUEUE.exists():
            uniq, seen = [], set()
            for ln in SUGGEST_QUEUE.read_text(encoding="utf-8").splitlines():
                d = ln.strip().lower()
                if d and d not in seen:
                    uniq.append(d); seen.add(d)
            return uniq
    except Exception:
        pass
    return []

def set_suggestions(domains: list[str]):
    try:
        SUGGEST_QUEUE.write_text("\n".join(domains) + ("\n" if domains else ""), encoding="utf-8")
    except Exception:
        pass

def push_suggestion(domain: str):
    cur = set(load_suggestions())
    if domain and (domain not in cur):
        try:
            with SUGGEST_QUEUE.open("a", encoding="utf-8") as f:
                f.write(domain + "\n")
        except Exception:
            pass

def is_allowlisted_host(host: str, merged: set[str]) -> tuple[bool, str]:
    for sd in merged:
        if host == sd or host.endswith("." + sd):
            return True, sd
    return False, ""

# ───────────────────────── (Optional) numeric features (for wrappers) ─────────
def _has_ip(url):
    try:
        host = (urlparse(url).netloc or url).split('@')[-1].split(':')[0]
        return 1 if re.match(r"^(?:\d{1,3}\.){3}\d{1,3}$", host) else 0
    except: return 0

def _count_chars(url, chars="-._@?=&%/"): return sum(url.count(c) for c in chars)

def _pct_encoded_ratio(url):
    try: return len(re.findall(r"%[0-9A-Fa-f]{2}", url)) / max(len(url), 1)
    except: return 0.0

def _get_tld_len(url):
    try:
        host = (urlparse(url).netloc or "").split('@')[-1].split(':')[0]
        parts = host.split('.'); return len(parts[-1]) if len(parts)>1 else 0
    except: return 0

def _url_features_series(url):
    u = str(url).strip(); p = urlparse(u)
    host, path, query, scheme = p.netloc or "", p.path or "", p.query or "", (p.scheme or "").lower()
    return pd.Series({
        "len_url": len(u), "len_host": len(host), "len_path": len(path),
        "num_digits": sum(ch.isdigit() for ch in u), "num_dots": u.count('.'),
        "num_hyphens": u.count('-'), "num_special": _count_chars(u),
        "has_https": 1 if scheme == "https" else 0, "has_ip": _has_ip(u),
        "pct_encoded": _pct_encoded_ratio(u), "tld_len": _get_tld_len(u),
        "at_in_url": 1 if '@' in u else 0,
        "subdir_depth": max(0, len([t for t in path.split('/') if t])),
        "query_kv_pairs": len(query.split('&')) if query else 0,
    })

def _build_numeric_features(df_like):
    if isinstance(df_like, pd.DataFrame):
        s = df_like["url"].astype(str)
    elif isinstance(df_like, pd.Series):
        s = df_like.astype(str)
    else:
        s = pd.Series(list(df_like)).astype(str)
    return s.apply(_url_features_series).astype(np.float32)

# ───────────────────────── Legacy wrappers (if your joblib uses them) ─────────
class LinearSVMOldWrapper:
    def __init__(self, pipe): self.pipe = pipe
    def predict(self, X):
        s = X["url"] if isinstance(X, pd.DataFrame) else pd.Series(list(X))
        return self.pipe.predict(s.to_numpy())
    def decision_function(self, X):
        s = X["url"] if isinstance(X, pd.DataFrame) else pd.Series(list(X))
        return self.pipe.decision_function(s.to_numpy())
    def predict_proba(self, X):
        d = self.decision_function(X)
        p = (d - d.min()) / (d.max() - d.min() + 1e-9)
        return np.vstack([1 - p, p]).T

class HashSGDPipeline:
    def __init__(self, vec, clf): self.vec, self.clf = vec, clf
    def predict_proba(self, X):
        s = X["url"] if isinstance(X, pd.DataFrame) else pd.Series(list(X))
        Xt = self.vec.transform(s)
        return self.clf.predict_proba(Xt)
    def predict(self, X):
        s = X["url"] if isinstance(X, pd.DataFrame) else pd.Series(list(X))
        Xt = self.vec.transform(s)
        return self.clf.predict(Xt)

class HashSGDSVMWrapper:
    def __init__(self, vec, clf): self.vec, self.clf = vec, clf
    def decision_function(self, X):
        s = X["url"] if isinstance(X, pd.DataFrame) else pd.Series(list(X))
        Xt = self.vec.transform(s)
        return self.clf.decision_function(Xt)
    def predict(self, X):
        s = X["url"] if isinstance(X, pd.DataFrame) else pd.Series(list(X))
        Xt = self.vec.transform(s)
        return self.clf.predict(Xt)
    def predict_proba(self, X):
        d = self.decision_function(X)
        p = (d - d.min())/(d.max()-d.min()+1e-9)
        return np.vstack([1-p, p]).T

class HashNBWrapper:
    def __init__(self, vec, clf): self.vec, self.clf = vec, clf
    def predict_proba(self, X):
        s = X["url"] if isinstance(X, pd.DataFrame) else pd.Series(list(X))
        Xt = self.vec.transform(s)
        return self.clf.predict_proba(Xt)
    def predict(self, X):
        s = X["url"] if isinstance(X, pd.DataFrame) else pd.Series(list(X))
        Xt = self.vec.transform(s)
        return self.clf.predict(Xt)

class RFNumericWrapper:
    def __init__(self, rf): self.rf = rf
    def predict_proba(self, X): return self.rf.predict_proba(_build_numeric_features(X))
    def predict(self, X):      return self.rf.predict(_build_numeric_features(X))

class XGBNumericWrapper:
    def __init__(self, xgb): self.xgb = xgb
    def predict_proba(self, X): return self.xgb.predict_proba(_build_numeric_features(X).values)
    def predict(self, X):      return self.xgb.predict(_build_numeric_features(X).values)

# ───────────────────────── Model + threshold ───────────────────
def _load_threshold(path: str) -> float:
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "threshold" in meta:       return float(meta["threshold"])
        if "Thr_at_Best_F1" in meta:  return float(meta["Thr_at_Best_F1"])
    except Exception:
        pass
    return 0.10  # sensible default for phishing risk

@st.cache_resource
def load_model_and_threshold():
    mp = Path(MODEL_FILE)
    if not mp.exists():
        raise FileNotFoundError(f"Model missing: {mp.resolve()}")
    model = joblib.load(mp)
    thr = _load_threshold(META_FILE) if Path(META_FILE).exists() else 0.10
    st.sidebar.success(f"Loaded model: {mp.resolve()}")
    return model, float(thr)

model, default_thr = load_model_and_threshold()

def score_one_url(url: str) -> float:
    cleaned = clean_url(url)
    df = pd.DataFrame({"url": [cleaned]})
    try:
        if hasattr(model, "predict_proba"):
            try:
                return float(model.predict_proba(df)[:, 1][0])
            except Exception:
                return float(model.predict_proba([cleaned])[:, 1][0])
        if hasattr(model, "decision_function"):
            try:
                d = float(model.decision_function(df)[0])
            except Exception:
                d = float(model.decision_function([cleaned])[0])
            return sigmoid(d)
        # fallback to label
        try:
            y = int(model.predict(df)[0])
        except Exception:
            y = int(model.predict([cleaned])[0])
        return float(y)
    except Exception:
        # If model fails, return neutral score; UI will show error
        return 0.0

# ───────────────────────── Rule overlay ────────────────────────
def rule_overlay(url: str, host: str, registrable: str):
    """
    Returns (bump, flags, force_phish, sld_raw, sld_deleet, sld_norm, dist_hit, tok, dist)
    bump        : small numeric bump to risk score for warning signals
    force_phish : if True, label PHISHING regardless of threshold (brand misuse)
    """
    p = urlparse(url)
    path = (p.path or "").lower()

    reserved  = any(host.endswith(t) for t in RESERVED_TLDS)
    puny      = ("xn--" in host)
    weird_port= (p.port not in (None, 80, 443, 22))

    # PSL extraction
    ext       = tldextract.extract(host or "")
    sld_raw   = (ext.domain or "").lower()
    sld_deleet= de_leet(sld_raw)
    sld_norm  = normalize_confusables(sld_deleet)

    # subdomain depth
    deep = subdomain_depth_psl(host, registrable)

    # Literal brand hits (in normalized sld, full host, or path)
    literal_hit = any(
        (tok in sld_norm) or (tok in host) or (tok in path)
        for tok in BRAND_TOKENS
    )

    # Edit-distance brand hit
    dist_hit, closest_tok, dist_val = brand_distance_hit(sld_norm, max_dist=1)

    brand_hit = literal_hit or dist_hit
    brand_ok  = (registrable in BRAND_ALLOW)

    flags = []
    if reserved:   flags.append("reserved_tld")
    if puny:       flags.append("punycode")
    if deep >= 3:  flags.append(f"deep_subdomain:{deep}")
    if brand_hit and not brand_ok:
        flags.append("brand_mismatch")
    if dist_hit:
        flags.append(f"brand_edit:{closest_tok}:{dist_val}")
    if weird_port: flags.append("weird_port")

    # conservative small bump; brand mismatch is a hard override
    bump = 0.15 if (reserved or puny or deep >= 3 or weird_port or (brand_hit and not brand_ok)) else 0.0
    force_phish = (brand_hit and not brand_ok)

    return bump, flags, force_phish, sld_raw, sld_deleet, sld_norm, dist_hit, closest_tok, dist_val

# ───────────────────────── UI ──────────────────────────────────
st.sidebar.title("PhishGuard")
page = st.sidebar.radio("Navigate", ["Scan","Dashboard","Tips","Feedback","Admin"])
threshold = st.sidebar.slider("Decision threshold", 0.00, 1.00, float(default_thr), 0.01)

user_allow = load_user_allowlist()
merged_allow = SAFE_DOMAINS | user_allow

# ───────── Scan
if page == "Scan":
    st.title("🛡️ Scan a Link")
    url_in = st.text_input("Paste a URL", placeholder="https://www.amazon.com/")

    if st.button("Analyze", type="primary"):
        if not url_in:
            st.warning("Please paste a URL first.")
        else:
            cleaned = clean_url(url_in)
            host = extract_host(cleaned)
            base_dom = base_domain_from_host(host)

            # check allowlist first
            hit, which = is_allowlisted_host(host, merged_allow)
            if hit:
                score = 0.0
                decision = "LEGITIMATE"
                st.success(f"LEGITIMATE ✅ | Score: {score:.2f}")
            else:
                try:
                    base_score = score_one_url(cleaned)
                    bump, flags, force_phish, sld_raw, sld_deleet, sld_norm, dist_hit, closest_tok, dist_val = \
                        rule_overlay(cleaned, host, base_dom)
                    score = float(min(1.0, max(0.0, base_score + bump)))

                    if force_phish:
                        decision = "PHISHING"
                        st.error(f"PHISHING 🚨 | Score: {score:.2f}")
                    else:
                        if score >= threshold:
                            decision = "PHISHING"
                            st.error(f"PHISHING 🚨 | Score: {score:.2f}")
                        else:
                            decision = "LEGITIMATE"
                            st.success(f"LEGITIMATE ✅ | Score: {score:.2f}")
                except Exception:
                    decision = "ERROR"
                    score = -1.0
                    st.error("Prediction error. Please try another URL.")

            # keep logging
            log_history({
                "ts": datetime.utcnow().isoformat(),
                "url": cleaned,
                "host": host,
                "base_domain": base_dom,
                "decision": decision,
                "score": round(float(score), 4),
                "threshold": round(float(threshold), 2),
            })


# ───────── Dashboard
elif page == "Dashboard":
    st.title("📊 Dashboard — Last 10 Scans")
    items = read_last_history(10)
    if not items:
        st.info("No scans yet. Go to Scan and try a URL.")
    else:
        for it in items:
            color = "#fecaca" if it["decision"] == "PHISHING" else "#bbf7d0"
            st.markdown(
                f"<div style='background:{color};border-radius:14px;padding:12px;border:1px solid #e5e7eb'>"
                f"<b>{it['decision']}</b> — {it['url']}<br>"
                f"<span style='color:#64748b;font-size:12px'>Score {it['score']:.2f} • {it['ts']}</span>"
                f"</div>", unsafe_allow_html=True
            )

# ───────── Tips
elif page == "Tips":
    st.title("🎓 Phishing Awareness")
    st.write("- Always check the real base domain (e.g., **amaz0n.com ≠ amazon.com**).")
    st.write("- HTTPS alone does not guarantee safety.")
    st.write("- Beware of urgency, misspellings, and link shorteners.")
    st.write("- Avoid logging in from email links; open the site directly.")

# ───────── Feedback
elif page == "Feedback":
    st.title("💬 Feedback")
    url = st.text_input("URL you checked")
    decision_seen = st.selectbox("What did the app say?", ["LEGITIMATE","PHISHING"])
    your_label = st.selectbox("What do YOU think it is?", ["legitimate (safe)","phishing (unsafe)"])
    notes = st.text_area("Notes (optional)")
    suggest = st.checkbox("Suggest adding this domain to the safe list")

    if st.button("Submit"):
        cleaned = clean_url(url)
        host = extract_host(cleaned)
        base_dom = base_domain_from_host(host)
        log_feedback({
            "ts": datetime.utcnow().isoformat(),
            "url": cleaned,
            "host": host,
            "decision": decision_seen.lower(),
            "score": "",
            "threshold": "",
            "user_label": "legitimate" if "legitimate" in your_label else "phishing",
            "notes": notes,
        })
        if suggest and base_dom:
            push_suggestion(base_dom)
        st.success("Thanks! Your feedback was recorded.")

# ───────── Admin
elif page == "Admin":
    st.title("🔐 Admin")
    if not ADMIN_PIN:
        st.info("Set ADMIN_PIN env var to enable Admin page.")
    else:
        pin = st.text_input("Enter PIN", type="password", placeholder="•••••")
        if pin != ADMIN_PIN:
            if pin:
                st.error("Wrong PIN.")
        else:
            st.success("Admin mode on.")
            st.subheader("Review Suggestions")
            sugg = load_suggestions()
            st.write("Pending:", sugg if sugg else "—")

            approve = st.multiselect("Approve", sugg)
            if st.button("Approve selected"):
                ua = load_user_allowlist()
                ua.update(approve)
                save_user_allowlist(ua)
                remaining = [d for d in sugg if d not in set(approve)]
                set_suggestions(remaining)
                st.success(f"Approved: {', '.join(approve) if approve else '—'}")

            decline = st.multiselect("Remove from pending", sugg, key="decline_box")
            if st.button("Remove selected"):
                remaining = [d for d in sugg if d not in set(decline)]
                set_suggestions(remaining)
                st.success(f"Removed: {', '.join(decline) if decline else '—'}")

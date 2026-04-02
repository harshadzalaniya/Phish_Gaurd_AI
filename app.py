import streamlit as st
import joblib
import pandas as pd
import re
import tldextract
import whois
import requests
import socket
from datetime import datetime
from rapidfuzz import fuzz
from fpdf import FPDF

# Safe TensorFlow import
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False

st.set_page_config(page_title="PhishGuard AI", page_icon="🔒", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>🔒 PhishGuard AI</h1>
    <p style='text-align: center; font-size: 1.1rem; color: #AAAAAA;'>
        Advanced Multi-Layer Phishing Detector • URL + Email + SMS + Typo Squatting
    </p>
    <p style='text-align: center; font-size: 0.95rem; color: #666666;'>
        College Internship Project by Harshad | Gujarat
    </p>
    <hr>
""", unsafe_allow_html=True)

# ====================== LOAD MODELS (Silent) ======================
@st.cache_resource
def load_models():
    text_model = None
    try:
        text_model = load_model("phishguard_text_model.h5", compile=False)
    except:
        pass
    url_model = joblib.load("phishguard_url_model.pkl")
    return url_model, text_model

url_model, text_model = load_models()

# ====================== HELPER FUNCTIONS ======================
def extract_url_features(url):
    features = {}
    ext = tldextract.extract(url)
    features['url_length'] = len(url)
    features['domain_length'] = len(ext.domain)
    features['tld_length'] = len(ext.suffix)
    features['has_ip'] = 1 if re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', url) else 0
    features['has_at'] = 1 if '@' in url else 0
    features['has_double_slash'] = 1 if '//' in url[8:] else 0
    features['has_hyphen'] = 1 if '-' in ext.domain else 0
    features['has_https'] = 1 if url.startswith('https') else 0
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_special_chars'] = len(re.findall(r'[@&%#?=/]', url))
    features['subdomain_count'] = len(ext.subdomain.split('.')) if ext.subdomain else 0
    try:
        w = whois.whois(ext.registered_domain)
        creation = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
        features['domain_age_days'] = (datetime.now() - creation).days if creation else -1
    except:
        features['domain_age_days'] = -1
    return pd.DataFrame([features])

def get_aligned_features(url):
    df = extract_url_features(url)
    if hasattr(url_model, 'feature_names_in_'):
        expected = url_model.feature_names_in_
        for col in expected:
            if col not in df.columns:
                df[col] = 0
        df = df[expected]
    return df

POPULAR_DOMAINS = ["google", "amazon", "microsoft", "apple", "paypal", "netflix", "facebook", "instagram",
                   "hdfcbank", "sbi", "icici", "axisbank", "bankofbaroda"]

def detect_typosquatting(domain):
    domain = domain.lower().split('.')[0]
    for legit in POPULAR_DOMAINS:
        score = fuzz.ratio(domain, legit)
        if score >= 82 and domain != legit:
            return True, legit, score
    return False, None, 0

# Real-time Internet Verification (Stronger)
def internet_url_check(url):
    try:
        # Check domain resolution
        domain = tldextract.extract(url).registered_domain
        socket.gethostbyname(domain)
        # Check if site responds
        response = requests.head(url, timeout=6, allow_redirects=True)
        return True, response.status_code
    except:
        return False, None

# Stricter Text Prediction with Explanation
@st.cache_resource
def get_tokenizer():
    return Tokenizer(num_words=10000)

tokenizer = get_tokenizer() if TF_AVAILABLE else None

def predict_text(text):
    if not TF_AVAILABLE or text_model is None or tokenizer is None:
        return 0.5, []
    
    text_lower = text.lower()
    phishing_keywords = ["won", "winner", "reward", "prize", "claim", "congratulations", "bank account",
                         "account details", "verify now", "urgent", "immediately", "limited time", "click here",
                         "password", "suspicious activity", "security alert", "you won", "reward claim"]
    
    matched = [word for word in phishing_keywords if word in text_lower]
    model_prob = text_model.predict(tokenizer.texts_to_sequences([text]), verbose=0)[0][0]
    final_prob = max(model_prob, 0.75 if len(matched) >= 2 else 0.5)
    return float(final_prob), matched

# ====================== TABS ======================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌐 URL Scanner", "✉️ Email Scanner", "📱 SMS Scanner", "🔍 Typo Squatting", "🔗 Hybrid Analyzer"])

# ------------------- URL Scanner -------------------
with tab1:
    st.subheader("URL Phishing Detection")
    url = st.text_input("Enter URL to check", placeholder="http://amaz0n.com")
    if st.button("🔍 Scan URL", type="primary"):
        if url:
            with st.spinner("Deep real-time analysis..."):
                is_typo, legit, score = detect_typosquatting(url)
                internet_ok, status = internet_url_check(url)
                features = get_aligned_features(url)
                prob = url_model.predict_proba(features)[0][1]
                
                if is_typo:
                    prob = max(prob, 0.95)
                if not internet_ok:
                    prob = max(prob, 0.85)
                
                risk = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
                st.markdown(f"**Result:** {'🔴 High Risk' if risk == 'High' else '🟠 Medium Risk' if risk == 'Medium' else '🟢 Low Risk'} (Confidence: {prob*100:.1f}%)")
                
                st.subheader("Why is this Phishing?")
                reasons = []
                if is_typo: reasons.append(f"🚨 Typo Squatting detected (looks like {legit})")
                if not internet_ok: reasons.append("⚠️ Domain does not exist on internet")
                if features['has_https'].iloc[0] == 0: reasons.append("🔒 No HTTPS (insecure)")
                if features['domain_age_days'].iloc[0] != -1 and features['domain_age_days'].iloc[0] < 30: reasons.append("🕒 Very new domain")
                if reasons:
                    for r in reasons:
                        st.write(r)
                else:
                    st.write("No major red flags detected.")

# ------------------- Email Scanner -------------------
with tab2:
    st.subheader("Email Phishing Detection")
    email_text = st.text_area("Paste Email Content", height=200)
    if st.button("Analyze Email"):
        if email_text.strip():
            with st.spinner("Analyzing..."):
                prob, matched = predict_text(email_text)
                st.markdown(f"**Result:** {'🔴 High Risk' if prob > 0.7 else '🟠 Medium Risk' if prob > 0.4 else '🟢 Low Risk'} (Confidence: {prob*100:.1f}%)")
                st.subheader("Why is this Phishing?")
                if matched:
                    st.write("Matched phishing keywords:", ", ".join(matched))
                else:
                    st.write("No strong phishing indicators found.")

# ------------------- SMS Scanner -------------------
with tab3:
    st.subheader("SMS / Smishing Detection")
    sms_text = st.text_area("Paste SMS Message", height=150)
    if st.button("Analyze SMS"):
        if sms_text.strip():
            with st.spinner("Analyzing..."):
                prob, matched = predict_text(sms_text)
                st.markdown(f"**Result:** {'🔴 High Risk' if prob > 0.7 else '🟠 Medium Risk' if prob > 0.4 else '🟢 Low Risk'} (Confidence: {prob*100:.1f}%)")
                st.subheader("Why is this Phishing?")
                if matched:
                    st.write("Matched phishing keywords:", ", ".join(matched))
                else:
                    st.write("No strong phishing indicators found.")

# ------------------- Typo Squatting -------------------
with tab4:
    st.subheader("Typo Squatting Checker")
    domain = st.text_input("Enter domain to check (e.g. go0gle.com)")
    if st.button("Check Typo Squatting"):
        if domain:
            is_typo, legit, score = detect_typosquatting(domain)
            if is_typo:
                st.error(f"🚨 TYPO SQUATTING DETECTED — Looks like **{legit}** ({score}% similar)")
                st.subheader("Why is this Phishing?")
                st.write(f"Domain is deliberately misspelled to trick users into visiting a fake site.")
            else:
                st.success("✅ No typo squatting detected")

# ------------------- Hybrid Analyzer -------------------
with tab5:
    st.subheader("Hybrid Analyzer")
    hybrid = st.text_area("Paste Email or SMS (can contain links)", height=200)
    if st.button("Run Hybrid Analysis"):
        if hybrid.strip():
            with st.spinner("Full multi-layer analysis..."):
                prob_text, matched = predict_text(hybrid)
                urls = re.findall(r'https?://\S+', hybrid)
                max_prob = prob_text
                for u in urls:
                    f = get_aligned_features(u)
                    p = url_model.predict_proba(f)[0][1]
                    if p > max_prob:
                        max_prob = p
                st.markdown(f"**Final Risk:** {'🔴 High Risk' if max_prob > 0.7 else '🟠 Medium Risk' if max_prob > 0.4 else '🟢 Low Risk'} (Max Confidence: {max_prob*100:.1f}%)")
                st.subheader("Why is this Phishing?")
                if matched:
                    st.write("Text contains:", ", ".join(matched))
                if urls:
                    st.write(f"Found {len(urls)} URL(s) — checked individually.")

st.caption("Most accurate version with real-time internet verification • Why Phishing? explanations added")

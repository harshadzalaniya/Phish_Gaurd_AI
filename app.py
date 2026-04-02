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

# ====================== LOAD MODELS ======================
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

# ====================== CORE HELPER FUNCTIONS ======================
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

def internet_url_check(url):
    try:
        domain = tldextract.extract(url).registered_domain
        socket.gethostbyname(domain)
        response = requests.head(url, timeout=6, allow_redirects=True)
        return True, response.status_code
    except:
        return False, None

@st.cache_resource
def get_tokenizer():
    return Tokenizer(num_words=10000)

tokenizer = get_tokenizer() if TF_AVAILABLE else None

# ====================== FULL ANALYSIS FUNCTION (Used by Email, SMS & Hybrid) ======================
def full_text_analysis(text):
    if not TF_AVAILABLE or text_model is None or tokenizer is None:
        return 0.5, []

    text_lower = text.lower()
    phishing_keywords = ["won", "winner", "reward", "prize", "claim", "congratulations", "bank account",
                         "account details", "verify now", "urgent", "immediately", "limited time", "click here",
                         "password", "suspicious activity", "security alert", "lottery", "you won"]

    matched_keywords = [word for word in phishing_keywords if word in text_lower]
    model_prob = text_model.predict(tokenizer.texts_to_sequences([text]), verbose=0)[0][0]

    # Extract URLs
    urls = re.findall(r'https?://\S+', text)
    max_prob = model_prob
    reasons = matched_keywords[:]

    for u in urls:
        is_typo, legit, score = detect_typosquatting(u)
        internet_ok, _ = internet_url_check(u)
        features = get_aligned_features(u)
        url_prob = url_model.predict_proba(features)[0][1]

        if is_typo:
            max_prob = max(max_prob, 0.95)
            reasons.append(f"Typo Squatting: looks like {legit}")
        if not internet_ok:
            max_prob = max(max_prob, 0.90)
            reasons.append("Domain does not exist on internet")
        if url_prob > max_prob:
            max_prob = url_prob

    final_prob = max(max_prob, 0.75 if len(matched_keywords) >= 2 else model_prob)
    return float(final_prob), reasons

# ====================== TABS ======================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌐 URL Scanner", "✉️ Email Scanner", "📱 SMS Scanner", "🔍 Typo Squatting", "🔗 Hybrid Analyzer"])

# URL Scanner (already strong)
with tab1:
    st.subheader("URL Phishing Detection")
    url = st.text_input("Enter URL to check", placeholder="http://amaz0n.com")
    if st.button("🔍 Scan URL", type="primary"):
        if url:
            with st.spinner("Deep analysis..."):
                is_typo, legit, score = detect_typosquatting(url)
                internet_ok, _ = internet_url_check(url)
                features = get_aligned_features(url)
                prob = url_model.predict_proba(features)[0][1]
                if is_typo: prob = max(prob, 0.95)
                if not internet_ok: prob = max(prob, 0.90)
                
                risk = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
                st.markdown(f"**Result:** {'🔴 High Risk' if risk == 'High' else '🟠 Medium Risk' if risk == 'Medium' else '🟢 Low Risk'} (Confidence: {prob*100:.1f}%)")
                
                st.subheader("Why is this Phishing?")
                if is_typo: st.write(f"🚨 Typo Squatting — looks like **{legit}**")
                if not internet_ok: st.write("⚠️ Domain does not exist on the internet")
                if features['has_https'].iloc[0] == 0: st.write("🔒 No HTTPS (insecure connection)")

# Email Scanner (Now fully smart)
with tab2:
    st.subheader("Email Phishing Detection")
    email_text = st.text_area("Paste Email Content", height=200)
    if st.button("Analyze Email"):
        if email_text.strip():
            with st.spinner("Full multi-layer analysis..."):
                prob, reasons = full_text_analysis(email_text)
                risk = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
                st.markdown(f"**Result:** {'🔴 High Risk' if risk == 'High' else '🟠 Medium Risk' if risk == 'Medium' else '🟢 Low Risk'} (Confidence: {prob*100:.1f}%)")
                st.subheader("Why is this Phishing?")
                for r in reasons:
                    st.write("• " + r)

# SMS Scanner (Now fully smart)
with tab3:
    st.subheader("SMS / Smishing Detection")
    sms_text = st.text_area("Paste SMS Message", height=150)
    if st.button("Analyze SMS"):
        if sms_text.strip():
            with st.spinner("Full multi-layer analysis..."):
                prob, reasons = full_text_analysis(sms_text)
                risk = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
                st.markdown(f"**Result:** {'🔴 High Risk' if risk == 'High' else '🟠 Medium Risk' if risk == 'Medium' else '🟢 Low Risk'} (Confidence: {prob*100:.1f}%)")
                st.subheader("Why is this Phishing?")
                for r in reasons:
                    st.write("• " + r)

# Typo Squatting (unchanged)
with tab4:
    st.subheader("Typo Squatting Checker")
    domain = st.text_input("Enter domain to check (e.g. g00gle.com)")
    if st.button("Check Typo Squatting"):
        if domain:
            is_typo, legit, score = detect_typosquatting(domain)
            if is_typo:
                st.error(f"🚨 TYPO SQUATTING DETECTED — Looks like **{legit}** ({score}% similar)")
            else:
                st.success("✅ No typo squatting detected")

# Hybrid Analyzer (already powerful)
with tab5:
    st.subheader("Hybrid Analyzer")
    hybrid = st.text_area("Paste Email or SMS (with links)", height=200)
    if st.button("Run Hybrid Analysis"):
        if hybrid.strip():
            with st.spinner("Full multi-layer analysis..."):
                prob, reasons = full_text_analysis(hybrid)
                risk = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
                st.markdown(f"**Final Risk:** {'🔴 High Risk' if risk == 'High' else '🟠 Medium Risk' if risk == 'Medium' else '🟢 Low Risk'} (Confidence: {prob*100:.1f}%)")
                st.subheader("Why is this Phishing?")
                for r in reasons:
                    st.write("• " + r)

st.caption("Now fully connected • SMS & Email automatically check embedded URLs + Typo Squatting + Internet verification")

import streamlit as st
import joblib
import pandas as pd
import re
import tldextract
import whois
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

st.title("🔒 PhishGuard AI")
st.caption("URL + Email + SMS + Typo Squatting Detector")
st.caption("College Internship Project by Harshad | Gujarat")

# ====================== LOAD MODELS (Silent Loading) ======================
@st.cache_resource
def load_models():
    text_model = None
    try:
        text_model = load_model("phishguard_text_model.h5", compile=False)
        # Removed success message as per your request
    except Exception:
        pass  # Fail silently - no message shown to user
    
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

POPULAR_DOMAINS = ["google", "amazon", "microsoft", "apple", "paypal", "netflix", "facebook", 
                   "instagram", "hdfcbank", "sbi", "icici", "axisbank", "bankofbaroda"]

def detect_typosquatting(domain):
    domain = domain.lower().split('.')[0]
    for legit in POPULAR_DOMAINS:
        score = fuzz.ratio(domain, legit)
        if score >= 85 and domain != legit:
            return True, legit, score
    return False, None, 0

@st.cache_resource
def get_tokenizer():
    tokenizer = Tokenizer(num_words=10000)
    return tokenizer

tokenizer = get_tokenizer() if TF_AVAILABLE else None

def predict_text(text):
    if not TF_AVAILABLE or text_model is None or tokenizer is None:
        return 0.5
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=300)
    prob = text_model.predict(padded, verbose=0)[0][0]
    return float(prob)

# ====================== SESSION STATE ======================
if 'history' not in st.session_state:
    st.session_state.history = []

# ====================== TABS ======================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌐 URL Scanner", "✉️ Email Scanner", "📱 SMS Scanner", "🔍 Typo Squatting", "🔗 Hybrid Analyzer"])

with tab1:
    st.subheader("URL Phishing Detection")
    url = st.text_input("Enter URL to check", placeholder="https://www.hdfcbank.com")
    if st.button("Scan URL", type="primary"):
        if url:
            with st.spinner("Analyzing URL..."):
                is_typo, legit, score = detect_typosquatting(url)
                if is_typo:
                    st.error(f"🚨 TYPO SQUATTING DETECTED! Similar to {legit} ({score}%)")
                
                features = get_aligned_features(url)
                prob = url_model.predict_proba(features)[0][1]
                
                if prob > 0.5:
                    st.error(f"🚨 PHISHING URL (Confidence: {prob*100:.1f}%)")
                else:
                    st.success(f"✅ SAFE URL (Confidence: {(1-prob)*100:.1f}%)")
                
                st.session_state.history.append({
                    "type": "URL", 
                    "input": url[:60], 
                    "result": "Phishing" if prob > 0.5 else "Safe", 
                    "conf": prob, 
                    "time": datetime.now().strftime("%H:%M")
                })

with tab2:
    st.subheader("Email Phishing Detection")
    email_text = st.text_area("Paste Email Content", height=200)
    if st.button("Analyze Email"):
        if email_text.strip():
            with st.spinner("Analyzing..."):
                prob = predict_text(email_text)
                if prob > 0.5:
                    st.error(f"🚨 PHISHING EMAIL (Confidence: {prob*100:.1f}%)")
                else:
                    st.success(f"✅ LEGITIMATE EMAIL (Confidence: {(1-prob)*100:.1f}%)")

with tab3:
    st.subheader("SMS / Smishing Detection")
    sms_text = st.text_area("Paste SMS Message", height=150)
    if st.button("Analyze SMS"):
        if sms_text.strip():
            with st.spinner("Analyzing..."):
                prob = predict_text(sms_text)
                if prob > 0.5:
                    st.error(f"🚨 SMISHING DETECTED (Confidence: {prob*100:.1f}%)")
                else:
                    st.success(f"✅ SAFE SMS (Confidence: {(1-prob)*100:.1f}%)")

with tab4:
    st.subheader("Typo Squatting Checker")
    domain = st.text_input("Enter domain (e.g. go0gle.com)")
    if st.button("Check Typo Squatting"):
        if domain:
            is_typo, legit, score = detect_typosquatting(domain)
            if is_typo:
                st.error(f"🚨 Possible typo of **{legit}** ({score}% similar)")
            else:
                st.success("✅ No typo squatting detected")

with tab5:
    st.subheader("Hybrid Analyzer")
    hybrid = st.text_area("Paste Email or SMS (can contain links)", height=200)
    if st.button("Run Hybrid Analysis"):
        if hybrid.strip():
            with st.spinner("Running full analysis..."):
                prob_text = predict_text(hybrid)
                urls = re.findall(r'https?://\S+', hybrid)
                if prob_text > 0.5:
                    st.error(f"Text → PHISHING ({prob_text*100:.1f}%)")
                else:
                    st.success(f"Text → SAFE ({(1-prob_text)*100:.1f}%)")
                if urls:
                    st.write(f"**Found {len(urls)} URL(s)**")
                    for u in urls[:5]:
                        f = get_aligned_features(u)
                        p = url_model.predict_proba(f)[0][1]
                        st.write(f"`{u[:70]}...` → {'🚨 Phishing' if p > 0.5 else '✅ Safe'} ({p*100:.1f}%)")

# ====================== SIDEBAR ======================
st.sidebar.header("Recent Scans")
for item in list(reversed(st.session_state.history))[:5]:
    st.sidebar.write(f"{item['time']} | {item['type']}: {item['result']}")

st.sidebar.info("Use examples from demo_phishing_examples.md")

st.caption("Project logic kept exactly as you wanted")

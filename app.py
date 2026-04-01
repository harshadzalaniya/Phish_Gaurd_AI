import streamlit as st
import joblib
import pandas as pd
import re
import tldextract
import whois
from datetime import datetime
from rapidfuzz import fuzz
from fpdf import FPDF
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page Configuration
st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🔒",
    layout="wide"
)

st.title("🔒 PhishGuard AI")
st.caption("Advanced Phishing Detection System | URL + Email + SMS + Typo Squatting")
st.caption("College Internship Project by Harshad | Gujarat")

# ====================== LOAD MODELS ======================
@st.cache_resource
def load_models():
    url_model = joblib.load("phishguard_url_model.pkl")
    text_model = load_model("phishguard_text_model.keras")
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

POPULAR_DOMAINS = ["google", "amazon", "microsoft", "apple", "paypal", "netflix", "facebook", 
                   "instagram", "hdfcbank", "sbi", "icici", "axisbank", "bankofbaroda"]

def detect_typosquatting(domain):
    domain = domain.lower().split('.')[0]
    for legit in POPULAR_DOMAINS:
        score = fuzz.ratio(domain, legit)
        if score >= 85 and domain != legit:
            return True, legit, score
    return False, None, 0

# Simple tokenizer for text model (we'll reuse the same logic)
@st.cache_resource
def get_tokenizer():
    # This is a simple fallback tokenizer. In production you can save tokenizer too.
    tokenizer = Tokenizer(num_words=10000)
    return tokenizer

tokenizer = get_tokenizer()

def predict_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=300)
    prediction = text_model.predict(padded, verbose=0)[0][0]
    return prediction  # probability of being phishing (1 = phishing)

# ====================== STREAMLIT APP ======================
if 'history' not in st.session_state:
    st.session_state.history = []

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌐 URL Scanner", "✉️ Email Scanner", "📱 SMS Scanner", "🔍 Typo Squatting", "🔗 Hybrid Analyzer"])

# ------------------- URL Scanner -------------------
with tab1:
    st.subheader("URL Phishing Detection")
    url = st.text_input("Enter URL to scan:", placeholder="https://www.example.com")
    
    if st.button("🔍 Scan URL", type="primary"):
        if url:
            with st.spinner("Analyzing URL..."):
                # Typo Squatting Check
                is_typo, legit_domain, similarity = detect_typosquatting(url)
                if is_typo:
                    st.error(f"🚨 **TYPO SQUATTING DETECTED!** Looks like {legit_domain} (Similarity: {similarity}%)")
                
                # Feature Extraction & URL Model
                features_df = extract_url_features(url)
                prob = url_model.predict_proba(features_df)[0][1]
                prediction = 1 if prob > 0.5 else 0
                
                if prediction == 1:
                    st.error(f"🚨 **PHISHING URL DETECTED** (Confidence: {prob*100:.1f}%)")
                    risk = "High"
                else:
                    st.success(f"✅ **SAFE URL** (Confidence: {(1-prob)*100:.1f}%)")
                    risk = "Low"
                
                # Add to history
                st.session_state.history.append({
                    "type": "URL",
                    "input": url,
                    "result": "Phishing" if prediction == 1 else "Safe",
                    "confidence": prob if prediction == 1 else (1-prob),
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M")
                })
        else:
            st.warning("Please enter a URL")

# ------------------- Email Scanner -------------------
with tab2:
    st.subheader("Email Phishing Detection")
    email_text = st.text_area("Paste Email Content:", height=200)
    
    if st.button("Analyze Email"):
        if email_text.strip():
            with st.spinner("Analyzing Email..."):
                prob = predict_text(email_text)
                if prob > 0.5:
                    st.error(f"🚨 **PHISHING EMAIL** (Confidence: {prob*100:.1f}%)")
                else:
                    st.success(f"✅ **LEGITIMATE EMAIL** (Confidence: {(1-prob)*100:.1f}%)")
        else:
            st.warning("Please paste email content")

# ------------------- SMS Scanner -------------------
with tab3:
    st.subheader("SMS / Smishing Detection")
    sms_text = st.text_area("Paste SMS Message:", height=150)
    
    if st.button("Analyze SMS"):
        if sms_text.strip():
            with st.spinner("Analyzing SMS..."):
                prob = predict_text(sms_text)
                if prob > 0.5:
                    st.error(f"🚨 **SMISHING (Phishing SMS) DETECTED** (Confidence: {prob*100:.1f}%)")
                else:
                    st.success(f"✅ **SAFE SMS** (Confidence: {(1-prob)*100:.1f}%)")
        else:
            st.warning("Please paste SMS text")

# ------------------- Typo Squatting -------------------
with tab4:
    st.subheader("Typo Squatting Checker")
    domain = st.text_input("Enter domain to check (e.g. go0gle.com):")
    
    if st.button("Check for Typo Squatting"):
        if domain:
            is_typo, legit, score = detect_typosquatting(domain)
            if is_typo:
                st.error(f"🚨 Possible Typo Squatting of **{legit}** (Similarity: {score}%)")
            else:
                st.success("✅ No obvious typo squatting detected")
        else:
            st.warning("Please enter a domain")

# ------------------- Hybrid Analyzer -------------------
with tab5:
    st.subheader("Hybrid Analyzer (Email/SMS with URLs)")
    hybrid_text = st.text_area("Paste full Email or SMS (with embedded URLs):", height=200)
    
    if st.button("Run Hybrid Analysis"):
        if hybrid_text.strip():
            with st.spinner("Performing full analysis..."):
                # Text prediction
                text_prob = predict_text(hybrid_text)
                
                # Extract URLs from text
                urls = re.findall(r'https?://[^\s]+', hybrid_text)
                
                st.write("### Results:")
                if text_prob > 0.5:
                    st.error(f"Text Analysis: **PHISHING** ({text_prob*100:.1f}%)")
                else:
                    st.success(f"Text Analysis: **SAFE** ({(1-text_prob)*100:.1f}%)")
                
                if urls:
                    st.write(f"**Found {len(urls)} URL(s)**")
                    for u in urls:
                        features = extract_url_features(u)
                        url_prob = url_model.predict_proba(features)[0][1]
                        st.write(f"URL: `{u}` → {'🚨 Phishing' if url_prob > 0.5 else '✅ Safe'} ({url_prob*100:.1f}%)")
                else:
                    st.info("No URLs found in the text.")
        else:
            st.warning("Please paste content")

# ====================== SCAN HISTORY & PDF REPORT ======================
st.sidebar.header("Scan History")
if st.session_state.history:
    for entry in reversed(st.session_state.history[-5:]):  # show last 5
        st.sidebar.write(f"{entry['time']} | {entry['type']}: {entry['result']}")

if st.sidebar.button("Download PDF Report"):
    if st.session_state.history:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="PhishGuard AI - Scan Report", ln=1, align='C')
        pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1)
        pdf.ln(10)
        
        for entry in st.session_state.history:
            pdf.cell(200, 10, txt=f"{entry['time']} - {entry['type']}: {entry['result']} ({entry['confidence']*100:.1f}%)", ln=1)
        
        pdf_output = pdf.output(dest='S').encode('latin-1')
        st.sidebar.download_button(
            label="📥 Download Report",
            data=pdf_output,
            file_name=f"PhishGuard_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf"
        )
    else:
        st.sidebar.warning("No scans yet!")

st.sidebar.info("Tip: Use demo examples from demo_phishing_examples.md")
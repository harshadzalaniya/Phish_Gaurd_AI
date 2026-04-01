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

# Try to import TensorFlow/Keras safely
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.error("TensorFlow not installed properly. Check requirements.txt")

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🔒",
    layout="wide"
)

st.title("🔒 PhishGuard AI")
st.caption("Advanced Phishing Detection | URL + Email + SMS + Typo Squatting")
st.caption("College Internship Project by Harshad | Gujarat, India")

# ====================== LOAD MODELS ======================
@st.cache_resource
def load_models():
    try:
        # Prefer .h5 for better compatibility on Streamlit Cloud
        text_model = load_model("phishguard_text_model.h5", compile=False)
        st.success("✅ Text (Email/SMS) model loaded successfully")
    except Exception as e:
        try:
            text_model = load_model("phishguard_text_model.keras", compile=False)
            st.warning("Loaded from .keras (fallback)")
        except Exception as e2:
            st.error(f"Model loading failed: {str(e2)}")
            text_model = None
    
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

POPULAR_DOMAINS = ["google", "amazon", "microsoft", "apple", "paypal", "netflix", "facebook", 
                   "instagram", "hdfcbank", "sbi", "icici", "axisbank", "bankofbaroda"]

def detect_typosquatting(domain):
    domain = domain.lower().split('.')[0]
    for legit in POPULAR_DOMAINS:
        score = fuzz.ratio(domain, legit)
        if score >= 85 and domain != legit:
            return True, legit, score
    return False, None, 0

# Simple tokenizer for text prediction
@st.cache_resource
def get_tokenizer():
    tokenizer = Tokenizer(num_words=10000)
    return tokenizer

tokenizer = get_tokenizer()

def predict_text(text):
    if text_model is None or not TF_AVAILABLE:
        return 0.5  # fallback
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
    url = st.text_input("Enter URL:", placeholder="https://www.hdfcbank.com")
    if st.button("Scan URL", type="primary"):
        if url:
            with st.spinner("Analyzing..."):
                is_typo, legit, sim = detect_typosquatting(url)
                if is_typo:
                    st.error(f"🚨 TYPO SQUATTING: Similar to {legit} ({sim}%)")
                
                features = extract_url_features(url)
                prob = url_model.predict_proba(features)[0][1]
                result = "Phishing" if prob > 0.5 else "Safe"
                color = "error" if prob > 0.5 else "success"
                st.write(f"**Result:** :{color}[**{result}**] (Confidence: {prob*100:.1f}%)")
                
                st.session_state.history.append({"type": "URL", "input": url[:50], "result": result, "conf": prob, "time": datetime.now().strftime("%H:%M")})

with tab2:
    st.subheader("Email Phishing Detection")
    email = st.text_area("Paste Email Text:", height=180)
    if st.button("Analyze Email"):
        if email.strip():
            with st.spinner("Analyzing..."):
                prob = predict_text(email)
                result = "Phishing" if prob > 0.5 else "Legitimate"
                st.write(f"**Result:** :{'error' if prob > 0.5 else 'success'}[**{result}**] ({prob*100:.1f}%)")

with tab3:
    st.subheader("SMS / Smishing Detection")
    sms = st.text_area("Paste SMS Message:", height=150)
    if st.button("Analyze SMS"):
        if sms.strip():
            with st.spinner("Analyzing..."):
                prob = predict_text(sms)
                result = "Smishing" if prob > 0.5 else "Safe"
                st.write(f"**Result:** :{'error' if prob > 0.5 else 'success'}[**{result}**] ({prob*100:.1f}%)")

with tab4:
    st.subheader("Typo Squatting Checker")
    domain = st.text_input("Enter domain (e.g. go0gle.com):")
    if st.button("Check Typo"):
        if domain:
            is_typo, legit, score = detect_typosquatting(domain)
            if is_typo:
                st.error(f"🚨 Possible typo of **{legit}** ({score}% similar)")
            else:
                st.success("✅ No typo squatting detected")

with tab5:
    st.subheader("Hybrid Analyzer")
    text = st.text_area("Paste Email or SMS (with links):", height=200)
    if st.button("Run Full Analysis"):
        if text.strip():
            with st.spinner("Running hybrid scan..."):
                prob_text = predict_text(text)
                urls = re.findall(r'https?://\S+', text)
                st.write(f"**Text Analysis:** {'🚨 Phishing' if prob_text > 0.5 else '✅ Safe'} ({prob_text*100:.1f}%)")
                if urls:
                    st.write(f"**{len(urls)} URL(s) found**")
                    for u in urls[:3]:
                        f = extract_url_features(u)
                        p = url_model.predict_proba(f)[0][1]
                        st.write(f"• `{u[:60]}...` → {'🚨 Phishing' if p > 0.5 else '✅ Safe'} ({p*100:.1f}%)")

# ====================== SIDEBAR ======================
st.sidebar.header("Recent Scans")
for h in list(reversed(st.session_state.history))[:5]:
    st.sidebar.write(f"{h['time']} | {h['type']}: {h['result']}")

if st.sidebar.button("📥 Download PDF Report"):
    if st.session_state.history:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "PhishGuard AI Scan Report", ln=1, align='C')
        for h in st.session_state.history:
            pdf.cell(200, 10, f"{h['time']} - {h['type']}: {h['result']} ({h['conf']*100:.1f}%)", ln=1)
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        st.sidebar.download_button("Download Report", pdf_bytes, "PhishGuard_Report.pdf", "application/pdf")

st.sidebar.info("📌 Use demo_phishing_examples.md for testing")

# Footer
st.caption("Deployed on Streamlit Community Cloud • Minimal & Clean for Internship Submission")

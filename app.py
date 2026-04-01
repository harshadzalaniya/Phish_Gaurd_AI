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
    st.error("TensorFlow/Keras not installed. Check requirements.txt on Streamlit Cloud.")

st.set_page_config(page_title="PhishGuard AI", page_icon="🔒", layout="wide")

st.title("🔒 PhishGuard AI")
st.caption("Advanced Phishing Detection System | Internship Project by Harshad")

# ====================== LOAD MODELS ======================
@st.cache_resource
def load_models():
    text_model = None
    try:
        # Try .h5 first (recommended for Streamlit)
        if TF_AVAILABLE:
            text_model = load_model("phishguard_text_model.h5", compile=False)
            st.success("✅ Email/SMS model (.h5) loaded successfully")
    except Exception as e_h5:
        try:
            # Fallback to .keras
            if TF_AVAILABLE:
                text_model = load_model("phishguard_text_model.keras", compile=False)
                st.warning("✅ Email/SMS model (.keras) loaded")
        except Exception as e:
            st.error(f"❌ Model loading failed: {str(e)}")
            st.info("Tip: Make sure phishguard_text_model.h5 is in the root of your GitHub repo")
    
    url_model = joblib.load("phishguard_url_model.pkl")
    return url_model, text_model

url_model, text_model = load_models()

# ====================== HELPER FUNCTIONS (same as before) ======================
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

@st.cache_resource
def get_tokenizer():
    from tensorflow.keras.preprocessing.text import Tokenizer
    return Tokenizer(num_words=10000)

tokenizer = get_tokenizer() if TF_AVAILABLE else None

def predict_text(text):
    if not TF_AVAILABLE or text_model is None or tokenizer is None:
        return 0.5
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=300)
    prob = text_model.predict(padded, verbose=0)[0][0]
    return float(prob)

# ====================== TABS (shortened for brevity - add your full tabs here) ======================
# ... (You can keep the tabs from my previous full app.py)

st.info("If you see model loading error, ensure **phishguard_text_model.h5** is uploaded to GitHub root.")

# Add your tabs (URL, Email, SMS, Typo, Hybrid) here - use the structure from previous message

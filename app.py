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
        Advanced Multi-Layer Phishing Detector with Intelligent AI Reasoning
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
                   "hdfcbank", "sbi", "icici", "axisbank", "bankofbaroda", "paytm", "phonepe"]

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

# ====================== ENHANCED AI REASONING ======================
def generate_ai_explanation(text, prob, reasons, is_url=False, url_prob=None):
    explanation = []
    
    # Overall Assessment
    if prob > 0.85:
        explanation.append("**🔴 High Confidence Phishing Attempt** - This content shows multiple strong indicators of a sophisticated phishing attack.")
    elif prob > 0.65:
        explanation.append("**🟠 Moderate to High Risk** - The message contains several suspicious elements commonly used in phishing campaigns.")
    elif prob > 0.4:
        explanation.append("**🟡 Medium Risk** - Some red flags are present, but not definitive.")
    else:
        explanation.append("**🟢 Low Risk** - The content appears legitimate based on current analysis.")

    # Specific Analysis
    if reasons:
        explanation.append("\n**Detected Indicators:**")
        for r in reasons[:7]:
            explanation.append("• " + r)

    # Intelligent Contextual Reasoning
    text_lower = text.lower()
    
    if any(word in text_lower for word in ["lottery", "jackpot", "you won", "won the $", "prize claim"]):
        explanation.append("\n**AI Reasoning:** This is a classic 'reward scam' tactic. Attackers create excitement and urgency by promising large sums of money (e.g., lottery winnings) to trick victims into sharing personal or banking details.")

    if any(word in text_lower for word in ["bank account", "account details", "verify now", "enter the details"]):
        explanation.append("\n**AI Reasoning:** Requesting sensitive information like bank account details is a major red flag. Legitimate organizations never ask for such information via unsolicited messages.")

    if "urgent" in text_lower or "immediately" in text_lower or "limited time" in text_lower:
        explanation.append("\n**AI Reasoning:** Use of urgency words (urgent, immediately, limited time) is a common psychological manipulation technique used in phishing to prevent the victim from thinking critically.")

    if any(typ in text_lower for typ in ["g00gle", "amaz0n", "paypa1", "faceb00k"]):
        explanation.append("\n**AI Reasoning:** The message contains a deliberately misspelled popular domain (typo-squatting), which is a well-known phishing technique to impersonate trusted brands.")

    if not reasons and prob < 0.4:
        explanation.append("\n**AI Reasoning:** No significant phishing patterns, urgency tactics, or suspicious links were detected. The message appears to be legitimate.")

    return explanation

# ====================== FULL ANALYSIS FUNCTION ======================
def full_text_analysis(text):
    if not TF_AVAILABLE or text_model is None or tokenizer is None:
        return 0.5, []

    text_lower = text.lower()
    
    critical_keywords = ["lottery", "you won", "jackpot", "reward claim", "won the $", "prize claim"]
    high_keywords = ["won", "winner", "reward", "prize", "claim now", "urgent", "immediately", 
                     "bank account", "account details", "verify now", "suspended", "locked"]
    medium_keywords = ["congratulations", "free gift", "refund", "delivery failed", "pay now", 
                       "security alert", "unauthorized", "click here", "limited time", "otp"]

    reasons = []
    score_boost = 0.0

    for kw in critical_keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
            reasons.append(f"Critical: {kw}")
            score_boost = max(score_boost, 0.45)

    for kw in high_keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
            reasons.append(f"High Risk: {kw}")
            score_boost = max(score_boost, 0.30)

    for kw in medium_keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
            reasons.append(f"Medium: {kw}")
            score_boost = max(score_boost, 0.15)

    model_prob = text_model.predict(tokenizer.texts_to_sequences([text]), verbose=0)[0][0]

    urls = re.findall(r'https?://\S+', text)
    max_prob = max(model_prob, score_boost)

    for u in urls:
        is_typo, legit, score = detect_typosquatting(u)
        internet_ok, _ = internet_url_check(u)
        features = get_aligned_features(u)
        url_prob = url_model.predict_proba(features)[0][1]

        if is_typo:
            max_prob = max(max_prob, 0.96)
            reasons.append(f"Typo Squatting: looks like {legit}")
        if not internet_ok:
            max_prob = max(max_prob, 0.92)
            reasons.append("Domain does not exist on internet")
        if url_prob > max_prob:
            max_prob = url_prob

    final_prob = max(max_prob, model_prob + score_boost)
    return float(final_prob), reasons

# ====================== 3 TABS ONLY ======================
tab1, tab2, tab3 = st.tabs(["🌐 URL Scanning", "✉️ Email Scanner", "🔗 Hybrid Scanner"])

with tab1:
    st.subheader("URL Scanning")
    url = st.text_input("Enter URL to check", placeholder="http://amaz0n.com")
    if st.button("🔍 Scan URL", type="primary"):
        if url:
            with st.spinner("Analyzing with AI..."):
                is_typo, legit, score = detect_typosquatting(url)
                internet_ok, _ = internet_url_check(url)
                features = get_aligned_features(url)
                prob = url_model.predict_proba(features)[0][1]
                if is_typo: prob = max(prob, 0.95)
                if not internet_ok: prob = max(prob, 0.90)
                
                risk = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
                st.markdown(f"**Result:** {'🔴 High Risk' if risk == 'High' else '🟠 Medium Risk' if risk == 'Medium' else '🟢 Low Risk'} (Confidence: {prob*100:.1f}%)")
                
                st.subheader("Why is this Phishing?")
                reasons = []
                if is_typo: reasons.append(f"Typo Squatting — looks like **{legit}**")
                if not internet_ok: reasons.append("Domain does not exist on the internet")
                if 'has_https' in features.columns and features['has_https'].iloc[0] == 0:
                    reasons.append("No HTTPS (insecure connection)")
                
                for r in reasons:
                    st.write("• " + r)
                
                st.subheader("AI Reasoning")
                ai_reasons = generate_ai_explanation(url, prob, reasons, is_url=True, url_prob=prob)
                for line in ai_reasons:
                    st.write(line)

with tab2:
    st.subheader("Email Scanner")
    email_text = st.text_area("Paste Email Content", height=200)
    if st.button("Analyze Email"):
        if email_text.strip():
            with st.spinner("Analyzing with AI..."):
                prob, reasons = full_text_analysis(email_text)
                risk = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
                st.markdown(f"**Result:** {'🔴 High Risk' if risk == 'High' else '🟠 Medium Risk' if risk == 'Medium' else '🟢 Low Risk'} (Confidence: {prob*100:.1f}%)")
                
                st.subheader("Why is this Phishing?")
                for r in reasons:
                    st.write("• " + r)
                
                st.subheader("AI Reasoning")
                ai_reasons = generate_ai_explanation(email_text, prob, reasons)
                for line in ai_reasons:
                    st.write(line)

with tab3:
    st.subheader("Hybrid Scanner (All-in-One)")
    hybrid_text = st.text_area("Paste SMS / Email (can contain links)", height=220)
    if st.button("Run Full Hybrid Analysis"):
        if hybrid_text.strip():
            with st.spinner("Analyzing with AI..."):
                prob, reasons = full_text_analysis(hybrid_text)
                risk = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
                st.markdown(f"**Final Result:** {'🔴 High Risk' if risk == 'High' else '🟠 Medium Risk' if risk == 'Medium' else '🟢 Low Risk'} (Confidence: {prob*100:.1f}%)")
                
                st.subheader("Why is this Phishing?")
                for r in reasons:
                    st.write("• " + r)
                
                st.subheader("AI Reasoning")
                ai_reasons = generate_ai_explanation(hybrid_text, prob, reasons)
                for line in ai_reasons:
                    st.write(line)

st.caption("Enhanced AI Reasoning Engine • Hybrid Scanner analyzes everything together")

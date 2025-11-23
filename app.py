import streamlit as st
import joblib
import re

# --- 1. SETUP & LOAD AI MODEL ---
@st.cache_resource
def load_ai():
    try:
        # Pfade wurden HIER korrigiert, um im Ordner 'data/' zu suchen
        model = joblib.load('data/toxic_classifier_model.joblib')
        vectorizer = joblib.load('data/tfidf_vectorizer.joblib')
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_ai()

# --- 2. TEXT CLEANING FUNCTION ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub('http\S+|www.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 3. APP INTERFACE (UI) ---

st.title("🛡️ RespectGuard AI")
st.subheader("Detect harmful language in social media comments")
st.write("This tool analyzes your text for toxicity before you post it.")

if model is None:
    st.error("ERROR: Model files not found! Please check the path and file names.")
else:
    # Input Area
    user_input = st.text_area("Enter your comment here:", height=150, placeholder="Type something...")

    # Button
    if st.button("Analyze Comment"):
        if user_input:
            # Verarbeitung
            clean = clean_text(user_input)
            vec = vectorizer.transform([clean])
            prob = model.predict_proba(vec)[0][1] * 100
            
            # Anzeige
            st.write("---")
            st.progress(int(prob))
            st.write(f"Toxicity Score: **{prob:.1f}%**")

            # Entscheidung (Threshold 50%)
            if prob >= 50:
                st.error("🛑 BLOCKED: High toxicity detected.")
                st.warning("Please rephrase your message to be more respectful.")
            else:
                st.success("✅ APPROVED: This comment looks respectful.")
                st.balloons() 
        else:
            st.info("Please enter some text first.")

import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# ── Page config ──────────────────────────────────────
st.set_page_config(
    page_title="Language Detector",
    page_icon="🌍",
    layout="centered"
)

# ── Load and train model ─────────────────────────────
@st.cache_resource
def load_model():
    # ✅ FIXED PATH HERE
    df = pd.read_csv('data/raw/language_dataset_milestone1.csv')
    
    def preprocess(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    df['clean_text'] = df['text'].apply(preprocess)
    
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), max_features=5000)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['language']
    
    model = SVC(kernel='linear')
    model.fit(X, y)
    
    return model, vectorizer

model, vectorizer = load_model()

# ── UI ───────────────────────────────────────────────
st.title("🌍 Kenyan Language Detector")
st.subheader("CSC423 NLP Project — Maryivy Kibali")
st.markdown("---")

st.write("Enter a short text and the system will detect which language it is!")

text_input = st.text_area("✍️ Type your text here:", height=100)

if st.button("🔍 Detect Language"):
    if text_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Preprocess and predict
        clean = text_input.lower()
        clean = clean.translate(str.maketrans('', '', string.punctuation))
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        vec = vectorizer.transform([clean])
        prediction = model.predict(vec)[0]
        
        # Display result
        st.markdown("---")
        st.success(f"✅ Detected Language: **{prediction}**")
        
        # Color per language
        colors = {
            "Swahili": "🟢",
            "English": "🔵", 
            "Sheng":   "🟣",
            "Luo":     "🔴"
        }
        emoji = colors.get(prediction, "⚪")
        st.markdown(f"## {emoji} {prediction}")

st.markdown("---")
st.caption("Built with Python · Scikit-learn · Streamlit")

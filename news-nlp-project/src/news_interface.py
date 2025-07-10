import streamlit as st
import joblib
import re
import spacy
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from rapidfuzz import process
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

model = joblib.load("news_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
vocab = set(vectorizer.get_feature_names_out())

def nameEntityRec(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

def textCleaning(text, preserved_entities):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'\b\w\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [word for word in tokens if word.lower() not in stop_words]
    tokens += preserved_entities
    return ' '.join(tokens)

def textNormalizing(text):
    return ' '.join([lemmatizer.lemmatize(token.lower()) for token in text.split()])

def correctTypo(word, vocab):
    best_match = process.extractOne(word, vocab, score_cutoff=80)
    return best_match[0] if best_match else word

def typoCorrection(text, vocab, preserved_entities):
    tokens = text.split()
    corrected = [token if token in preserved_entities or token in vocab or len(token) <= 2 else correctTypo(token, vocab) for token in tokens]
    return ' '.join(corrected)


st.set_page_config(page_title="News Categorization", layout="wide")
st.title("🗞️ BBC News Article Categorizer")
st.markdown("This tool analyzes and predicts the category of a given news article using a trained ML pipeline.")

user_input = st.text_area("✍️ Paste your news article here:", height=300)

if st.button("🔍 Predict Category"):
    if not user_input.strip():
        st.warning("Please provide input text.")
    else:
        entities = nameEntityRec(user_input)
        cleaned = textCleaning(user_input, entities)
        normalized = textNormalizing(cleaned)
        corrected = typoCorrection(normalized, vocab, entities)
        X_input = vectorizer.transform([corrected])
        prediction = model.predict(X_input)[0]

        st.success(f"✅ Predicted Category: **{prediction}**")

        # Show top 3 prediction probabilities if possible
        clf = model.named_steps['clf'] if hasattr(model, 'named_steps') else model
        if hasattr(clf, "predict_proba"):
            st.subheader("📊 Class Probabilities")
            probs = clf.predict_proba(X_input)[0]
            top_indices = np.argsort(probs)[::-1][:3]
            for i in top_indices:
                st.write(f"{clf.classes_[i]}: {probs[i]*100:.2f}%")

        st.info(f"Model Used: **{type(clf).__name__}**")

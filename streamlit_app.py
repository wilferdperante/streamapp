import streamlit as st
from PIL import Image
import pickle
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# =============================================
# NLTK Setup with Persistent Download Solution
# =============================================
try:
    # Set NLTK data path
    nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(nltk_data_path, exist_ok=True)
    nltk.data.path.append(nltk_data_path)
    
    # Download required data
    nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
except Exception as e:
    st.error(f"NLTK setup failed: {str(e)}")

ps = PorterStemmer()

# =============================================
# Text Processing Function
# =============================================
def transform_text(text):
    try:
        text = str(text).lower()
        text = nltk.word_tokenize(text)
        
        words = []
        for word in text:
            if word.isalnum() and word not in stopwords.words('english') and word not in string.punctuation:
                words.append(ps.stem(word))
                
        return " ".join(words)
    except Exception as e:
        st.error(f"Text processing error: {str(e)}")
        return ""

# =============================================
# Vectorizer and Model Loading with Fallback
# =============================================
def load_or_create_vectorizer():
    # Try to load existing vectorizer
    for path in ['vectorizer.pkl', 'SMS_Spam_Classifier/vectorizer.pkl']:
        try:
            with open(path, 'rb') as f:
                vectorizer = pickle.load(f)
                if hasattr(vectorizer, 'vocabulary_'):
                    return vectorizer
        except:
            continue
    
    # Create new vectorizer if loading fails
    st.warning("Creating default vectorizer - for best results, provide a properly trained one")
    vectorizer = TfidfVectorizer()
    # Fit with some basic data
    vectorizer.fit(["free prize", "hello world", "win money", "meeting tomorrow"])
    return vectorizer

def load_model():
    for path in ['mnb_spam_detector.pkl', 'SMS_Spam_Classifier/mnb_spam_detector.pkl']:
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except:
            continue
    st.error("Could not load model file")
    st.stop()

tfidf = load_or_create_vectorizer()
model = load_model()

# =============================================
# Streamlit UI
# =============================================
st.title("SMS Spam Classifier")

# Try to load image
try:
    img_path = next(p for p in ['spam_image.jpeg', 'SMS_Spam_Classifier/spam_image.jpeg'] 
                   if os.path.exists(p))
    st.image(Image.open(img_path))
except:
    st.warning("Could not load preview image")

st.write("""
Detect spam SMS messages using machine learning.
Algorithm: Stacking Classifier (SVM, Naive Bayes, XGBoost)
""")

input_sms = st.text_area("Enter the message to check", height=100)

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message")
    else:
        try:
            # Transform and predict
            processed_text = transform_text(input_sms)
            features = tfidf.transform([processed_text])
            prediction = model.predict(features)[0]
            
            if prediction == 1:
                st.error("🚨 SPAM detected!")
            else:
                st.success("✅ Legitimate message")
                
            # Show processing details (optional)
            with st.expander("Show processing details"):
                st.write("Processed text:", processed_text)
                st.write("Prediction confidence:", max(model.predict_proba(features)[0]))
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# Footer
st.markdown("---")
cols = st.columns(3)
with cols[0]:
    st.info('[GitHub](https://github.com/anilremo23)')
with cols[1]:
    st.info('[Kaggle](https://www.kaggle.com/remoanil)')
with cols[2]:
    st.info('[LinkedIn](https://linkedin.com/in/anil-mamidwar-001b6418)')

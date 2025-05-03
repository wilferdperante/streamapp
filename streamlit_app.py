import streamlit as st
from PIL import Image
import pickle
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# =============================================
# NLTK Setup with Persistent Download Solution
# =============================================
try:
    # Try to find the punkt resource
    nltk.data.find('tokenizers/punkt')
except LookupError:
    # If not found, download with custom path
    nltk.download('punkt', download_dir='/tmp/nltk_data')
    nltk.data.path.append('/tmp/nltk_data')

try:
    # Try to find stopwords
    nltk.data.find('corpora/stopwords')
except LookupError:
    # If not found, download with custom path
    nltk.download('stopwords', download_dir='/tmp/nltk_data')
    nltk.data.path.append('/tmp/nltk_data')

# Initialize stemmer
ps = PorterStemmer()

# =============================================
# Text Processing with Fallback Tokenizer
# =============================================
def transform_text(text):
    try:
        text = str(text).lower()
        y = []
        
        # Attempt NLTK tokenization with fallback
        try:
            text = nltk.word_tokenize(text)
        except:
            # Simple whitespace tokenizer fallback
            text = text.split()
            
        for i in text:
            if i.isalnum():
                y.append(i)
        text = y[:]
        y.clear()
        
        # Get stopwords with fallback
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()  # Empty set if stopwords not available
            
        for i in text:
            if i not in stop_words and i not in string.punctuation:
                y.append(i)
        text = y[:]
        y.clear()
        
        # Stemming
        for i in text:
            y.append(ps.stem(i))
            
        return " ".join(y)
    except Exception as e:
        st.error(f"Error processing text: {str(e)}")
        return ""

# =============================================
# Model Loading with Verification
# =============================================
try:
    # Try loading from absolute path first
    try:
        tfidf = pickle.load(open('/mount/src/streamapp/SMS_Spam_Classifier/vectorizer.pkl', 'rb'))
        model = pickle.load(open('/mount/src/streamapp/SMS_Spam_Classifier/mnb_spam_detector.pkl', 'rb'))
    except:
        # Fallback to relative path
        tfidf = pickle.load(open('SMS_Spam_Classifier/vectorizer.pkl', 'rb'))
        model = pickle.load(open('SMS_Spam_Classifier/mnb_spam_detector.pkl', 'rb'))
        
    # Verify vectorizer is fitted
    if not hasattr(tfidf, 'vocabulary_'):
        raise ValueError("Vectorizer is not fitted properly")
except Exception as e:
    st.error(f"Model loading failed: {str(e)}")
    st.stop()

# =============================================
# Streamlit UI
# =============================================
st.title("SMS Spam Classifier")

# Image with fallback
try:
    st.image(Image.open('SMS_Spam_Classifier/spam_image.jpeg'))
except:
    st.warning("Could not load preview image")

st.write("""
This classifier detects spam SMS messages using machine learning.
""")

input_sms = st.text_area("Enter the message to check")

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message")
    else:
        try:
            # Transform and predict
            transformed = transform_text(input_sms)
            vectorized = tfidf.transform([transformed])
            prediction = model.predict(vectorized)[0]
            
            if prediction == 1:
                st.error("Spam 🚨")
            else:
                st.success("Not Spam ✅")
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.write("Debug Info:")
            st.write(f"Input text: {input_sms}")
            st.write(f"Transformed text: {transformed}")

# Footer
st.markdown("---")
cols = st.columns(3)
with cols[0]:
    st.info('**[GitHub](https://github.com/anilremo23)**')
with cols[1]:
    st.info('**[Kaggle](https://www.kaggle.com/remoanil)**')
with cols[2]:
    st.info('**[LinkedIn](https://www.linkedin.com/in/anil-mamidwar-001b6418/)**')

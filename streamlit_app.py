import streamlit as st
from PIL import Image
import pickle
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from xgboost import XGBClassifier
import os

# Download all required NLTK data with error handling
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()

def transform_text(text):
    try:
        text = str(text).lower()
        y = []
        # Tokenization with error handling
        try:
            text = nltk.word_tokenize(text)
        except:
            # Fallback simple tokenization if punkt fails
            text = text.split()
            
        for i in text:
            if i.isalnum():
                y.append(i)
        text = y[:]
        y.clear()
        
        # Removing stopwords and punctuations
        stop_words = set(stopwords.words('english'))
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

# Load models - adjust paths as needed
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('mnb_spam_detector.pkl', 'rb'))
except Exception as e:
    st.error(f"Failed to load models: {str(e)}")
    st.stop()

# App UI
st.title("SMS Spam classifier")

try:
    st.image(Image.open('spam_image.jpeg'))
except:
    st.warning("Could not load image")

st.write("""
A spam classifier uses machine learning to distinguish between legitimate and unsolicited messages.
Algorithm used: stacking classifier (SVM, NB, Xgboost)
""")

input_sms = st.text_area("Enter the message to check")

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message")
    else:
        try:
            # Transform and predict
            transform_sms = transform_text(input_sms)
            if transform_sms:  # Only proceed if transformation succeeded
                vector_input = tfidf.transform([transform_sms]).toarray()
                prediction = model.predict(vector_input)[0]
                
                if prediction == 1:
                    st.error("Spam 🚨")
                else:
                    st.success("Not Spam ✅")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# Footer columns
c1, c2, c3 = st.columns(3)
with c1:
    st.info('**GitHub:[@anilremo23](https://github.com/anilremo23)**', icon="🧠")
with c2:
    st.info('**Kaggle:[@remoanil](https://www.kaggle.com/remoanil)**', icon="💻")
with c3:
    st.info('**LinkedIn:[@AnilMamidwar](https://www.linkedin.com/in/anil-mamidwar-001b6418/)**', icon="👨‍💼")

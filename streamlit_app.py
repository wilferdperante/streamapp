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
from xgboost import XGBClassifier

# =============================================
# NLTK Setup with Persistent Download Solution
# =============================================
try:
    # Set NLTK data path to a writable directory
    nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(nltk_data_path, exist_ok=True)
    nltk.data.path.append(nltk_data_path)
    
    # Download required NLTK data with error handling
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_path)
        
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', download_dir=nltk_data_path)
except Exception as e:
    st.error(f"NLTK setup failed: {str(e)}")

# Initialize stemmer
ps = PorterStemmer()

# =============================================
# Text Processing with Fallback Tokenizer
# =============================================
def transform_text(text):
    try:
        text = str(text).lower()
        y = []
        
        # Tokenization with fallback
        try:
            text = nltk.word_tokenize(text)
        except:
            # Fallback to simple whitespace tokenizer
            text = text.split()
            
        for i in text:
            if i.isalnum():
                y.append(i)
        text = y[:]
        y.clear()
        
        # Stopwords with fallback
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()
            
        for i in text:
            if i not in stop_words and i not in string.punctuation:
                y.append(i)
        text = y[:]
        y.clear()
        
        # Stemming
        for i in text:
            y.append(ps.stem(i))
            
        return " ".join(y)  # Return as space-separated string
    except Exception as e:
        st.error(f"Error processing text: {str(e)}")
        return ""

# =============================================
# Model Loading with Verification
# =============================================
try:
    # Try multiple possible paths for model files
    model_paths = [
        'SMS_Spam_Classifier/vectorizer.pkl',
        'vectorizer.pkl',
        '/mount/src/streamapp/vectorizer.pkl'
    ]
    
    for path in model_paths:
        try:
            tfidf = pickle.load(open(path, 'rb'))
            if hasattr(tfidf, 'vocabulary_'):  # Verify vectorizer is fitted
                break
        except:
            continue
            
    model_paths = [
        'SMS_Spam_Classifier/mnb_spam_detector.pkl',
        'mnb_spam_detector.pkl',
        '/mount/src/streamapp/mnb_spam_detector.pkl'
    ]
    
    for path in model_paths:
        try:
            model = pickle.load(open(path, 'rb'))
            break
        except:
            continue
            
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
    image_paths = [
        'SMS_Spam_Classifier/spam_image.jpeg',
        'spam_image.jpeg',
        '/mount/src/streamapp/spam_image.jpeg'
    ]
    for path in image_paths:
        try:
            st.image(Image.open(path))
            break
        except:
            continue
except:
    st.warning("Could not load preview image")

st.write("""
A spam classifier that detects SMS spam messages using machine learning.
Algorithm used: stacking classifier (SVM, NB, Xgboost)
""")

input_sms = st.text_area("Enter the message to check")

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message")
    else:
        try:
            # 1. Preprocess
            transform_sms = transform_text(input_sms)
            
            # 2. Vectorize
            vector_input = tfidf.transform([transform_sms]).toarray()
            
            # 3. Predict
            prediction = model.predict(vector_input)[0]
            
            # 4. Display
            if prediction == 1:
                st.error("Spam 🚨")
            else:
                st.success("Not Spam ✅")
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# Footer
cols = st.columns(3)
with cols[0]:
    st.info('**GitHub: [@anilremo23](https://github.com/anilremo23)**', icon="🧠")
with cols[1]:
    st.info('**Kaggle: [@remoanil](https://www.kaggle.com/remoanil)**', icon="💻")
with cols[2]:
    st.info('**LinkedIn: [@AnilMamidwar](https://www.linkedin.com/in/anil-mamidwar-001b6418/)**', icon="👨‍💼")

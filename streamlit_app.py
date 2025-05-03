import streamlit as st
from PIL import Image
import pickle
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from xgboost import XGBClassifier

# Initialize stemmer
ps = PorterStemmer()

# Load stopwords once
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    tokens = [i for i in tokens if i not in stop_words]
    tokens = [ps.stem(i) for i in tokens]
    return tokens

# Load vectorizer and model
tfidf = pickle.load(open('SMS_Spam_Classifier/vectorizer.pkl', 'rb'))
model = pickle.load(open('SMS_Spam_Classifier/mnb_spam_detector.pkl', 'rb'))

# App title and image
st.title("📩 SMS Spam Classifier")
st.image(Image.open('SMS_Spam_Classifier/spam_image.jpeg'))

# App description
st.write("""
A spam classifier uses machine learning to distinguish between legitimate and unsolicited SMS messages.
It employs algorithms to analyze content and other features to flag messages as spam or not spam.

**Model:** Stacking Classifier (SVM, Naive Bayes, XGBoost)
""")

# User input
input_sms = st.text_area("Enter the message to check")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message first!")
    else:
        # Preprocess text
        transformed_sms = transform_text(input_sms)
        transformed_sms = " ".join(transformed_sms)

        # Vectorize text
        vector_input = tfidf.transform([transformed_sms]).toarray()

        # Predict
        prediction = model.predict(vector_input)[0]

        # Display result
        if prediction == 1:
            st.error("🚨 Spam message detected!")
        else:
            st.success("✅ This message is not spam.")

# Footer with social links
c1, c2, c3 = st.columns(3)
with c1:
    st.info('**GitHub: [@anilremo23](https://github.com/anilremo23)**', icon="🧠")
with c2:
    st.info('**Kaggle: [@remoanil](https://www.kaggle.com/remoanil)**', icon="💻")
with c3:
    st.info('**LinkedIn: [@AnilMamidwar](https://www.linkedin.com/in/anil-mamidwar-001b6418/)**', icon="👨‍💼")

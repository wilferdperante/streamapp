import streamlit as st
import pickle
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from xgboost import XGBClassifier

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()

# Function to preprocess the text
def transform_text(text):
    text = text.lower()
    y = []
    text = nltk.word_tokenize(text)

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the models
tfidf = pickle.load(open('SMS_Spam_Classifier/vectorizer.pkl','rb'))
model = pickle.load(open('SMS_Spam_Classifier/mnb_spam_detector.pkl','rb'))

# Streamlit app UI
st.title("📩 SMS Spam Classifier")

# Text input for user
input_sms = st.text_area("Enter the message")

# Prediction block
if st.button("Predict"):
    if input_sms:
        # Preprocess text
        transformed_sms = transform_text(input_sms)

        # Convert to vector
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        prediction = model.predict(vector_input)[0]

        # Display result
        if prediction == 1:
            st.success("🚨 This is a **Spam** message.")
        else:
            st.info("✅ This is a **Ham (Not Spam)** message.")
    else:
        st.warning("Please enter a message first!")

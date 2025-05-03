import streamlit as st  #used for streamlit api reference
# below all libraries were part of SMS Spam classifier model building and hence add them again.
import pickle     #to load the saved pickle files
import pandas as pd
import numpy as np
import string
import nltk      #natural language tool kit used for text processing
from nltk.corpus import stopwords  #text processing
import string
from nltk.stem.porter import PorterStemmer  #text processing
import pandas as pd
ps=PorterStemmer()   
from xgboost import XGBClassifier
nltk.download('punkt')
nltk.download('stopwords')
#function to convert SMS text to numerical form ,SMS text  we will receive on our deployed app to predict.
def transform_text(text):
        text=text.lower()
        y=[]
        #tokenization
        text=nltk.word_tokenize(text)
        for i in text:
            if i.isalnum():
                y.append(i)
        text=y[:]
        y.clear()
        #removing stopwords and punctuations
        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)
        text=y[:]
        y.clear()
    
        #stemming applied on text
        for i in text:
            y.append(ps.stem(i))
        return y
#loading  both the models from respective directory
tfidf=pickle.load(open('SMS_Spam_Classifier/vectorizer.pkl','rb'))
model=pickle.load(open('SMS_Spam_Classifier/mnb_spam_detector.pkl','rb'))
#streamlit app title
st.title("SMS Spam classifier")        
 
 
 
#text input where user will enter the SMS text to predict (As shown above)
input_sms= st.text_area("Enter the message")
 
#predict button , when clicked will execute the process
if st.button("Predict"):
    if input_sms:
        # Convert text to vector form using vectorizer
        vector_input = vectorizer.transform([input_sms])

        # Predict using the model
        prediction = model.predict(vector_input)[0]

        # Display result
        if prediction == 1:
            st.success("This is a Spam message.")
        else:
            st.info("This is a Ham (Not Spam) message.")
    else:
        st.warning("Please enter a message first!")
 
#3.predict - passing the converted text to model to predict if it is spam or ham
 
prediction= model.predict(vector_input)[0]
 
#4.display- the result on app itself , if prediction result is 1 then ui(button) will display  Spam else Not Spam
    
if prediction==1:
    st.header("Spam")
else:
    st.header("Not Spam")


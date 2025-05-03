from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Initialize them
vectorizer = TfidfVectorizer()
model = MultinomialNB()

import pickle

with open('SMS_Spam_Classifier/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('SMS_Spam_Classifier/mnb_spam_detector.pkl', 'wb') as f:
    pickle.dump(model, f)
import streamlit as st
import joblib
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords

# Load precomputed TF-IDF vectors and vectorizer during Streamlit initialization
vectorizer = joblib.load('tfidf_vectorizer.joblib')
X_transformed = joblib.load('tfidf_vectors.joblib')

# Load the trained Logistic Regression model
lr_model = joblib.load('Fake_News_Detector.joblib')  # Replace with your actual model filename

# Define stemming function
ps = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content

# Streamlit app
st.title("Fake News Detector")

# Create a text input box
user_input = st.text_area("Enter some text below:", "")

# Create a button to trigger prediction
if st.button("Predict"):
    # Apply stemming to the user input
    user_input_stemmed = stemming(user_input)

    # Transform the stemmed user input using the pre-fitted vectorizer
    user_input_transformed = vectorizer.transform([user_input_stemmed])

    # Use the trained model to make a prediction
    pred_lr = lr_model.predict(user_input_transformed)  # Use the loaded model here

    # Format output based on prediction
    if pred_lr[0] == 0:
        st.markdown("<p style='color:red;font-weight:bold;'>This piece of news cannot be Verified.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:green;font-weight:bold;'>This piece of news is Verified.</p>", unsafe_allow_html=True)

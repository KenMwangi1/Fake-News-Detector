# app.py

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load your Keras model
model = load_model(r"C:\Users\Ken\Desktop\My Learning\Portfolio Projects\Fake News Detector\Fake News Detector.h5")

# Streamlit app
def main():
    st.title("Fake News Detection Engine")

    # Add a text input box
    user_input = st.text_area("Enter the news text:")

    if st.button("Detect"):
        if user_input:
            # Create and fit the tokenizer on the input text
            tokenizer = create_and_fit_tokenizer([user_input])

            # Preprocess the input text
            preprocessed_text = preprocess_text(user_input, tokenizer)

            # Make prediction
            prediction = make_prediction(model, preprocessed_text)

            # Display the result
            st.write("Prediction:", "Fake" if prediction > 0.5 else "Real")

# Function to create and fit the tokenizer on provided text data
def create_and_fit_tokenizer(text_data):
    tokenizer = Tokenizer(num_words=10000)  # Adjust based on your requirements
    tokenizer.fit_on_texts(text_data)
    return tokenizer

# Function to preprocess the text (adjust based on your training preprocessing steps)
def preprocess_text(text, tokenizer):
    # Tokenize and pad the text using the provided tokenizer
    text_sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(text_sequences, maxlen=1000)  # Adjust maxlen
    return padded_sequence

# Function to make predictions using the loaded model
def make_prediction(model, input_text):
    # Implement your prediction logic here
    predictions = model.predict(input_text)
    return predictions[0, 0]  # Assuming a binary classification model

if __name__ == "__main__":
    main()

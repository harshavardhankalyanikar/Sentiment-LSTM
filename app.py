import streamlit as st
import pickle
import re

from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load Model
model = load_model("models/lstm_sentiment_model.keras")

# Load Tokenizer
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_length = 50

stop_words = set(stopwords.words('english'))

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

def predict_sentiment(review):

    review = clean_text(review)

    seq = tokenizer.texts_to_sequences([review])

    padded = pad_sequences(
        seq,
        maxlen=max_length,
        padding='post'
    )

    prediction = model.predict(padded)

    score = prediction[0][0]

    return score

st.set_page_config(
    page_title="Restaurant Review Sentiment Analysis",
    page_icon="🍽️"
)

st.title("🍽️ Restaurant Review Sentiment Analysis")

review = st.text_area(
    "Enter Restaurant Review"
)

if st.button("Predict"):

    if review.strip() == "":
        st.warning("Please enter a review")
    else:

        score = predict_sentiment(review)

        if score > 0.5:

            st.success(
                f"Positive Review 😊\n\nConfidence: {score:.2f}"
            )

        else:

            st.error(
                f"Negative Review 😞\n\nConfidence: {1-score:.2f}"
            )
import streamlit as st
import numpy as np
import re
import emoji
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load
import tensorflow as tf

# -------------------------------
# Load model & tokenizer
# -------------------------------
model = tf.keras.models.load_model("best_model.h5", compile=False)
tokenizer = load("tokenizer.joblib")

max_length = 80

# -------------------------------
# Internal Preprocessing
# -------------------------------
def emoji_to_text(text):
    text = emoji.demojize(text)
    text = re.sub(r':([a-z_]+):', r'\1', text)
    return text


def clean_text(text):
    text = str(text).lower()

    emoji_text = emoji_to_text(text)

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    text = text + " " + emoji_text + " " + emoji_text

    text = re.sub(r'\s+', ' ', text).strip()

    return text


# -------------------------------
# Prediction (Smart Hybrid System)
# -------------------------------
def predict_sentiment(text):
    processed_text = clean_text(text)

    seq = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(seq, maxlen=max_length, padding="post")

    prediction = model.predict(padded, verbose=0)[0]

    sentiment_labels = ["Negative", "Neutral", "Positive"]
    sentiment = sentiment_labels[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    # -------------------------------
    # Emoji Lists
    # -------------------------------
    negative_emojis = ['😡', '😠', '😞', '😢', '😭', '👎']
    positive_emojis = ['😊', '😍', '😁', '😂', '❤️', '👍', '🔥']
    sarcasm_emojis = ['😂', '😏', '🙄']

    # -------------------------------
    # Strong Negative Keywords
    # -------------------------------
    negative_words = [
        "hate", "worst", "bad", "terrible", "awful",
        "disappointed", "useless", "annoying", "problem",
        "issue", "sucks", "garbage", "broken",
        "crash", "crashed", "crashing", "lag", "laggy",
        "slow", "error", "fail", "failed", "failing",
        "bug", "bugs", "glitch", "glitches",
        "freeze", "freezing", "stuck"
    ]

   
    sarcasm_phrases = [
        "yeah right",
        "as if",
        "just perfect",
        "works perfectly",
        "nice job",
        "great job",
        "amazing",
        "perfect",
        "love this",
        "so good"
    ]

    text_lower = text.lower()

    has_negative_word = any(word in text_lower for word in negative_words)
    has_sarcasm_emoji = any(e in text for e in sarcasm_emojis)
    has_sarcasm_phrase = any(phrase in text_lower for phrase in sarcasm_phrases)

    # -------------------------------
    # 🔥 Advanced Sarcasm Detection
    # -------------------------------
    if (has_negative_word and has_sarcasm_emoji) or \
       (has_sarcasm_phrase and has_sarcasm_emoji):
        return "Negative", 0.95

    # -------------------------------
    # Emoji Overrides
    # -------------------------------
    if text.strip() in negative_emojis:
        return "Negative", 1.0

    if text.strip() in positive_emojis:
        return "Positive", 1.0

    if any(e in text for e in negative_emojis):
        return "Negative", confidence

    if any(e in text for e in positive_emojis):
        return "Positive", confidence

    return sentiment, confidence


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.title("💬 Twitter Sentiment Analysis")
st.write("Enter text below to analyze sentiment as Negative, Neutral, or Positive.")

user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    if user_input.strip():
        sentiment, confidence = predict_sentiment(user_input)

        st.subheader(f"Predicted Sentiment: {sentiment}")
        

    else:
        st.warning("Please enter some text for analysis.")
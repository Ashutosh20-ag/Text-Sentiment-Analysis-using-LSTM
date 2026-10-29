git add ."""
Streamlit app for LSTM Sentiment Analysis
Run:  streamlit run app.py
(pehle main.py chala kar sentiment_lstm.keras bana lo)
"""
import re
import streamlit as st
import keras
from keras.datasets import imdb
from keras.utils import pad_sequences

VOCAB_SIZE = 10000
MAX_LEN = 200
MODEL_PATH = "sentiment_bilstm.keras"


@st.cache_resource
def load_resources():
    model = keras.models.load_model(MODEL_PATH)
    word_index = imdb.get_word_index()
    return model, word_index


def encode_text(text, word_index):
    words = re.findall(r"[a-z']+", text.lower())
    seq = [1]  # start token
    for w in words:
        if w in word_index and word_index[w] + 3 < VOCAB_SIZE:
            seq.append(word_index[w] + 3)
        else:
            seq.append(2)  # unknown
    return pad_sequences([seq], maxlen=MAX_LEN, padding="pre", truncating="post")


st.set_page_config(page_title="Sentiment Analysis", page_icon="🎬")
st.title("🎬 Text Sentiment Analysis using LSTM")
st.write("Koi bhi English movie review likho, model batayega ki wo positive hai ya negative.")

model, word_index = load_resources()

text = st.text_area("Apna review likho:", height=150,
                    placeholder="e.g. The movie was amazing, I loved the acting!")

if st.button("Analyze"):
    if not text.strip():
        st.warning("Pehle kuch text likho.")
    else:
        prob = float(model.predict(encode_text(text, word_index), verbose=0)[0][0])
        if prob > 0.5:
            st.success(f"Positive 😊  (confidence: {prob:.0%})")
        else:
            st.error(f"Negative 😞  (confidence: {1 - prob:.0%})")
        st.progress(prob, text=f"Positive score: {prob:.2f}")

st.caption("Model: Embedding + LSTM trained on the IMDB reviews dataset (English only).")
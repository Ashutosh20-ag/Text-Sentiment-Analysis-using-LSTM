# 🎬 Text Sentiment Analysis using LSTM

A deep learning project that classifies movie reviews as **Positive** or **Negative** using an LSTM neural network, with a Streamlit web app for live predictions.

## 📌 Project Overview

- **Problem:** Automatically detect the sentiment (positive / negative) of a text review.
- **Dataset:** IMDB Movie Reviews (50,000 reviews: 25,000 train + 25,000 test), loaded via Keras.
- **Models:** Basic LSTM and Bidirectional LSTM (BiLSTM).
- **Frameworks:** TensorFlow / Keras, scikit-learn, Streamlit.

## 🧠 Model Architecture

### Model 1: Basic LSTM
```
Embedding(10000, 64) → LSTM(64) → Dropout(0.5) → Dense(1, sigmoid)
```

### Model 2: Bidirectional LSTM
```
Embedding(10000, 128, mask_zero=True)
→ Bidirectional(LSTM(64, return_sequences=True))
→ Bidirectional(LSTM(32))
→ Dropout(0.5) → Dense(1, sigmoid)
```

**Settings:** vocabulary = 10,000 words, max review length = 200, optimizer = Adam, loss = binary crossentropy, EarlyStopping on validation loss.

## 📊 Results

| Model | Test Accuracy |
|-------|---------------|
| Basic LSTM | **main.py** |
| Bidirectional LSTM | **main_bilstm.py%** |

Training graphs: `training_graphs.png` and `training_graphs_bilstm.png`

> Replace XX.XX% with the values printed by `main.py` and `main_bilstm.py`.

## 🗂 Project Structure

```
sentiment-lstm/
├── main.py                  # Basic LSTM training
├── main_bilstm.py           # BiLSTM training
├── app.py                   # Streamlit web app
├── requirements.txt
├── sentiment_lstm.keras     # Saved basic model
├── sentiment_bilstm.keras   # Saved BiLSTM model
└── README.md
```

## ⚙️ Installation & Usage

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python main.py            # or: python main_bilstm.py

# 4. Run the web app
streamlit run app.py
```

## 🖥 Web App

Type any English review and the app shows the predicted sentiment with a confidence score.

*(Add a screenshot of the app here: `![App Screenshot](screenshot.png)`)*

## ⚠️ Limitations

- Trained only on English movie reviews, so it may not work well on other domains.
- Sarcasm and very short sentences can be misclassified.

## 🚀 Future Improvements

- Use pre-trained embeddings (GloVe / Word2Vec)
- Add a neutral class (3-class classification)
- Try Transformer models (BERT)
- Train on other datasets (Amazon reviews, tweets)

## 👤 Author

Your Name — [GitHub](https://github.com/Ashutosh20-ag)
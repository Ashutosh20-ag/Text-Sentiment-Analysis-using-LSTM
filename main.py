"""
Text Sentiment Analysis with LSTM (IMDB dataset)
Run:  python main.py
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")  # graph ko file me save karega, window nahi kholega
import matplotlib.pyplot as plt

import keras
from keras import layers
from keras.datasets import imdb
from keras.utils import pad_sequences
from sklearn.metrics import classification_report

# ------------------ Settings ------------------
VOCAB_SIZE = 10000   # top 10,000 words
MAX_LEN = 200        # har review ki fixed length
EPOCHS = 10
BATCH_SIZE = 64
MODEL_PATH = "sentiment_lstm.keras"

# ------------------ 1. Data load ------------------
print("Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
print("Train samples:", len(x_train), "| Test samples:", len(x_test))

# ------------------ 2. Padding ------------------
# padding="pre" (shuru me zeros) LSTM ke liye better hota hai
x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="pre", truncating="post")
x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="pre", truncating="post")

# ------------------ 3. Model ------------------
model = keras.Sequential([
    keras.Input(shape=(MAX_LEN,)),
    layers.Embedding(VOCAB_SIZE, 64),
    layers.LSTM(64),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)
model.summary()

# ------------------ 4. Training ------------------
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=2, restore_best_weights=True
)

history = model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[early_stop],
)

# ------------------ 5. Evaluation ------------------
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc:.2%}")

y_pred = (model.predict(x_test, verbose=0) > 0.5).astype(int).ravel()
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Graphs save karo
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()

plt.tight_layout()
plt.savefig("training_graphs.png")
print("Graphs saved: training_graphs.png")

# ------------------ 6. Model save ------------------
model.save(MODEL_PATH)
print("Model saved:", MODEL_PATH)

# ------------------ 7. Apna text predict karo ------------------
word_index = imdb.get_word_index()


def encode_text(text):
    words = re.findall(r"[a-z']+", text.lower())
    seq = [1]  # 1 = start token
    for w in words:
        if w in word_index and word_index[w] + 3 < VOCAB_SIZE:
            seq.append(word_index[w] + 3)  # IMDB me indices 3 se shift hote hain
        else:
            seq.append(2)  # 2 = unknown word
    return pad_sequences([seq], maxlen=MAX_LEN, padding="pre", truncating="post")


def predict_sentiment(text):
    prob = float(model.predict(encode_text(text), verbose=0)[0][0])
    label = "Positive" if prob > 0.5 else "Negative"
    return label, prob


if __name__ == "__main__":
    samples = [
        "This movie was absolutely fantastic and amazing",
        "Worst film ever, total waste of time",
    ]
    for s in samples:
        label, prob = predict_sentiment(s)
        print(f"\nText: {s}\n-> {label} (score: {prob:.2f})")

    print("\nApna review likho (band karne ke liye 'q' type karo):")
    while True:
        text = input("> ").strip()
        if text.lower() == "q":
            break
        if text:
            label, prob = predict_sentiment(text)
            print(f"   {label} (score: {prob:.2f})")
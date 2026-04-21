from flask import Flask, request, jsonify
import tensorflow as tf
import gdown
import pickle, numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

app = Flask(__name__)

MODEL_URL = "https://drive.google.com/uc?id=1LEFF3gbokadK4FRS6v3hLefO0Q5rbQlh"
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH)

with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(text, max_sequence_len=50):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding="pre")
    predicted = np.argmax(model.predict(token_list), axis=-1)
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word
    return ""

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_text = data.get("text", "")
    next_word = predict_next_word(input_text)
    return jsonify({"next_word": next_word})

if __name__ == "__main__":
    app.run()

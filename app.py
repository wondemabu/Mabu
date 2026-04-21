import streamlit as st
import tensorflow as tf
import gdown
import os
import pickle, numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 🔹 Download model from Google Drive
url = "https://drive.google.com/uc?id=1LEFF3gbokadK4FRS6v3hLefO0Q5rbQlh"
output = "/content/model_fasttext.h5"
gdown.download(url, output, quiet=False)

# 🔹 Load model
model = tf.keras.models.load_model("/content/model_fasttext.h5")

# 🔹 Load tokenizer (you can also host tokenizer.pkl externally if it's large)
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# 🔹 Prediction function
def predict_next_word(text, max_sequence_len=50):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=-1)
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word
    return ""

# 🔹 Streamlit UI
st.title("Wolaytta Next Word Predictor")
input_text = st.text_input("Enter text:")
if st.button("Predict"):
    st.write("Predicted next word:", predict_next_word(input_text))

import streamlit as st
from tensorflow.keras.models import load_model
import pickle, numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer and model
with open('/content/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
model = load_model('/content/model_fasttext.h5')

def predict_next_word(text, max_sequence_len=50):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=-1)
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word
    return ""

st.title("Wolaytta Next Word Predictor")
input_text = st.text_input("Enter text:")
if st.button("Predict"):
    st.write("Predicted next word:", predict_next_word(input_text))

import streamlit as st
import tensorflow as tf
import os
import pickle
import numpy as np

from src.utils import load_tokenizer  # Make sure utils.py is in src/ folder

# Page config
st.set_page_config(page_title="Sequential Recipe Step Analyzer", page_icon="🍳", layout="centered")

# Function to predict next ingredient
def predict_next_ingredient(model, tokenizer, input_text, top_k=5):
    max_length = model.input_shape[1]
    seq = tokenizer.texts_to_sequences([input_text.lower()])[0]
    seq_padded = tf.keras.preprocessing.sequence.pad_sequences(
        [seq], maxlen=max_length, padding="post", truncating="post"
    )
    pred_probs = model.predict(seq_padded, verbose=0)[0]
    top_indices = pred_probs.argsort()[-top_k:][::-1]
    pred_index = np.random.choice(top_indices)
    for word, index in tokenizer.word_index.items():
        if index == pred_index:
            return word
    return "<unknown>"

# Load resources with caching
@st.cache_resource
def load_resources():
    tokenizer_path = os.path.join("models", "tokenizer.pkl")
    model_path = os.path.join("outputs", "models", "ingredient_gru_model")

    # Load tokenizer
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # Load model
    model = tf.keras.models.load_model(model_path)

    return tokenizer, model

# Load model and tokenizer
try:
    tokenizer, model = load_resources()
except Exception as e:
    st.error("Failed to load model or tokenizer. Check paths and files!")
    st.stop()  # Stop the app here if loading fails

# UI
st.title("🍳 Sequential Recipe Step Analyzer")
user_input = st.text_input("Enter ingredients separated by commas:", placeholder="egg, salt, pepper")
if st.button("Predict Next Ingredient"):
    if not user_input.strip():
        st.warning("Please enter at least one ingredient.")
    else:
        input_text = " ".join([i.strip() for i in user_input.split(",")])
        next_ingredient = predict_next_ingredient(model, tokenizer, input_text)
        st.success(f"✅ Predicted Next Ingredient: **{next_ingredient}**")

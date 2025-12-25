import streamlit as st
import tensorflow as tf
from src.utils import load_tokenizer
import numpy as np
import os
# Page configuration
st.set_page_config(
    page_title="Sequential Recipe Step Analyzer 🍳",
    page_icon="🍲",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Function to predict next ingredient with top-k sampling
def predict_next_ingredient(model, tokenizer, input_text, top_k=5):
    # Get correct input length from model
    max_length = model.input_shape[1]

    # Convert words to sequence
    seq = tokenizer.texts_to_sequences([input_text.lower()])[0]

    # Pad sequence to match model input
    seq_padded = tf.keras.preprocessing.sequence.pad_sequences(
        [seq], maxlen=max_length, padding="post", truncating="post"
    )

    # Predict probabilities
    pred_probs = model.predict(seq_padded, verbose=0)[0]

    # Top-k sampling
    top_indices = pred_probs.argsort()[-top_k:][::-1]
    pred_index = np.random.choice(top_indices)

    # Convert index back to word
    for word, index in tokenizer.word_index.items():
        if index == pred_index:
            return word
    return "<unknown>"

# Cache model and tokenizer
@st.cache_resource


def load_resources():
    tokenizer_path = os.path.join("models", "tokenizer.pkl")
    tokenizer = load_tokenizer(path=tokenizer_path)
    
    model_path = os.path.join("outputs", "models", "ingredient_gru_model")
    model = tf.keras.models.load_model(model_path)
    
    return tokenizer, model


# Main UI
st.title("🍳 Sequential Recipe Step Analyzer")
st.markdown(
    """
    Predict the **next ingredient** in your recipe using a GRU-based model.
    Enter a sequence of ingredients below, separated by commas.
    """
)
st.markdown("---")

# Input Section
st.markdown("### 🥄 Enter Ingredients")
user_input = st.text_input(
    "Type ingredients separated by commas:",
    placeholder="salt, pepper, oil"
)
predict_btn = st.button("Predict Next Ingredient", use_container_width=True)

# Prediction output
if predict_btn:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter at least one ingredient.")
    else:
        input_text = " ".join([i.strip() for i in user_input.split(",")])
        next_ingredient = predict_next_ingredient(model, tokenizer, input_text)
        st.success(f"✅ Predicted Next Ingredient: **{next_ingredient}**")

st.markdown("---")

# About / Project Info
with st.expander("ℹ️ About this Project"):
    st.markdown(
        """
        **Sequential Recipe Step Analyzer** predicts the next ingredient in a recipe sequence using:
        - A cleaned recipe dataset  
        - GRU (Gated Recurrent Unit) model for sequence prediction  
        - Top-k sampling for more realistic predictions  
        - Interactive Streamlit UI  

        **Features:**  
        - Enter your ingredients and get real-time predictions  
        """
    )

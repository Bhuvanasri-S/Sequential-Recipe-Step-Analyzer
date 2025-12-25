import numpy as np
import tensorflow as tf
import os

from model import build_gru_model
from utils import load_tokenizer

if __name__ == "__main__":

    print("Loading training data...")

    # Load input and target sequences
    X = np.load("outputs/results/X_train.npy")
    y = np.load("outputs/results/y_train.npy")

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("Loading tokenizer...")
    # Use your actual path
    tokenizer = load_tokenizer(path="models/tokenizer.pkl")
    vocab_size = len(tokenizer.word_index) + 1

    input_length = X.shape[1]

    print("Building GRU model...")
    model = build_gru_model(
        vocab_size=vocab_size,
        input_length=input_length
    )

    model.summary()

    print("Training started...")
    history = model.fit(
        X,
        y,
        epochs=10,
        batch_size=64,
        validation_split=0.1
    )

    # Ensure output folder exists
    os.makedirs("outputs/models", exist_ok=True)

    # Save trained model
    model.save("outputs/models/ingredient_gru_model")

    print("Training completed")
    print("Model saved at outputs/models/ingredient_gru_model")

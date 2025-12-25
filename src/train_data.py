import numpy as np
import tensorflow as tf
import os

from model import build_gru_model
from utils import load_tokenizer


if __name__ == "__main__":

    print("Loading training data...")

    X = np.load("outputs/results/X_train.npy")
    y = np.load("outputs/results/y_train.npy")

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()
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

    # Ensure model directory exists
    os.makedirs("outputs/models", exist_ok=True)

    model.save("outputs/models/ingredient_gru_model")

    print("Training completed")
    print("Model saved at outputs/models/ingredient_gru_model")

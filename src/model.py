import tensorflow as tf


def build_gru_model(
    vocab_size,
    embedding_dim=64,
    gru_units=128,
    input_length=None
):
    """
    Builds and compiles a GRU-based sequence model.
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=input_length
        ),

        tf.keras.layers.GRU(
            gru_units,
            return_sequences=False
        ),

        tf.keras.layers.Dense(
            vocab_size,
            activation="softmax"
        )
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    # Test model creation
    dummy_vocab_size = 10000
    dummy_input_length = 20

    model = build_gru_model(
        vocab_size=dummy_vocab_size,
        input_length=dummy_input_length
    )

    model.summary()

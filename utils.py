import pandas as pd
import tensorflow as tf
import pickle


def load_clean_data(csv_path=r"C:\Users\HP\OneDrive\Sequence recipe step analyzer\data\cleaned_data.csv"):
    """
    Loads cleaned recipe text from CSV.
    """
    df = pd.read_csv(csv_path)
    texts = df["text"].astype(str).tolist()
    return texts


def create_tokenizer(texts, vocab_size=10000):
    """
    Fits a tokenizer on recipe texts.
    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=vocab_size,
        oov_token="<OOV>"
    )
    tokenizer.fit_on_texts(texts)
    return tokenizer


def text_to_sequences(tokenizer, texts, max_length=30):
    """
    Converts texts to padded numeric sequences.
    """
    sequences = tokenizer.texts_to_sequences(texts)

    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    return padded_sequences


def save_tokenizer(tokenizer, path=r"C:\Users\HP\OneDrive\Sequence recipe step analyzer\models\tokenizer.pkl"):
    """
    Saves tokenizer for future inference.
    """
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)


def load_tokenizer(path=r"C:\Users\HP\OneDrive\Sequence recipe step analyzer\models\tokenizer.pkloutputs/models/tokenizer.pkl"):
    """
    Loads saved tokenizer.
    """
    with open(path, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


if __name__ == "__main__":
    texts = load_clean_data()
    tokenizer = create_tokenizer(texts)
    sequences = text_to_sequences(tokenizer, texts)

    save_tokenizer(tokenizer)

    print("Tokenization completed")
    print("Total sequences:", sequences.shape)

import json
import pandas as pd
import os


def load_and_clean_data(
    raw_path=r"C:\Users\HP\OneDrive\Sequence recipe step analyzer\data\recipe.json",
    output_path=r"C:\Users\HP\OneDrive\Sequence recipe step analyzer\data\cleaned_data.csv"
):
    """
    Loads Kaggle recipe ingredients dataset,
    cleans it, and saves as CSV for GRU processing.
    """

    print("Loading raw recipe data...")

    with open(raw_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"Total recipes found: {len(data)}")

    cleaned_texts = []

    for recipe in data:
        ingredients = recipe.get("ingredients", [])

        # Skip recipes without ingredients
        if not ingredients:
            continue

        # Join ingredients into a single sequence
        text = " ".join(ingredients).lower()

        cleaned_texts.append(text)

    # Create DataFrame
    df = pd.DataFrame(cleaned_texts, columns=["text"])

    print(f"Cleaned recipes count: {len(df)}")

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save cleaned data
    df.to_csv(output_path, index=False)

    print(f"Cleaned data saved at: {output_path}")

    return df


if __name__ == "__main__":
    load_and_clean_data()

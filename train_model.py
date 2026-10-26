
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "WELFake_Dataset.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "fake_news_model.joblib"


def main():
    print("Loading dataset...")

    data = pd.read_csv(DATA_PATH)

    print("Dataset columns:", data.columns.tolist())

    # Dataset ke column names ko standard format mein lao
    data.columns = data.columns.str.strip().str.lower()

    if "text" not in data.columns or "label" not in data.columns:
        raise ValueError(
            "Dataset mein text aur label columns nahi mile."
        )

    data = data[["text", "label"]].copy()
    data["text"] = data["text"].fillna("").astype(str).str.strip()
    data = data[data["text"] != ""]
    data = data.drop_duplicates(subset=["text"])

    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data.dropna(subset=["label"])

    print("Total usable articles:", len(data))
    print("Label counts:")
    print(data["label"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2)
        )),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    print("\nTraining model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nAccuracy:", round(
        accuracy_score(y_test, predictions), 4
    ))

    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved successfully!")
    print("Location:", MODEL_PATH)


if __name__ == "__main__":
    main()

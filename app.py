
from pathlib import Path

import joblib
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "fake_news_model.joblib"

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form.get("news", "").strip()

    if not news_text:
        return render_template(
            "index.html",
            error="Please enter some news text."
        )

    if len(news_text) < 50:
        return render_template(
            "index.html",
            error="Please enter a longer news article for a more useful prediction.",
            news=news_text
        )

    prediction = int(model.predict([news_text])[0])
    probabilities = model.predict_proba([news_text])[0]

    # Classes [0, 1] ke order mein probabilities
    class_probabilities = dict(
        zip(model.classes_.astype(int), probabilities)
    )

    if prediction == 0:
        result = "Likely Real News"
    else:
        result = "Likely Fake News"

    # Model confidence: predicted class ki probability
    confidence = round(
        float(class_probabilities[prediction]) * 100, 2
    )

    print("News entered:", news_text, flush=True)
    print("Raw prediction:", prediction, flush=True)
    print("Confidence:", confidence, flush=True)

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        news=news_text,
        disclaimer=(
            "This is an AI prediction, not a verified fact. "
            "Please confirm important claims using reliable sources."
        )
    )


if __name__ == "__main__":
    app.run(debug=False)

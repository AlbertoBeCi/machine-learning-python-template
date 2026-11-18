from pathlib import Path
import pandas as pd
from transformers import pipeline

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
RAW_PATH = Path("data/raw/reviews.csv")
PROCESSED_PATH = Path("data/processed/reviews_with_sentiment.csv")


def map_sentiment_band(stars: int) -> str:
    """Map 1-5 star prediction to negative, neutral, or positive band."""
    if stars <= 2:
        return "negative"
    if stars == 3:
        return "neutral"
    return "positive"


def main() -> None:
    df = pd.read_csv(RAW_PATH)

    classifier = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
    )

    results = classifier(df["review_text"].tolist(), batch_size=32, truncation=True)
    df["pred_rating"] = [int(r["label"].split()[0]) for r in results]
    df["pred_score"] = [round(float(r["score"]), 4) for r in results]
    df["sentiment_band"] = df["pred_rating"].apply(map_sentiment_band)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved {len(df)} rows to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()

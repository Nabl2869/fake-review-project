from pathlib import Path
import re

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"

INPUT_FILE = PROCESSED / "reviews_clean.parquet"
OUTPUT_FILE = PROCESSED / "reviews_text_features.parquet"


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df = pd.read_parquet(INPUT_FILE)

print(f"Loaded {len(df):,} reviews")


# ------------------------------------------------------------
# 1. Word count
# ------------------------------------------------------------

if "n_words" not in df.columns:
    df["n_words"] = (
        df["review_text"]
        .fillna("")
        .str.split()
        .str.len()
    )


# ------------------------------------------------------------
# 2. Character count
# ------------------------------------------------------------

df["n_chars"] = (
    df["review_text"]
    .fillna("")
    .str.len()
)


# ------------------------------------------------------------
# 3. First-person pronoun rate
# ------------------------------------------------------------

FIRST_PERSON = {
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves"
}


def first_person_rate(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0

    words = re.findall(r"\b[a-zA-Z']+\b", text.lower())

    if len(words) == 0:
        return 0.0

    first_person_count = sum(
        1 for word in words
        if word in FIRST_PERSON
    )

    return first_person_count / len(words)


print("Calculating first-person pronoun rate...")

df["first_person_rate"] = (
    df["review_text"]
    .apply(first_person_rate)
)


# ------------------------------------------------------------
# 4. VADER sentiment score
# ------------------------------------------------------------

analyzer = SentimentIntensityAnalyzer()


def sentiment_score(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0

    scores = analyzer.polarity_scores(text)

    # Compound score ranges from -1 to +1
    return scores["compound"]


print("Calculating VADER sentiment scores...")

df["sentiment_score"] = (
    df["review_text"]
    .apply(sentiment_score)
)


# ------------------------------------------------------------
# 5. Convert star rating into sentiment class
#
# 1-2 stars = negative
# 3 stars   = neutral
# 4-5 stars = positive
# ------------------------------------------------------------

def rating_sentiment(rating):
    if rating <= 2:
        return -1
    elif rating >= 4:
        return 1
    else:
        return 0


df["rating_sentiment"] = (
    df["rating"]
    .apply(rating_sentiment)
)


# ------------------------------------------------------------
# 6. Convert VADER score into sentiment class
#
# Standard VADER thresholds:
#
# compound >= 0.05   -> positive
# compound <= -0.05  -> negative
# otherwise          -> neutral
# ------------------------------------------------------------

def text_sentiment_class(score):
    if score >= 0.05:
        return 1
    elif score <= -0.05:
        return -1
    else:
        return 0


df["text_sentiment_class"] = (
    df["sentiment_score"]
    .apply(text_sentiment_class)
)


# ------------------------------------------------------------
# 7. Sentiment / star-rating agreement
#
# 1 = text sentiment agrees with rating
# 0 = text sentiment does not agree
# ------------------------------------------------------------

df["sentiment_rating_agree"] = (
    df["rating_sentiment"]
    == df["text_sentiment_class"]
).astype(int)


# ------------------------------------------------------------
# Feature checks
# ------------------------------------------------------------

print("\n--- Text feature checks ---")

features = [
    "n_words",
    "n_chars",
    "first_person_rate",
    "sentiment_score",
    "sentiment_rating_agree"
]

print(df[features].describe())


# ------------------------------------------------------------
# Compare fake vs genuine reviews
# ------------------------------------------------------------

print("\nAverage word count by label:")

print(
    df.groupby("fake")["n_words"]
    .mean()
)


print("\nAverage VADER sentiment score by label:")

print(
    df.groupby("fake")["sentiment_score"]
    .mean()
)


print("\nSentiment/rating agreement rate by label:")

print(
    df.groupby("fake")["sentiment_rating_agree"]
    .mean()
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

df.to_parquet(
    OUTPUT_FILE,
    index=False
)

print("\nSaved text feature data to:")
print(OUTPUT_FILE)

print("\nDone.")
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"

INPUT_FILE = PROCESSED / "reviews_behavior_features.parquet"

TRAIN_FILE = PROCESSED / "train_reviews.parquet"
TEST_FILE = PROCESSED / "test_reviews.parquet"

TRAIN_TFIDF = PROCESSED / "tfidf_train.npz"
TEST_TFIDF = PROCESSED / "tfidf_test.npz"

VECTORIZER_FILE = PROCESSED / "tfidf_vectorizer.joblib"


# ------------------------------------------------------------
# Load processed reviews
# ------------------------------------------------------------

df = pd.read_parquet(INPUT_FILE)

print(f"Loaded {len(df):,} reviews")


# ------------------------------------------------------------
# Prepare text
# ------------------------------------------------------------

df["review_text"] = df["review_text"].fillna("")


# ------------------------------------------------------------
# Create combined stratification key
#
# This preserves BOTH:
# - fake/genuine label distribution
# - reviewer history group distribution
# ------------------------------------------------------------

df["stratify_key"] = (
    df["fake"].astype(str)
    + "_"
    + df["history_group"].astype(str)
)


# ------------------------------------------------------------
# 80/20 train/test split
# ------------------------------------------------------------

indices = np.arange(len(df))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=df["stratify_key"]
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()


# Remove temporary stratification column
train_df = train_df.drop(columns=["stratify_key"])
test_df = test_df.drop(columns=["stratify_key"])


# ------------------------------------------------------------
# Basic split checks
# ------------------------------------------------------------

print(f"Training reviews: {len(train_df):,}")
print(f"Testing reviews:  {len(test_df):,}")

print(
    f"Training fake rate: "
    f"{train_df['fake'].mean() * 100:.2f}%"
)

print(
    f"Testing fake rate: "
    f"{test_df['fake'].mean() * 100:.2f}%"
)


# ------------------------------------------------------------
# Check fake counts by reviewer history group
# ------------------------------------------------------------

print("\n--- TRAIN fake counts by history group ---")

train_fake_counts = (
    train_df[train_df["fake"] == 1]
    .groupby("history_group", observed=False)
    .size()
)

print(train_fake_counts)


print("\n--- TEST fake counts by history group ---")

test_fake_counts = (
    test_df[test_df["fake"] == 1]
    .groupby("history_group", observed=False)
    .size()
)

print(test_fake_counts)


# ------------------------------------------------------------
# Check total counts by reviewer history group
# ------------------------------------------------------------

print("\n--- TRAIN total counts by history group ---")

print(
    train_df["history_group"]
    .value_counts()
    .sort_index()
)


print("\n--- TEST total counts by history group ---")

print(
    test_df["history_group"]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# TF-IDF
#
# ngram_range=(1, 2)
#   1 = individual words
#   2 = two-word phrases / bigrams
#
# fit_transform is used ONLY on training text.
# Test data only uses transform().
# ------------------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.95,
    max_features=20000,
    sublinear_tf=True
)

print("\nFitting TF-IDF on TRAINING reviews only...")

X_train_tfidf = vectorizer.fit_transform(
    train_df["review_text"]
)

print("Transforming test reviews...")

X_test_tfidf = vectorizer.transform(
    test_df["review_text"]
)


# ------------------------------------------------------------
# TF-IDF checks
# ------------------------------------------------------------

print("\n--- TF-IDF checks ---")

print(
    f"Training matrix shape: "
    f"{X_train_tfidf.shape}"
)

print(
    f"Testing matrix shape: "
    f"{X_test_tfidf.shape}"
)

print(
    f"Vocabulary size: "
    f"{len(vectorizer.vocabulary_):,}"
)


feature_names = vectorizer.get_feature_names_out()

print("\nFirst 20 TF-IDF features:")

for feature in feature_names[:20]:
    print(feature)


# ------------------------------------------------------------
# Save train/test review tables
# ------------------------------------------------------------

train_df.to_parquet(
    TRAIN_FILE,
    index=False
)

test_df.to_parquet(
    TEST_FILE,
    index=False
)


# ------------------------------------------------------------
# Save sparse TF-IDF matrices
# ------------------------------------------------------------

save_npz(
    TRAIN_TFIDF,
    X_train_tfidf
)

save_npz(
    TEST_TFIDF,
    X_test_tfidf
)


# ------------------------------------------------------------
# Save fitted vectorizer
# ------------------------------------------------------------

joblib.dump(
    vectorizer,
    VECTORIZER_FILE
)


# ------------------------------------------------------------
# Done
# ------------------------------------------------------------

print("\nSaved:")

print(TRAIN_FILE)
print(TEST_FILE)
print(TRAIN_TFIDF)
print(TEST_TFIDF)
print(VECTORIZER_FILE)

print("\nDone.")
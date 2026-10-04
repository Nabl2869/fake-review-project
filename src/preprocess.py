from pathlib import Path
import pandas as pd
import numpy as np

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "yelpzip"
PROCESSED = ROOT / "processed"

PROCESSED.mkdir(parents=True, exist_ok=True)

# Load metadata
meta = pd.read_csv(
    DATA / "metadata",
    sep="\t",
    header=None,
    names=["user_id", "prod_id", "rating", "label", "date"]
)

print(f"Loaded {len(meta):,} metadata rows")

# Convert date
meta["date"] = pd.to_datetime(meta["date"], errors="coerce")

# YelpZip label:
# -1 = filtered/fake
#  1 = genuine
meta["fake"] = (meta["label"] == -1).astype(int)

# Load review text
texts = []

with open(DATA / "reviewContent", encoding="utf-8", errors="replace") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t", 3)

        if len(parts) == 4:
            texts.append(parts[3])
        else:
            texts.append("")

print(f"Loaded {len(texts):,} review texts")

# Make sure metadata and text files match
if len(texts) != len(meta):
    raise ValueError(
        f"metadata has {len(meta):,} rows but "
        f"reviewContent has {len(texts):,} rows"
    )

meta["review_text"] = texts

# Basic checks
print("\n--- Data checks ---")
print(f"Reviews: {len(meta):,}")
print(f"Reviewers: {meta['user_id'].nunique():,}")
print(f"Businesses: {meta['prod_id'].nunique():,}")
print(f"Missing dates: {meta['date'].isna().sum():,}")
print(f"Fake rate: {meta['fake'].mean() * 100:.2f}%")

# Reviewer total review counts
review_counts = meta.groupby("user_id").size()

meta["total_user_reviews"] = meta["user_id"].map(review_counts)

# Reviewer history groups
bins = [0, 1, 2, 5, 20, np.inf]
labels = ["1", "2", "3-5", "6-20", "21+"]

meta["history_group"] = pd.cut(
    meta["total_user_reviews"],
    bins=bins,
    labels=labels
)

# Simple text length
meta["n_words"] = (
    meta["review_text"]
    .fillna("")
    .str.split()
    .str.len()
)

# Sort consistently
meta = meta.sort_values(
    ["date", "user_id", "prod_id"]
).reset_index(drop=True)

# Save
output = PROCESSED / "reviews_clean.parquet"

meta.to_parquet(
    output,
    index=False
)

print("\nReviewer history groups:")
print(meta["history_group"].value_counts().sort_index())

print(f"\nSaved cleaned data to:")
print(output)

print("\nDone.")
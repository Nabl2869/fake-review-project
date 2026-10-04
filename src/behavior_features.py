from pathlib import Path
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"

INPUT_FILE = PROCESSED / "reviews_text_features.parquet"
OUTPUT_FILE = PROCESSED / "reviews_behavior_features.parquet"

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df = pd.read_parquet(INPUT_FILE)

print(f"Loaded {len(df):,} reviews")

# Sort by date so all history calculations make sense
df = df.sort_values(
    ["date", "user_id", "prod_id"]
).reset_index(drop=True)

# ------------------------------------------------------------
# Helper: reviewer features by day
# ------------------------------------------------------------

# Important:
# Reviews on the SAME date do not count as prior history.
# We first summarize each user's activity per date.

user_day = (
    df.groupby(["user_id", "date"])
    .agg(
        reviews_that_day=("rating", "size"),
        extreme_that_day=("rating", lambda s: ((s == 1) | (s == 5)).sum())
    )
    .reset_index()
)

# Prior review count
user_day["prior_review_count"] = (
    user_day.groupby("user_id")["reviews_that_day"]
    .cumsum()
    - user_day["reviews_that_day"]
)

# Prior extreme review count
user_day["prior_extreme_count"] = (
    user_day.groupby("user_id")["extreme_that_day"]
    .cumsum()
    - user_day["extreme_that_day"]
)

# Prior extreme rating rate
user_day["prior_extreme_rate"] = np.where(
    user_day["prior_review_count"] > 0,
    user_day["prior_extreme_count"] / user_day["prior_review_count"],
    np.nan
)

# Reviewer first review date
first_date = (
    user_day.groupby("user_id")["date"]
    .transform("min")
)

# Reviewer age in days BEFORE the current date
user_day["reviewer_age_days"] = (
    user_day["date"] - first_date
).dt.days

# For the first review date, no prior history exists
user_day.loc[
    user_day["prior_review_count"] == 0,
    "reviewer_age_days"
] = np.nan

print("Calculating reviewer daily-history features...")

# Sort so each user's dates are in chronological order
user_day = user_day.sort_values(
    ["user_id", "date"]
).reset_index(drop=True)

# Maximum number of reviews posted on any PRIOR day
user_day["prior_max_reviews_one_day"] = (
    user_day.groupby("user_id")["reviews_that_day"]
    .transform(lambda s: s.cummax().shift(1))
)

# If there is no history, mark as missing
user_day.loc[
    user_day["prior_review_count"] == 0,
    "prior_max_reviews_one_day"
] = np.nan

# ------------------------------------------------------------
# Merge reviewer features back to individual reviews
# ------------------------------------------------------------

df = df.merge(
    user_day[
        [
            "user_id",
            "date",
            "prior_review_count",
            "prior_extreme_rate",
            "reviewer_age_days",
            "prior_max_reviews_one_day"
        ]
    ],
    on=["user_id", "date"],
    how="left"
)

# ------------------------------------------------------------
# Business prior average rating
# ------------------------------------------------------------

# Again, reviews from the same date must NOT count as history.

business_day = (
    df.groupby(["prod_id", "date"])
    .agg(
        rating_sum_that_day=("rating", "sum"),
        reviews_that_day=("rating", "size")
    )
    .reset_index()
)

business_day["prior_business_rating_sum"] = (
    business_day.groupby("prod_id")["rating_sum_that_day"]
    .cumsum()
    - business_day["rating_sum_that_day"]
)

business_day["prior_business_review_count"] = (
    business_day.groupby("prod_id")["reviews_that_day"]
    .cumsum()
    - business_day["reviews_that_day"]
)

business_day["prior_business_avg_rating"] = np.where(
    business_day["prior_business_review_count"] > 0,
    business_day["prior_business_rating_sum"]
    / business_day["prior_business_review_count"],
    np.nan
)

df = df.merge(
    business_day[
        [
            "prod_id",
            "date",
            "prior_business_review_count",
            "prior_business_avg_rating"
        ]
    ],
    on=["prod_id", "date"],
    how="left"
)

# ------------------------------------------------------------
# Rating deviation from business history
# ------------------------------------------------------------

df["rating_deviation_from_business"] = (
    df["rating"] - df["prior_business_avg_rating"]
).abs()

# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------

print("\n--- Behavioral feature checks ---")

features = [
    "prior_review_count",
    "prior_extreme_rate",
    "reviewer_age_days",
    "prior_max_reviews_one_day",
    "prior_business_review_count",
    "prior_business_avg_rating",
    "rating_deviation_from_business"
]

print(df[features].describe())

print("\nSingle-review accounts:")

singleton = df[df["total_user_reviews"] == 1]

print(
    singleton[
        [
            "prior_review_count",
            "prior_extreme_rate",
            "reviewer_age_days",
            "prior_max_reviews_one_day"
        ]
    ].head(10)
)

print("\nPrior review count distribution:")
print(
    df["prior_review_count"]
    .value_counts()
    .sort_index()
    .head(10)
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

df.to_parquet(
    OUTPUT_FILE,
    index=False
)

print(f"\nSaved behavioral feature data to:")
print(OUTPUT_FILE)

print("\nDone.")
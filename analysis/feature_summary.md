# Feature Extraction Summary

## Dataset

We are using the YelpZip fake-review dataset.

The preprocessing pipeline loads:

- 608,598 reviews
- 260,277 reviewers
- 5,044 businesses

The YelpZip label is converted into:

- 0 = genuine review
- 1 = fake/filtered review

Dates are parsed and reviewer-history groups are created:

- 1 review
- 2 reviews
- 3-5 reviews
- 6-20 reviews
- 21+ reviews


## Preprocessing

File:

`src/preprocess.py`

This script:

- loads the YelpZip metadata
- loads the review text
- checks that row counts match
- parses review dates
- creates the fake/genuine label
- calculates each user's total review count
- creates reviewer-history groups
- calculates basic review word count
- saves the cleaned data as a Parquet file


## Text Features

File:

`src/text_features.py`

Features created:

### Review length

- number of words
- number of characters

### First-person pronoun rate

Measures how often first-person words such as:

- I
- me
- my
- we
- our
- us

appear relative to the total number of words.

### VADER sentiment

VADER produces a compound sentiment score from -1 to +1.

- negative values = negative sentiment
- values near 0 = neutral sentiment
- positive values = positive sentiment

### Sentiment/rating agreement

The star rating is converted into:

- 1-2 stars = negative
- 3 stars = neutral
- 4-5 stars = positive

The VADER sentiment score is also classified as negative, neutral, or positive.

The feature equals:

- 1 when the text sentiment agrees with the star rating
- 0 when the text sentiment does not agree with the star rating


## TF-IDF Text Features

File:

`src/tfidf_features.py`

TF-IDF uses:

- individual words (unigrams)
- two-word phrases (bigrams)

The vocabulary is limited to 20,000 features.

The data is divided into:

- 80% training
- 20% testing

The split is stratified jointly by:

- fake/genuine label
- reviewer-history group

This helps preserve the fake-review distribution inside each history group, including the smaller 21+ group.

TF-IDF is fitted only on the training data to prevent data leakage.

The saved sparse matrices are:

- `tfidf_train.npz`
- `tfidf_test.npz`

The fitted TF-IDF vectorizer is saved as:

- `tfidf_vectorizer.joblib`


## Reviewer Behavioral Features

File:

`src/behavior_features.py`

These features describe the reviewer.

Reviewer behavioral features use only reviews from dates before the target review.

Reviews from the same day do not count as prior history.

### Prior review count

Number of reviews previously written by the reviewer.

### Prior extreme-rating rate

Percentage of earlier reviewer ratings that were either:

- 1 star
- 5 stars

### Reviewer age

Number of days between the reviewer's first review and the target review.

### Maximum reviews in one prior day

Largest number of reviews the reviewer posted on any earlier day.

For a reviewer with no earlier history:

- prior review count = 0
- the other reviewer-history features remain missing (NaN)

This is intentional because missing reviewer history is part of the main research question.


## Business Features

File:

`src/behavior_features.py`

These features describe the restaurant rather than the reviewer.

They are calculated using only reviews from dates before the target review.

### Prior business review count

Number of reviews the business received before the target review.

### Prior business average rating

Average rating of the business using only earlier reviews.

### Rating deviation from business average

Absolute difference between the target review's rating and the business's previous average rating.

These business features are NOT included in the Part 1 standard baseline.

They are saved separately for Part 2, where the project tests whether business-side signals can help when reviewer history is unavailable.


## Reviewer History Groups

Reviewer-history groups are based on the reviewer's total number of reviews in the full dataset:

- 1
- 2
- 3-5
- 6-20
- 21+

These groups are used for evaluation and match the distribution analysis used in the project proposal.

They are different from `prior_review_count`.

`history_group` describes the reviewer's total activity in the dataset for evaluation purposes.

`prior_review_count` describes how much reviewer history was actually available before a particular review was posted.


## Missing Reviewer History

For a reviewer with no earlier reviews:

- prior review count = 0
- reviewer-history features are missing (NaN)

This is intentional.

The project is specifically studying what signals are available when reviewer history is missing or limited.

Business-side features can still exist even when reviewer-history features are unavailable.


## Data Leakage Protection

Time-dependent features only use reviews that happened before the target review.

Reviews from the same day are not treated as prior history.

The TF-IDF vocabulary is learned only from the training set.

The train/test split is created before fitting TF-IDF.

These rules help prevent future or test-set information from leaking into model features.


## Feature Groups for Modeling

### Part 1 Standard Baseline

The Part 1 baseline will use:

- text features
- reviewer behavioral features
- network features

Business-side features are excluded from the Part 1 baseline.

### Part 2 Business-Side Detector

The Part 2 detector will use business-side signals that remain available even when reviewer history is missing.

The business features already created are:

- prior business review count
- prior business average rating
- rating deviation from business average

Additional business-side features may be added later as part of Part 2.


## Current Status

Nathaniel's Weeks 6-7 tasks are complete:

- preprocessing pipeline
- text feature extraction
- VADER sentiment extraction
- TF-IDF unigrams and bigrams
- reviewer behavioral feature extraction
- strict date-cutoff logic
- train/test split stratified by label and reviewer-history group

Some business-side features were also created early and are being saved for Part 2.

The next phase is baseline classification and evaluation by reviewer-history group.

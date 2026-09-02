# Movie Recommendation System

A rating-based movie recommendation engine built to test whether content-based filtering actually beats a simple popularity baseline, applying and extending TF-IDF-based recommendation ideas first explored in an earlier project (HoPy club, 2023).

## Goal

Given a user's movie rating history, recommend unseen movies they're likely to enjoy — and, more importantly, quantify whether a content-based (TF-IDF) approach actually outperforms a naive popularity/random baseline, rather than assuming it does.

## Method

- **Data pipeline:** `dataSplitter.py` splits ratings into train / held-out test sets; `joiner.py` merges raw data sources; `testData.py` prepares the evaluation set; user interaction history is tracked via `user_seen_train.json` / `mapping_log.csv`.
- **Feature engineering:** `buildTfidf.py` builds TF-IDF vectors for movies from their metadata; `buildUserProf.py` aggregates the TF-IDF vectors of a user's highly-rated movies into a single user preference profile.
- **Recommendation strategies (compared head-to-head):**
  1. `rankNRetrieve_TFIDF.py` — content-based ranking by cosine similarity between a user's profile vector and candidate movie vectors.
  2. `rankNRetrieve_PopNRand.py` — popularity-ranked and random baselines, used as a control.
- **Evaluation:** `evaluator.py` scores each strategy's recommendations against each user's held-out actual ratings, with results written to `result/`.

## Demo

<!-- TODO (Shanga): fill in actual numbers from result/, e.g. -->
<!-- - Precision@K / Recall@K for TF-IDF vs. popularity baseline: -->
<!-- - Did TF-IDF meaningfully beat the baseline, and by how much: -->
<!-- - Any notable failure cases (e.g. cold-start users): -->

## Tech Stack

Python, TF-IDF vectorization (scikit-learn-style), user-profile-based content filtering

## Project Structure

```
├── ML_data/     # training / model data
├── result/      # evaluation output
├── *.py         # pipeline scripts (see Method above)
```

## Status

2026 personal project, applying TF-IDF recommendation concepts first implemented in an earlier HoPy club project (2023).


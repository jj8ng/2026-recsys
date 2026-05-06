import pandas as pd
import numpy as np
import os

# Optional plotting
import matplotlib.pyplot as plt

def preview_csv(path, n=5):
    print(f"File: {path}")
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("\nColumns:", list(df.columns))
    display = df.head(n)
    print(f"\nFirst {n} rows:")
    print(display.to_string(index=False))
    return df

def inspect_movies(movies_path, sample_n=5):
    df = preview_csv(movies_path, n=sample_n)
    # If movie dataset has separate fields, try to show plot field preview
    for col in ['plot', 'overview', 'description', 'summary']:
        if col in df.columns:
            print(f"\nSample of '{col}':")
            for i, txt in enumerate(df[col].dropna().astype(str).head(3), 1):
                print(f"  ({i}) {txt[:200]}{'...' if len(txt)>200 else ''}")
            break
    return df

def file_row_count(path, exclude_header=True):
    if not os.path.exists(path):
        return None
    try:
        # try pandas if installed (fast chunked read)
        import pandas as pd
        cnt = 0
        for chunk in pd.read_csv(path, chunksize=100000):
            cnt += len(chunk)
    except Exception:
        # fallback: count lines (fast enough)
        cnt = 0
        with open(path, 'rb') as f:
            for _ in f:
                cnt += 1
        if exclude_header and cnt>0:
            cnt -= 1  # remove header line
        return cnt
    if exclude_header and cnt>0:
        cnt -= 1
    return cnt

def inspect_ratings(ratings_path, show_hist=True, auto_suggest=True, suggest_percentile=0.75, manual_threshold=None):
    """
    Load ratings.csv, print schema and stats.
    If auto_suggest True: suggest threshold = percentile (default 75th).
    If manual_threshold provided, use that.
    Returns: df_ratings, suggested_threshold
    """
    df = preview_csv(ratings_path, n=5)
    if 'rating' not in df.columns:
        print("Warning: 'rating' column not found.")
        return df, None

    print("\nRating value counts:")
    print(df['rating'].value_counts().sort_index())

    print("\nRating descriptive stats:")
    print(df['rating'].describe())

    # percentiles
    pcts = df['rating'].quantile([0.25, 0.5, 0.75, 0.9]).to_dict()
    print("\nSelected percentiles:", {k: float(v) for k, v in pcts.items()})

    if show_hist:
        try:
            plt.figure(figsize=(6,3))
            df['rating'].hist(bins=20)
            plt.title("Ratings distribution")
            plt.xlabel("rating")
            plt.ylabel("count")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print("Plot failed:", e)

    suggested = None
    if manual_threshold is not None:
        suggested = manual_threshold
        print(f"\nUsing manual threshold = {suggested}")
    elif auto_suggest:
        # suggest percentile as threshold (e.g., top 25% ratings)
        suggested = float(df['rating'].quantile(suggest_percentile))
        print(f"\nAuto-suggest threshold at {int(suggest_percentile*100)}th percentile = {suggested}")
        # also suggest simple rule if discrete (e.g., if ratings are 1..5)
        unique_vals = sorted(df['rating'].unique())
        if all(float(x).is_integer() for x in unique_vals):
            # snap to nearest integer <= suggested
            suggested_int = int(np.floor(suggested))
            print(f"Discrete ratings detected. Suggest rating >= {suggested_int} as positive.")
            suggested = suggested_int

    else:
        print("\nNo threshold suggested (auto_suggest=False and no manual threshold).")

    return df, suggested

# Example usage:
if __name__ == "__main__":
    # Edit these paths as needed
    ratings_csv = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/2026-recsys_dataset/ratings.csv"
    movies_csv = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/2026-recsys_dataset/movies.csv"    # or plots CSV
    wiki_csv = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/2026-recsys_dataset/wiki_movie_plots_deduped.csv"
    item_table_csv = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/item_table.csv"

    movies_df = inspect_movies(item_table_csv)
    # ratings_df, threshold = inspect_ratings(ratings_csv, show_hist=True, auto_suggest=True)
    # print("Suggested threshold:", threshold)
    print(item_table_csv, file_row_count(item_table_csv))

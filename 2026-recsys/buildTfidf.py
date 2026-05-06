import os
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy import sparse

# CONFIG
ITEM_TABLE = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/ML_data/item_table.csv"
OUT_DIR   = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/ML_data"
VECT_PKL  = os.path.join(OUT_DIR, "tfidf_vectorizer.pkl")
ITEM_NPZ  = os.path.join(OUT_DIR, "item_matrix.npz")
MAP_CSV   = os.path.join(OUT_DIR, "itemid_row_map.csv")

# minimal text cleaning
def clean_text(s):
    if pd.isna(s):
        return ""
    t = str(s)
    t = t.lower()
    t = re.sub(r"<[^>]+>", " ", t)            # remove HTML
    t = re.sub(r"[\r\n]+", " ", t)            # newlines to space
    t = re.sub(r"[\t]+", " ", t)
    t = re.sub(r"[^\w\s'.,:-]", " ", t)       # keep some punctuation optionally
    t = re.sub(r"\s+", " ", t).strip()
    return t

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(ITEM_TABLE, low_memory=False)
    # detect plot column
    plot_col = None
    for c in ['plot','overview','description','story','plot_summary']:
        if c in df.columns:
            plot_col = c
            break
    if plot_col is None:
        raise RuntimeError("No plot column found in item_table.csv. Columns: " + ", ".join(df.columns))

    # keep only necessary columns and fillna
    df = df[['movieId','title',plot_col]].rename(columns={plot_col: 'plot'})
    df['plot'] = df['plot'].fillna("").astype(str)

    # clean text
    print("Cleaning text...")
    df['plot_clean'] = df['plot'].map(clean_text)

    texts = df['plot_clean'].tolist()
    print(f"Number of items: {len(texts)}")

    # fit TF-IDF
    print("Fitting TF-IDF vectorizer...")
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=20000, stop_words='english')
    A = vec.fit_transform(texts)   # sparse matrix (n_items, V)
    print("TF-IDF shape:", A.shape)

    # L2 normalize rows
    print("Normalizing item vectors (L2 rows)...")
    A_norm = normalize(A, norm='l2', axis=1)

    # save artifacts
    print("Saving vectorizer to:", VECT_PKL)
    with open(VECT_PKL, "wb") as f:
        pickle.dump(vec, f)

    print("Saving item matrix to:", ITEM_NPZ)
    sparse.save_npz(ITEM_NPZ, A_norm, compressed=True)

    # save mapping movieId -> row index
    print("Saving id->row map to:", MAP_CSV)
    df_map = df[['movieId','title']].copy()
    df_map['row_index'] = range(len(df_map))
    df_map.to_csv(MAP_CSV, index=False)

    print("Done.")

if __name__ == "__main__":
    main()

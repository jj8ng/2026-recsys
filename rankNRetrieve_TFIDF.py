import os
import json
import numpy as np
import pandas as pd
from scipy import sparse

# === PATHS - edit if needed ===
ITEM_MATRIX_NPZ = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/ML_data/item_matrix.npz"
USER_PROFILES_NPZ = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/ML_data/user_profiles.npz"
ITEM_MAP_CSV = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/ML_data/itemid_row_map.csv"
USER_MAP_CSV = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/ML_data/user_map.csv"
SEEN_JSON = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/user_seen_train.json"
OUT_RECS = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/result_baseline/recommendations.csv"

K = 10  # top-K

# === load artifacts ===
print("Loading item matrix:", ITEM_MATRIX_NPZ)
A = sparse.load_npz(ITEM_MATRIX_NPZ)         # shape (n_items, n_features) CSR
n_items = A.shape[0]

print("Loading user profiles:", USER_PROFILES_NPZ)
U = sparse.load_npz(USER_PROFILES_NPZ)       # shape (n_users, n_features) CSR
n_users = U.shape[0]
print("Item rows:", n_items, "User rows:", n_users)

print("Loading item map:", ITEM_MAP_CSV)
item_map = pd.read_csv(ITEM_MAP_CSV)         # expects movieId, title, row_index
# ensure integer row_index
item_map['row_index'] = item_map['row_index'].astype(int)
row_to_movie = dict(zip(item_map['row_index'], item_map['movieId']))
movieid_to_title = dict(zip(item_map['movieId'], item_map['title']))

print("Loading user map:", USER_MAP_CSV)
user_map = pd.read_csv(USER_MAP_CSV)         # expects userId, user_row_index
user_map['user_row_index'] = user_map['user_row_index'].astype(int)
row_to_user = dict(zip(user_map['user_row_index'], user_map['userId']))

print("Loading seen train JSON:", SEEN_JSON)
with open(SEEN_JSON, "r", encoding="utf-8") as f:
    seen_map = json.load(f)                  # keys are userId as strings -> list of movieIds

# convert seen movieIds -> row indices for faster masking
print("Converting seen movieIds -> row indices...")
movieid_to_row = dict(zip(item_map['movieId'], item_map['row_index']))
seen_row_map = {}
for user_str, mids in seen_map.items():
    rows = [movieid_to_row.get(int(m)) for m in mids if movieid_to_row.get(int(m)) is not None]
    seen_row_map[int(user_str)] = set(rows)

# === ranking loop ===
print("Ranking users, K =", K)
rows_out = []
for user_row in range(n_users):
    user_id = int(row_to_user[user_row])
    # user profile as dense vector (1D)
    u_vec = U[user_row]            # sparse 1 x D
    # compute scores: A (n_items x D) dot u_vec.T -> dense (n_items,)
    scores = A.dot(u_vec.T).toarray().ravel() if sparse.issparse(A) else A.dot(u_vec.T).ravel()
    # mask seen train items
    seen_rows = seen_row_map.get(user_id, set())
    if seen_rows:
        for r in seen_rows:
            if 0 <= r < n_items:
                scores[r] = -1e9

    # get top-K indices efficiently
    if K >= len(scores):
        topk_idx = np.argsort(scores)[::-1]
    else:
        # argpartition gives unordered top-K
        topk_idx = np.argpartition(scores, -K)[-K:]
        topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]

    # record results: rank 1..K
    for rank, idx in enumerate(topk_idx, start=1):
        movie_id = int(row_to_movie.get(idx, -1))
        title = movieid_to_title.get(movie_id, "")
        score = float(scores[idx])
        rows_out.append({
            "userId": user_id,
            "rank": rank,
            "movieId": movie_id,
            "title": title,
            "score": score
        })

# save to CSV
df_out = pd.DataFrame(rows_out)
df_out.to_csv(OUT_RECS, index=False)
print("Wrote recommendations to:", OUT_RECS)

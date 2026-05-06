import os
import pandas as pd
import numpy as np
from scipy import sparse
import pickle

# === PATHS (edit if needed) ===
ITEM_MATRIX_NPZ = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/ML_data/item_matrix.npz"
ITEM_MAP_CSV    = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/ML_data/itemid_row_map.csv"
TRAIN_CSV       = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/ML_data/interactions_train.csv"
OUT_DIR         = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/ML_data"
OUT_PROFILES    = os.path.join(OUT_DIR, "user_profiles.npz")   # CSR matrix
OUT_USER_MAP    = os.path.join(OUT_DIR, "user_map.csv")        # userId -> row index

# === load artifacts ===
print("Loading item matrix:", ITEM_MATRIX_NPZ)
A = sparse.load_npz(ITEM_MATRIX_NPZ)           # shape (n_items, n_features), CSR
print("Item matrix shape:", A.shape)

print("Loading item map:", ITEM_MAP_CSV)
item_map = pd.read_csv(ITEM_MAP_CSV)           # expects movieId,row_index (and title)
# ensure mapping from movieId -> row_index
if 'row_index' not in item_map.columns:
    # try common alternatives
    if 'index' in item_map.columns:
        item_map = item_map.rename(columns={'index':'row_index'})
if 'movieId' not in item_map.columns:
    raise RuntimeError("itemid_row_map.csv must contain movieId column")
movieid_to_row = {int(r.movieId): int(r.row_index) for _, r in item_map.iterrows()}

print("Loading train interactions:", TRAIN_CSV)
df = pd.read_csv(TRAIN_CSV)
# normalize column names if needed
if 'userId' not in df.columns:
    for c in df.columns:
        if c.lower() in ('userid','user_id'):
            df = df.rename(columns={c:'userId'})
if 'movieId' not in df.columns:
    for c in df.columns:
        if c.lower() in ('movieid','movie_id'):
            df = df.rename(columns={c:'movieId'})

if 'userId' not in df.columns or 'movieId' not in df.columns:
    raise RuntimeError("Train CSV must contain userId and movieId columns")

# group interactions by user and convert movieIds -> row indices (only keep known items)
print("Mapping train interactions to item row indices and grouping by user...")
user_groups = {}
missing = 0
for uid, group in df.groupby('userId'):
    movie_ids = group['movieId'].tolist()
    rows = [movieid_to_row.get(int(mid)) for mid in movie_ids]
    rows = [r for r in rows if r is not None]
    if len(rows) == 0:
        missing += 1
        continue
    user_groups[int(uid)] = rows

print(f"Users with train interactions (after mapping): {len(user_groups)}; users skipped (no mapped items): {missing}")

# Build user profile rows (sparse) by averaging item rows
print("Computing user profile rows (sparse mean of item vectors)...")
user_ids = sorted(user_groups.keys())
profile_rows = []
for uid in user_ids:
    idxs = user_groups[uid]
    # slice returns sparse matrix (len(idxs), n_features)
    sub = A[idxs, :]
    # compute mean along rows; result is 1 x n_features sparse matrix
    # .mean(axis=0) returns numpy.matrix; convert to csr
    mean_vec = sub.mean(axis=0)
    # mean_vec may be a numpy.matrix or sparse; ensure CSR 1xN
    if sparse.issparse(mean_vec):
        row = mean_vec.tocsr()
    else:
        # numpy matrix -> convert to csr
        arr = np.asarray(mean_vec).ravel()
        row = sparse.csr_matrix(arr)
    # ensure L2-normalize the row
    norm = sparse.linalg.norm(row)
    if norm > 0:
        row = row.multiply(1.0 / norm)
    profile_rows.append(row)

# Stack rows into a CSR matrix (n_users x n_features)
if len(profile_rows) == 0:
    raise RuntimeError("No user profiles computed (no mapped train interactions).")

print("Stacking user profile rows into matrix...")
user_profiles = sparse.vstack(profile_rows, format='csr')

print("User profiles shape:", user_profiles.shape)

# Save user_profiles as sparse NPZ and user map CSV
os.makedirs(OUT_DIR, exist_ok=True)
print("Saving user profiles to:", OUT_PROFILES)
sparse.save_npz(OUT_PROFILES, user_profiles, compressed=True)

# save user map: userId -> row index
user_map_df = pd.DataFrame({'userId': user_ids, 'user_row_index': list(range(len(user_ids)))})
user_map_df.to_csv(OUT_USER_MAP, index=False)
print("Saved user map to:", OUT_USER_MAP)

print("Done. Loaded", A.shape[0], "items; created profiles for", len(user_ids), "users.")

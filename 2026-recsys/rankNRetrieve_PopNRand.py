import json, numpy as np, pandas as pd

# === PATHS - edit ===
TRAIN_CSV    = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/ML_data/interactions_train.csv"
ITEM_MAP_CSV = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/ML_data/itemid_row_map.csv"
SEEN_JSON    = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/user_seen_train.json"
OUT_POP_CSV  = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/ML_data/baseline_popularity.csv"
OUT_RND_CSV  = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/result/recommendations_Random.csv"
K = 10

# === load ===
train = pd.read_csv(TRAIN_CSV)
item_map = pd.read_csv(ITEM_MAP_CSV)   # expects columns movieId,title,row_index
movieid_to_title = dict(zip(item_map['movieId'], item_map['title']))
all_movie_ids = item_map['movieId'].astype(int).tolist()

with open(SEEN_JSON, 'r', encoding='utf-8') as f:
    seen_map = json.load(f)  # keys are userId (strings) -> list of movieIds

# === Popularity baseline ===
pop_counts = train['movieId'].value_counts().to_dict()  # movieId -> count
# ensure all item ids present
for mid in all_movie_ids:
    pop_counts.setdefault(int(mid), 0)
# global ordered list by popularity desc (tie-break by movieId)
sorted_by_pop = [mid for mid, _ in sorted(pop_counts.items(), key=lambda x: (-x[1], x[0]))]

rows = []
for user_str, seen_list in seen_map.items():
    user = int(user_str)
    seen = set(int(x) for x in seen_list)
    cnt = 0
    for mid in sorted_by_pop:
        if mid in seen:
            continue
        cnt += 1
        rows.append({"userId": user, "rank": cnt, "movieId": int(mid),
                     "title": movieid_to_title.get(int(mid), ""), "score": int(pop_counts.get(int(mid),0))})
        if cnt >= K:
            break

pd.DataFrame(rows).to_csv(OUT_POP_CSV, index=False)
print("Wrote popularity baseline:", OUT_POP_CSV)

# === Random baseline ===
rng = np.random.default_rng(42)
rows = []
candidate_pool = all_movie_ids
for user_str, seen_list in seen_map.items():
    user = int(user_str)
    seen = set(int(x) for x in seen_list)
    avail = [m for m in candidate_pool if m not in seen]
    if not avail:
        continue
    picks = rng.choice(avail, size=min(K, len(avail)), replace=False)
    for rank, mid in enumerate(picks, start=1):
        rows.append({"userId": user, "rank": rank, "movieId": int(mid),
                     "title": movieid_to_title.get(int(mid), ""), "score": 0.0})

pd.DataFrame(rows).to_csv(OUT_RND_CSV, index=False)
print("Wrote random baseline:", OUT_RND_CSV)

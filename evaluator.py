import os, math
import pandas as pd
import numpy as np

# === EDIT PATHS ===
TEST_CSV = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/ML_data/interactions_test.csv"
RECS_FILES = {
    "random": "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/result/recommendations_Random.csv",
    "tfidf":  "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/result/recommendations_TFIDF.csv",
    # add "pop": "path/to/pop.csv" if needed
}
OUT_DIR = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/result"

K = 10

# === helpers ===
def dcg_at_k(relevances):
    """rel = list of 0/1 relevance in rank order (length <= K)."""
    return sum((2**r - 1) / math.log2(i+2) for i, r in enumerate(relevances))  # r is 0/1 -> 2^r-1==r

def ndcg_at_k(relevances, k):
    rel = relevances[:k]
    dcg = dcg_at_k(rel)
    # ideal DCG (best possible): sorted relevances (all 1s first)
    ideal = sorted(rel, reverse=True)
    idcg = dcg_at_k(ideal)
    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(relevances, k):
    rel = relevances[:k]
    return sum(rel) / k

# === load test ground truth ===
test_df = pd.read_csv(TEST_CSV)
# normalize column names
if 'userId' not in test_df.columns:
    for c in test_df.columns:
        if c.lower() in ('userid','user_id'):
            test_df = test_df.rename(columns={c:'userId'})
if 'movieId' not in test_df.columns:
    for c in test_df.columns:
        if c.lower() in ('movieid','movie_id'):
            test_df = test_df.rename(columns={c:'movieId'})

# For leave-one-out test, there is typically one test item per user. Build mapping: user -> set(test_items)
test_map = test_df.groupby('userId')['movieId'].apply(lambda s: set(s.tolist())).to_dict()

# Evaluate each recommendation file
results = {}
for name, rec_path in RECS_FILES.items():
    recs = pd.read_csv(rec_path)
    # ensure rank ordering
    recs = recs.sort_values(['userId','rank'])
    per_user = []
    users = sorted(recs['userId'].unique())
    for u in users:
        rec_u = recs[recs['userId']==u].sort_values('rank').head(K)
        rec_list = rec_u['movieId'].tolist()
        gt = test_map.get(u, set())
        # relevances: 1 if recommended item in gt else 0
        relevances = [1 if mid in gt else 0 for mid in rec_list]
        hit = 1 if any(relevances) else 0
        prec = precision_at_k(relevances, K)
        ndcg = ndcg_at_k(relevances, K)
        per_user.append({'userId': u, 'hit@10': hit, 'prec@10': prec, 'ndcg@10': ndcg})
    per_user_df = pd.DataFrame(per_user)
    # aggregate
    mean_hit = per_user_df['hit@10'].mean()
    std_hit  = per_user_df['hit@10'].std(ddof=0)
    mean_prec = per_user_df['prec@10'].mean()
    std_prec  = per_user_df['prec@10'].std(ddof=0)
    mean_ndcg = per_user_df['ndcg@10'].mean()
    std_ndcg  = per_user_df['ndcg@10'].std(ddof=0)
    results[name] = {
        'per_user': per_user_df,
        'agg': (mean_hit,std_hit, mean_prec,std_prec, mean_ndcg,std_ndcg)
    }
    # save per-user metrics
    per_user_df.to_csv(os.path.join(OUT_DIR, f"metrics_per_user_{name}.csv"), index=False)

# Print summary
print("Method\tHit@10 (mean±std)\tPrec@10 (mean±std)\tNDCG@10 (mean±std)")
for name, info in results.items():
    mh, sh, mp, sp, mn, sn = info['agg']
    print(f"{name}\t{mh:.4f}±{sh:.4f}\t{mp:.4f}±{sp:.4f}\t{mn:.4f}±{sn:.4f}")

# Optional: compare methods pairwise using mean difference (paired) or save CSVs for external tests

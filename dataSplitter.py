import os
import json
import pandas as pd

# CONFIG - edit if needed
RATINGS_CSV = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/2026-recsys_dataset/ratings.csv"
OUT_DIR = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys"
MIN_INTERACTIONS = 2  # keep users with >= this many interactions

# Output filenames
OUT_TRAIN = os.path.join(OUT_DIR, "interactions_train.csv")
OUT_VAL   = os.path.join(OUT_DIR, "interactions_val.csv")
OUT_TEST  = os.path.join(OUT_DIR, "interactions_test.csv")
OUT_SEEN  = os.path.join(OUT_DIR, "user_seen_train.json")  # dict: user -> list of train movieIds

def ensure_out_dir(d):
    os.makedirs(d, exist_ok=True)

def load_ratings(path):
    df = pd.read_csv(path)
    # normalize common column names
    cols = [c.lower() for c in df.columns]
    # try to find standard names
    mapping = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("user_id","userid","userId".lower()):
            mapping[c] = "userId"
        if lc in ("movie_id","movieid","movieId".lower()):
            mapping[c] = "movieId"
        if lc in ("rating",):
            mapping[c] = "rating"
        if lc in ("timestamp","time"):
            mapping[c] = "timestamp"
    df = df.rename(columns=mapping)
    # ensure required columns exist
    if not {"userId","movieId"}.issubset(df.columns):
        raise RuntimeError("ratings CSV must contain user and movie id columns (found: %s)" % list(df.columns))
    # parse timestamp if present
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s', errors='coerce')
        except Exception:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    else:
        # create artificial incremental timestamp per user if none exists
        df["timestamp"] = pd.NA
    return df

def split_leave_one_out(df):
    # filter users with at least MIN_INTERACTIONS
    counts = df.groupby("userId").size()
    good_users = counts[counts >= MIN_INTERACTIONS].index
    df = df[df["userId"].isin(good_users)].copy()
    # ensure timestamp order: if all NaT, use original order index
    if df["timestamp"].isna().all():
        df["_orig_order"] = range(len(df))
        df = df.sort_values(["userId", "_orig_order"])
    else:
        # sort by timestamp ascending, tie-breaker by original order
        df["_orig_order"] = range(len(df))
        df = df.sort_values(["userId", "timestamp", "_orig_order"])

    train_rows = []
    val_rows = []
    test_rows = []
    user_seen_train = {}

    for user, group in df.groupby("userId"):
        group = group.reset_index(drop=True)
        n = len(group)
        if n == 2:
            # train = first, test = last
            train = group.iloc[[0]]
            test  = group.iloc[[1]]
            val = pd.DataFrame(columns=group.columns)  # empty
        else:
            # n >= 3
            train = group.iloc[: n - 2]
            val   = group.iloc[[n - 2]]
            test  = group.iloc[[n - 1]]
        # append
        train_rows.append(train)
        val_rows.append(val)
        test_rows.append(test)
        # seen_train list for masking (movieIds in train)
        seen_ids = train["movieId"].tolist()
        user_seen_train[str(user)] = seen_ids

    train_df = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame(columns=df.columns)
    val_df   = pd.concat(val_rows, ignore_index=True)   if val_rows   else pd.DataFrame(columns=df.columns)
    test_df  = pd.concat(test_rows, ignore_index=True)  if test_rows  else pd.DataFrame(columns=df.columns)

    # drop helper column
    for d in (train_df, val_df, test_df):
        if "_orig_order" in d.columns:
            d.drop(columns=["_orig_order"], inplace=True)

    return train_df, val_df, test_df, user_seen_train

def save_outputs(train_df, val_df, test_df, user_seen_train):
    train_df.to_csv(OUT_TRAIN, index=False)
    val_df.to_csv(OUT_VAL, index=False)
    test_df.to_csv(OUT_TEST, index=False)
    with open(OUT_SEEN, "w", encoding="utf-8") as f:
        json.dump(user_seen_train, f, ensure_ascii=False, indent=2)
    print("Wrote:", OUT_TRAIN)
    print("Wrote:", OUT_VAL)
    print("Wrote:", OUT_TEST)
    print("Wrote:", OUT_SEEN)

def main():
    ensure_out_dir(OUT_DIR)
    print("Loading ratings from:", RATINGS_CSV)
    df = load_ratings(RATINGS_CSV)
    print("Total interactions:", len(df))
    train_df, val_df, test_df, user_seen_train = split_leave_one_out(df)
    print("Train rows:", len(train_df), "Val rows:", len(val_df), "Test rows:", len(test_df))
    save_outputs(train_df, val_df, test_df, user_seen_train)

if __name__ == "__main__":
    main()

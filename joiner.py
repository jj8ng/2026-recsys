import os
import pandas as pd
import re
from rapidfuzz import process, fuzz
from datetime import datetime

# === CONFIG (수정할 것) ===
MOVIES_CSV = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/2026-recsys_dataset/movies.csv"
PLOTS_CSV  = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/2026-recsys_dataset/wiki_movie_plots_deduped.csv"
OUT_ITEM_TABLE = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/item_table.csv"
OUT_MAPPING_LOG = "/Users/sowcel/Documents/EWHA/4-1/Recsys/2026-recsys/mapping_log.csv"

AUTO_THRESH = 90
AMBIG_LOW = 70
TOP_K = 5

# === helpers ===
def normalize_title(t):
    if pd.isna(t):
        return ""
    t = str(t).lower().strip()
    t = re.sub(r"[^\w\s']"," ", t)
    t = re.sub(r"\s+"," ", t).strip()
    return t

def extract_year_from_title(title):
    if pd.isna(title):
        return None
    m = re.search(r"\((\d{4})\)", str(title))
    if m:
        return int(m.group(1))
    return None

def safe_str(x):
    return "" if pd.isna(x) else str(x)

# === load files safely ===
assert os.path.exists(MOVIES_CSV), f"movies csv not found: {MOVIES_CSV}"
assert os.path.exists(PLOTS_CSV), f"plots csv not found: {PLOTS_CSV}"

movies = pd.read_csv(MOVIES_CSV, low_memory=False)
plots  = pd.read_csv(PLOTS_CSV, low_memory=False)

# identify plot title/text columns
plot_title_col = next((c for c in plots.columns if c.lower() in ['title','movie_title','name']), None)
plot_text_col  = next((c for c in plots.columns if c.lower() in ['plot','overview','description','story','plot_summary']), None)
plot_year_col  = next((c for c in plots.columns if c.lower() in ['year','release_year','movie_year','release']), None)

if plot_title_col is None:
    raise RuntimeError("No title-like column in plots CSV")
if plot_text_col is None:
    plots['__plot_text__'] = ""
    plot_text_col = '__plot_text__'

# normalize and extract
plots = plots.copy()
plots['plot_title_raw'] = plots[plot_title_col].astype(str)
plots['plot_title_norm'] = plots['plot_title_raw'].apply(normalize_title)
if plot_year_col:
    plots['plot_year'] = pd.to_numeric(plots[plot_year_col], errors='coerce').astype('Int64')
else:
    plots['plot_year'] = pd.NA

movies = movies.copy()
# common MovieLens column is movieId and title, genres
if 'movieId' not in movies.columns and 'movie_id' in movies.columns:
    movies.rename(columns={'movie_id':'movieId'}, inplace=True)
movies['movieId'] = movies['movieId']
movies['title_raw'] = movies['title'].astype(str)
movies['title_year'] = movies['title_raw'].apply(extract_year_from_title).astype('Int64')
movies['title_norm'] = movies['title_raw'].apply(lambda x: normalize_title(re.sub(r"\(\d{4}\)","", safe_str(x))))

# build lookup dicts for exact matches
plots_reset = plots.reset_index()
plots_by_title_year = { (r['plot_title_norm'], int(r['plot_year'])): r.name for _,r in plots_reset.iterrows() if pd.notna(r['plot_year']) }
plots_by_title = { r['plot_title_norm']: r.name for _,r in plots_reset.iterrows() }

# exact matching pass
matches = {}
unmatched_idxs = []
for idx,row in movies.iterrows():
    key_year = (row['title_norm'], int(row['title_year'])) if pd.notna(row['title_year']) else None
    matched_idx = None
    method = None
    score = None
    if key_year and key_year in plots_by_title_year:
        matched_idx = plots_by_title_year[key_year]
        method = 'exact_year'
        score = 100.0
    elif row['title_norm'] in plots_by_title:
        matched_idx = plots_by_title[row['title_norm']]
        method = 'exact_title'
        score = 100.0
    if matched_idx is not None:
        matches[idx] = (matched_idx, method, score)
    else:
        unmatched_idxs.append(idx)

# fuzzy matching for unmatched
choices = plots['plot_title_norm'].tolist()
choice_indices = plots.index.tolist()
fuzzy_results = {}
ambiguous = {}

for idx in unmatched_idxs:
    query = movies.at[idx,'title_norm']
    # rapidfuzz extract
    res = process.extract(query, choices, scorer=fuzz.WRatio, limit=TOP_K)
    # res = [(match_str, score, pos), ...]
    if not res:
        fuzzy_results[idx] = (None, 'no_candidates', 0.0)
        continue
    # compute final score with optional year boost
    best = None
    top_list = []
    for match_str, scr, pos in res:
        plot_idx = choice_indices[pos]
        plot_row = plots.loc[plot_idx]
        year_ok = False
        if pd.notna(movies.at[idx,'title_year']) and pd.notna(plot_row['plot_year']):
            year_ok = (int(movies.at[idx,'title_year']) == int(plot_row['plot_year']))
        final_score = scr + (5 if year_ok else 0)
        top_list.append((plot_idx, plot_row['plot_title_raw'], final_score, year_ok))
    top_list.sort(key=lambda x: x[2], reverse=True)
    best = top_list[0]
    if best[2] >= AUTO_THRESH:
        fuzzy_results[idx] = (best[0], 'fuzzy_auto', float(best[2]))
    elif best[2] >= AMBIG_LOW:
        ambiguous[idx] = top_list
        fuzzy_results[idx] = (best[0], 'fuzzy_ambiguous', float(best[2]))
    else:
        fuzzy_results[idx] = (None, 'no_good_match', float(best[2]))

# build mapping log entries
log_rows = []
for idx,row in movies.iterrows():
    movieId = row['movieId']
    m_title = row['title_raw']
    m_year = int(row['title_year']) if pd.notna(row['title_year']) else None

    matched_plot_idx = None
    method = None
    score = None
    status = 'unmatched'

    if idx in matches:
        matched_plot_idx, method, score = matches[idx]
        status = 'matched'
    else:
        fm = fuzzy_results.get(idx)
        if fm:
            if fm[1] in ('fuzzy_auto','fuzzy_ambiguous'):
                if fm[0] is not None:
                    matched_plot_idx = fm[0]
                    method = fm[1]
                    score = fm[2]
                    status = 'matched' if fm[1]=='fuzzy_auto' else 'ambiguous'
            else:
                method = fm[1]
                score = fm[2]
                status = 'unmatched'

    plot_title = plots.at[matched_plot_idx, 'plot_title_raw'] if matched_plot_idx is not None else None
    plot_year  = int(plots.at[matched_plot_idx, 'plot_year']) if (matched_plot_idx is not None and pd.notna(plots.at[matched_plot_idx,'plot_year'])) else None
    plot_text  = safe_str(plots.at[matched_plot_idx, plot_text_col]) if matched_plot_idx is not None else None

    log_rows.append({
        'movieId': movieId,
        'movie_title': m_title,
        'movie_year': m_year,
        'plot_index': int(matched_plot_idx) if matched_plot_idx is not None else None,
        'plot_title': plot_title,
        'plot_year': plot_year,
        'match_method': method,
        'match_score': score,
        'status': status,
        'timestamp': datetime.utcnow().isoformat()
    })

log_df = pd.DataFrame(log_rows)
log_df.to_csv(OUT_MAPPING_LOG, index=False)
print("Wrote mapping log:", OUT_MAPPING_LOG)

# build final item table for matched entries
matched = log_df[log_df['status'].isin(['matched','ambiguous'])]
items = []
for _,r in matched.iterrows():
    mrow = movies[movies['movieId']==r['movieId']].iloc[0]
    plot_row = plots.loc[int(r['plot_index'])] if pd.notna(r['plot_index']) else None
    items.append({
        'movieId': r['movieId'],
        'title': r['movie_title'],
        'year': r['movie_year'],
        'genres': safe_str(mrow['genres']) if 'genres' in mrow.index else "",
        'plot': safe_str(plot_row[plot_text_col]) if plot_row is not None else "",
        'plot_source_title': safe_str(r['plot_title']),
    })

item_df = pd.DataFrame(items)
item_df.to_csv(OUT_ITEM_TABLE, index=False)
print("Wrote item table:", OUT_ITEM_TABLE)

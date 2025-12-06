'''
python pre_processing.py
'''

import re
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from regression_task.regression_algo import pretrain_all_regression

# --- helper: unify posting id column name across files ---
def standardize_posting_id(df: pd.DataFrame) -> pd.DataFrame:
    if "posting_id" not in df.columns:
        for c in ["id", "job_id", "postingId", "postingID", "PostingID"]:
            if c in df.columns:
                df = df.rename(columns={c: "posting_id"})
                break
    return df

# Standard normalization for strings
def norm(s: str) -> str:
    _KEEP = re.compile(r"[^a-z0-9+#./\- ]")
    s = s.lower()
    s = _KEEP.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_lexicon(path: str, min_len=1, max_len=3) -> dict:
    df = pd.read_csv(path, low_memory=False)
    col = "Skill" if "Skill" in df.columns else df.columns[0] # Column with keywords
    skills = [norm(x) for x in df[col].dropna().astype(str)] # Text normalization
    skills = [s for s in skills if s and min_len <= len(s.split()) <= max_len] # extract skills that are up to max_len words long
    # bucket by phrase length for faster n-gram lookups
    # if the skill is three words, it's located in in key of 3 and inside that value list.
    buckets = defaultdict(set)
    for s in skills:
        buckets[len(s.split())].add(s)
    return dict(buckets)  # {1: {...}, 2: {...}, 3: {...}}

def extract_skills(text: str, lex_buckets: dict) -> set:
    t = norm(text)
    toks = t.split()
    found = set()
    # n is the number of words in the keywords (from min length to max length)
    for n, bucket in lex_buckets.items():
        if not bucket: 
            continue
        for i in range(len(toks) - n + 1):
            ph = " ".join(toks[i:i+n])
            if ph in bucket:
                found.add(ph)
    return found # found key words in job description



if __name__ == "__main__":

    postings_csv="datasets/archive/postings.csv"
    lexicon_csv="datasets/universal_skills_catalog.csv"

    df = pd.read_csv(postings_csv, low_memory=False)
    df = standardize_posting_id(df)

    # Show relavent columns
    df['listed_time'] = pd.to_datetime(df['listed_time'] / 1000, unit='s')
    df['expiry'] = pd.to_datetime(df['expiry'] / 1000, unit='s')
    df[df['normalized_salary'] < 0] = 0


    lex_buckets = build_lexicon(lexicon_csv, min_len=1, max_len=4)
    print('1) Buckets Built')

    # ---- Build skill -> posting_id set from the FULL filtered postings (no dedupe) ----
    # This ensures salary stats reflect actual postings carrying the skill.
    skill_to_posts = {}
    for r, row in df.iterrows():
        pid = row["posting_id"]
        text = str(row["description"]) if pd.notna(row["description"]) else ""
        skills_in_row = extract_skills(text, lex_buckets)
        for s in skills_in_row:
            if s not in skill_to_posts:
                skill_to_posts[s] = set()
            skill_to_posts[s].add(pid)
    print('2) Skills to Postrings relationships mapped')

    for key in skill_to_posts.keys():
        df[key] = 0
    print('3) Skill columns initialized')

    for key, pids in skill_to_posts.items():
        df.loc[df['posting_id'].isin(pids), key] = 1
    print('4) Skill columns assigned present values')
    
    df.to_csv('datasets/clean_job_postings.csv', index=False)

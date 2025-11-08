'''
python associative_task/pre_processing.py
'''
# https://www.geeksforgeeks.org/machine-learning/association-rule/
# Question to answer: 
# Which skill bundles co-occur within roles? (Associative)

import re
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

# --- helper: annualize/normalize salary fields robustly ---
def normalize_salary(sal_df: pd.DataFrame) -> pd.Series:
    """
    Returns a float Series with annualized salaries.
    Tries common schemas: [min/max], [annual_salary], or [salary + pay_period].
    """

    #s = pd.Series(np.nan, index=sal_df.index, dtype="float64")
    s = sal_df

    # midpoint if min/max present
    if {"min_salary", "max_salary"}.issubset(sal_df.columns):
        sal_df['salary'] = (
            pd.to_numeric(sal_df["min_salary"], errors="coerce")
            + pd.to_numeric(sal_df["max_salary"], errors="coerce")
        ) / 2

    if "med_salary" in sal_df.columns:
        med_salary = pd.to_numeric(sal_df["med_salary"], errors="coerce")
        sal_df.loc[med_salary.notna(), 'salary'] = med_salary[med_salary.notna()]

    # explicit annual if present
    if "annual_salary" in sal_df.columns:
        sal_df = pd.to_numeric(sal_df["annual_salary"], errors="coerce").fillna(s)

    # generic salary + pay period (hourly, weekly, monthly, yearly, etc.)
    if "salary" in sal_df.columns:
        base = pd.to_numeric(sal_df["salary"], errors="coerce")
        if "pay_period" in sal_df.columns:
            factor = sal_df["pay_period"].astype(str).str.lower().map({
                "hour": 2080, "hourly": 2080,
                "day": 260, "daily": 260,
                "week": 52, "weekly": 52,
                "biweekly": 26,
                "month": 12, "monthly": 12,
                "year": 1, "annual": 1, "annum": 1, "yearly": 1
            })
            sal_df.loc[factor.notna(), 'salary'] *= factor[factor.notna()]

    return sal_df

# --- helper: unify posting id column name across files ---
def standardize_posting_id(df: pd.DataFrame) -> pd.DataFrame:
    if "posting_id" not in df.columns:
        for c in ["id", "job_id", "postingId", "postingID", "PostingID"]:
            if c in df.columns:
                df = df.rename(columns={c: "posting_id"})
                break
    return df
        
_KEEP = re.compile(r"[^a-z0-9+#./\- ]")

# Standard normalization for strings
def norm(s: str) -> str:
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

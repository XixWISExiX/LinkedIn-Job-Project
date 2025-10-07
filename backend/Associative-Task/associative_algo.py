'''
python Associative-Task/associative_algo.py --config=Associative-Task/associative_args.yml
'''
# Association Rule Mining for Pairs

import argparse
import yaml
from typing import Any, Dict
from pre_processing import *

def mine_pairs_from_scratch(
    postings_csv="datasets/archive/postings.csv",
    lexicon_csv="datasets/universal_skills_catalog.csv",
    salaries_csv="datasets/archive/jobs/salaries.csv",
    company=None, title=None,
    min_support=0.03,            # fraction of postings
    min_confidence=0.35,
    top_k=100,
    max_phrase_len=3,
    min_skill_df=10              # keep skills that appear in >= this many postings
):

    # 1) Load & filter
    df = pd.read_csv(postings_csv, low_memory=False)
    df = standardize_posting_id(df)

    if company:
        df = df[df["company_name"] == company]
    if title:
        df = df[df["title"].astype(str).str.contains(title, case=False, na=False, regex=False)]
    if df.empty:
        return pd.DataFrame(columns=["antecedent","consequent","support","confidence","lift"])

    df_full = df.reset_index(drop=True)

    # Optional dedupe by description
    desc = df["description"].fillna("").astype(str)
    desc = desc.drop_duplicates().reset_index(drop=True)

    # 2) Build lexicon buckets
    lex_buckets = build_lexicon(lexicon_csv, min_len=1, max_len=max_phrase_len)

    # 3) First pass: document frequency to prune lexicon
    doc_skills = [] # list of sets for each row (object)
    df_counter = Counter()
    for d in desc:
        sk = extract_skills(d, lex_buckets)
        doc_skills.append(sk)
        for s in sk:
            df_counter[s] += 1 # incrementing the skills counter because it was found.

    # ---- Salary prep (load once) ----
    sal = pd.read_csv(salaries_csv, low_memory=False)
    sal = standardize_posting_id(sal)
    normalize_salary(sal)
    sal = sal[pd.to_numeric(sal["salary"], errors="coerce").notna()][["posting_id","salary"]]

    # ---- Build skill -> posting_id set from the FULL filtered postings (no dedupe) ----
    # This ensures salary stats reflect actual postings carrying the skill.
    skill_to_posts = {}
    for r, row in df_full.iterrows():
        pid = row["posting_id"]
        text = str(row["description"]) if pd.notna(row["description"]) else ""
        skills_in_row = extract_skills(text, lex_buckets)
        for s in skills_in_row:
            if s not in skill_to_posts:
                skill_to_posts[s] = set()
            skill_to_posts[s].add(pid)

    # prune rare skills
    keep = {s for s,c in df_counter.items() if c >= min_skill_df} # Filter the skills with the minimum skill amount
    if not keep:
        return pd.DataFrame(columns=["antecedent","consequent","support","confidence","lift"])

    # rebuild mapping after prune
    skills = sorted(keep) # alphabetical order
    idx = {s:i for i,s in enumerate(skills)}
    n_docs, n_skills = len(desc), len(skills)

    # 4) Build boolean matrix (n_docs × n_skills)
    M = np.zeros((n_docs, n_skills), dtype=np.uint8)
    for r, sk in enumerate(doc_skills):
        cols = [idx[s] for s in sk if s in keep]
        if cols:
            M[r, cols] = 1 # (object x skills) matrix initialization

    # 5) Compute counts
    col_counts = M.sum(axis=0).astype(np.int32)                # |A|
    co_counts = M.T.dot(M).astype(np.int32)                    # |A ∧ B|
    N = float(n_docs)
    min_sup_docs = max(1, int(np.ceil(min_support * N)))

    def skill_mean_salary(s):
        pids = skill_to_posts.get(s, set())
        if not pids:
            return np.nan, 0
        tmp = pd.DataFrame({"posting_id": list(pids)})
        merged = tmp.merge(sal, on="posting_id", how="inner")
        if merged.empty:
            return np.nan, 0
        return float(merged["salary"].mean()), len(merged)
    skill_salary_mean = {}
    skill_salary_n = {}
    for s in skills:
        m, n = skill_mean_salary(s)
        skill_salary_mean[s] = m
        skill_salary_n[s] = n

    # Overall salary baseline for this filtered universe (useful for premiums)
    all_sal = df_full[["posting_id"]].drop_duplicates().merge(sal, on="posting_id", how="inner")
    overall_mean_salary = float(all_sal["salary"].mean()) if not all_sal.empty else np.nan

    # 6) Build rules (pairs only) + attach salary stats
    rules = []
    for i in range(n_skills):
        ai = col_counts[i]
        if ai < min_sup_docs:
            continue
        for j in range(n_skills):
            if i == j:
                continue
            cij = int(co_counts[i, j])
            if cij < min_sup_docs:
                continue
            aj = col_counts[j]
            supp = cij / N
            conf = cij / ai if ai else 0.0          # A -> B
            if conf < min_confidence:
                continue
            lift = conf / (aj / N) if aj else 0.0

            A = skills[i]
            B = skills[j]

            # Pair salary: intersection of posting sets
            posts_A = skill_to_posts.get(A, set())
            posts_B = skill_to_posts.get(B, set())
            pair_posts = posts_A.intersection(posts_B)
            pair_mean = np.nan
            pair_n = 0
            if pair_posts:
                tmp = pd.DataFrame({"posting_id": list(pair_posts)})
                merged = tmp.merge(sal, on="posting_id", how="inner")
                if not merged.empty:
                    pair_mean = float(merged["salary"].mean())
                    pair_n = len(merged)

            rules.append((
                A, B, supp, conf, lift,
                skill_salary_mean.get(A, np.nan), skill_salary_n.get(A, 0),
                skill_salary_mean.get(B, np.nan), skill_salary_n.get(B, 0),
                pair_mean, pair_n,
                overall_mean_salary
            ))

    if not rules:
        return pd.DataFrame(columns=[
            "antecedent","consequent","support","confidence","lift",
            "antecedent_mean_salary","antecedent_n",
            "consequent_mean_salary","consequent_n",
            "pair_mean_salary","pair_n","overall_mean_salary"
        ])

    out = pd.DataFrame(rules, columns=[
        "antecedent","consequent","support","confidence","lift",
        "antecedent_mean_salary","antecedent_n",
        "consequent_mean_salary","consequent_n",
        "pair_mean_salary","pair_n","overall_mean_salary"
    ])

    # Nice percentage helpers
    out["support_pct"] = (out["support"] * 100).round(2)
    out["confidence_pct"] = (out["confidence"] * 100).round(2)

    # Optional premiums vs overall
    out["antecedent_premium_vs_overall"] = (out["antecedent_mean_salary"] - out["overall_mean_salary"]).round(2)
    out["consequent_premium_vs_overall"] = (out["consequent_mean_salary"] - out["overall_mean_salary"]).round(2)
    out["pair_premium_vs_overall"]       = (out["pair_mean_salary"] - out["overall_mean_salary"]).round(2)

    # Sort & top_k
    out = out.sort_values(["lift","confidence","support"], ascending=False).head(top_k)
    #out = out.sort_values(["pair_premium_vs_overall"], ascending=False).head(top_k)

    # Package like before
    cols_order = [
        "antecedent","consequent",
        "support","support_pct","confidence","confidence_pct","lift",
        "antecedent_mean_salary","antecedent_n","antecedent_premium_vs_overall",
        "consequent_mean_salary","consequent_n","consequent_premium_vs_overall",
        "pair_mean_salary","pair_n","pair_premium_vs_overall",
        "overall_mean_salary"
    ]
    out = out[cols_order]

    #print(out[['antecedent', 'antecedent_mean_salary', 'antecedent_n', 'antecedent_premium_vs_overall']])
    #print(out[['consequent', 'consequent_mean_salary', 'consequent_n', 'consequent_premium_vs_overall']])
    #print(out[['confidence', 'pair_mean_salary', 'pair_n', 'pair_premium_vs_overall', 'overall_mean_salary']])
    #e=1/0

    return out

def build_arg_parser() -> argparse.ArgumentParser:
    '''
    Creates what arguments to take in and use in the Assoiation Rule Mining Process.
    '''

    p = argparse.ArgumentParser(description="Association Rule Mining Data Parser",
                                fromfile_prefix_chars='@')

    # -- Filter Args -- 
    p.add_argument("--company-filter",  type=str, default=None, help="seed to put into distribution sampling")
    p.add_argument("--title-filter",  type=str, default=None, help="seed to put into distribution sampling")

    # -- Mining Args -- 
    p.add_argument("--min-support",  type=float, default=0.05, help="min support for association mining output")
    p.add_argument("--min-confidence",  type=float, default=0.4, help="min confidence for association mining output")
    p.add_argument("--top-k",  type=int, default=50, help="top k entries for association mining output")
    p.add_argument("--max-phrase-len",  type=int, default=4, help="max skill phrase length (words) for association mining output")
    p.add_argument("--min-skill-df",  type=int, default=5, help="min skill difference for association mining output")

    # -- Input Paths -- 
    p.add_argument("--job-postings-csv-in",  type=str, default="datasets/archive/postings.csv", help="path to read job postings CSV")
    p.add_argument("--skills-lexicon-csv-in",  type=str, default="datasets/universal_skills_catalog.csv", help="path to read skills lexicon CSV")
    p.add_argument("--salaries-csv-in",  type=str, default="datasets/archive/jobs/salaries.csv", help="path to read salaries CSV")

    # -- Output CSV Columns -- 
    p.add_argument("--out-csv-cols",  type=str, nargs='+', default=['antecedent', 'consequent', 'confidence', 'overall_mean_salary', 'pair_premium_vs_overall', 'antecedent_premium_vs_overall', 'antecedent_n'], help="path to save CSV output of associative algo analysis")

    # -- Output Path -- 
    p.add_argument("--out-csv",  type=str, default="datasets/output/association-output0.csv", help="path to save CSV output of associative pair analysis")

    # -- YAML Config --
    p.add_argument("--config", type=str, help="Path to YAML config", default=None)

    return p


def flatten(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def yaml_keys_to_arg_names(flat_cfg: Dict[str, Any]) -> Dict[str, Any]:
    # keep only the last segment and convert to argparse-style (underscores)
    out = {}
    for k, v in flat_cfg.items():
        key = k.split(".")[-1].replace("-", "_")
        out[key] = v
    return out


if __name__ == "__main__":
    parser = build_arg_parser()

    # YAML Parsing
    args, _ = parser.parse_known_args()

    # If YAML given, load it and set as parser defaults (so CLI still overrides)
    if args.config:
        with open(args.config, "r") as f:
            cfg_raw = yaml.safe_load(f) or {}
        cfg_flat  = flatten(cfg_raw)                  # e.g., "mining.min_support" -> 0.05
        cfg_mapped = yaml_keys_to_arg_names(cfg_flat) # e.g., "min_support" -> 0.05
        # filter keys to only those known by argparse
        known = {a.dest for a in parser._actions}
        cfg_defaults = {k: v for k, v in cfg_mapped.items() if k in known}
        parser.set_defaults(**cfg_defaults)

    args = parser.parse_args()

    company = args.company_filter
    title = args.title_filter

    res = mine_pairs_from_scratch(
        postings_csv=args.job_postings_csv_in,
        lexicon_csv=args.skills_lexicon_csv_in,
        salaries_csv=args.salaries_csv_in,
        company=company,
        title=title,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        top_k=args.top_k,
        max_phrase_len=args.max_phrase_len,
        min_skill_df=args.min_skill_df
    )

    print("Role:", title)
    print("Company:", company)
    print("Column Names:", res.columns)

    #print(res[['antecedent', 'consequent', 'confidence', 'overall_mean_salary', 'pair_premium_vs_overall', 'pair_n']].to_string(index=False))
    #print(res[['antecedent', 'consequent', 'confidence', 'overall_mean_salary', 'antecedent_premium_vs_overall', 'antecedent_n']].to_string(index=False))
    #print(res[['antecedent', 'consequent', 'confidence', 'overall_mean_salary', 'pair_premium_vs_overall', 'antecedent_premium_vs_overall', 'antecedent_n']].to_string(index=False))
    print("Association Pair Mining DataFrame:")
    print(res[args.out_csv_cols])

    res.to_csv(args.out_csv, index=False)
 

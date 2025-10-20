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

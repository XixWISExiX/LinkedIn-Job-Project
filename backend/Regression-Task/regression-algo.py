'''
python Regression-Task/associative_algo.py --config=Regression-Task/regression_args.yml
'''
# Regression algorithm for salary

import argparse
import yaml
from typing import Any, Dict
from pre_processing import *



def build_arg_parser() -> argparse.ArgumentParser:
    '''
    Creates what arguments to take in and use for the regression algorithm
    '''

    p = argparse.ArgumentParser(description="Regression Parser",
                                fromfile_prefix_chars='@')


    # -- Input Paths -- 
    p.add_argument("--salaries-csv-in",  type=str, default="datasets/archive/jobs/salaries.csv", help="path to read salaries CSV")

    # -- Output CSV Columns -- 
    p.add_argument("--out-csv-cols",  type=str, nargs='+', default=['antecedent', 'consequent', 'confidence', 'overall_mean_salary', 'pair_premium_vs_overall', 'antecedent_premium_vs_overall', 'antecedent_n'], help="path to save CSV output of regression algo analysis")

    # -- Output Path -- 
    p.add_argument("--out-csv",  type=str, default="datasets/output/regression-output0.csv", help="path to save CSV output of regression analysis")

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
        cfg_flat  = flatten(cfg_raw)              
        cfg_mapped = yaml_keys_to_arg_names(cfg_flat)
        # filter keys to only those known by argparse
        known = {a.dest for a in parser._actions}
        cfg_defaults = {k: v for k, v in cfg_mapped.items() if k in known}
        parser.set_defaults(**cfg_defaults)

    args = parser.parse_args()

    company = args.company_filter
    title = args.title_filter





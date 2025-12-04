'''
python Regression-Task/regression-algo.py --config=Regression-Task/regression_args.yml
'''
# Regression algorithm for salary

import numpy as np
from pre_processing import *
from sklearn.metrics import r2_score


def perform_regression(career_data, salary_data):
    # --- Compute coefficients using the Normal Equation ---
    # β = (XᵀX)⁻¹ Xᵀy
    beta = np.linalg.pinv(career_data.T @ career_data) @ (career_data.T @ salary_data)

    return beta


def predict_salary(coefficient, x_values):

    y_predictions = x_values @ coefficient
    return y_predictions   

def perform_r2_score(y_actual, y_predicted):

    r2 = r2_score(y_actual, y_predicted)
    return r2

def mean_squared_error(y_actual, y_predicted):

    return np.mean((y_predicted - y_actual)**2)



'''
def build_arg_parser() -> argparse.ArgumentParser:
    
    #Creates what arguments to take in and use for the regression algorithm
    

    p = argparse.ArgumentParser(description="Regression Parser",
                                fromfile_prefix_chars='@')


    # -- Input Paths -- 
    p.add_argument("--salaries-csv-in",  type=str, default="datasets/archive/jobs/salaries.csv", help="path to read salaries CSV")

    # -- Output CSV Columns -- 
    p.add_argument("--out-csv-cols",  type=str, nargs='+', default=['salary_mean_squared_error', 'salary_r2_score', 'salary_explained_variance_score', 'salary_max_error', 'salary_confusion_matrix', 'salary_f1_score'], help="path to save CSV output of regression algo analysis")

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
'''









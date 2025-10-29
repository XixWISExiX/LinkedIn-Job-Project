'''
python Regression-Task/associative_algo.py --config=Regression-Task/regression_args.yml
'''
# Regression algorithm for salary

import argparse
import yaml
import pandas as pd
import numpy as np
from typing import Any, Dict
from pre_processing import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, max_error,confusion_matrix, f1_score

# --- Load the dataset ---
salary_data = pd.read_csv("datasets/archive/jobs/salaries.csv")


# --- Preprocess categorical data ---
# Convert text columns (like job_title, experience_level, company_size) to numbers
label_encoders = {}
for col in ['experience_level', 'employment_type', 'job_title', 'company_size']:
    le = LabelEncoder()
    salary_data[col] = le.fit_transform(salary_data[col])
    label_encoders[col] = le  # store encoders if you need to decode later

# --- Define features (X) and target (y) ---
X = salary_data[['experience_level', 'employment_type', 'job_title', 'company_size']]
y = salary_data['salary_in_usd']

# --- Split into training and testing sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Add a column of ones to X_train for the intercept term ---
X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]  # add bias column

# --- Compute coefficients using the Normal Equation ---
# β = (XᵀX)⁻¹ Xᵀy
beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_train

# --- Predict on test data ---
y_pred = X_test @ beta


# --- Store the model evaluations ---
#print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
#print("R² Score:", r2_score(y_test, y_pred))
salary_mean_squared_error = mean_squared_error(y_test, y_pred)
salary_r2_score = r2_score(y_test, y_pred)
salary_explained_variance_score = explained_variance_score(y_test, y_pred)
salary_max_error = max_error(y_test, y_pred)
salary_confusion_matrix = confusion_matrix(y_test, y_pred)
salary_f1_score = f1_score(y_test, y_pred)




def build_arg_parser() -> argparse.ArgumentParser:
    '''
    Creates what arguments to take in and use for the regression algorithm
    '''

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









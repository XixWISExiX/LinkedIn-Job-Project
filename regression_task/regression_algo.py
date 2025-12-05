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

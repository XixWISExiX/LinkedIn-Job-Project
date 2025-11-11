import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from regression_task.regression_algo import run_regression_algorithm

st.title("Salary Predictor — Estimate Pay from Role, Company & Skills")

# NOTE: This part doesn't need to be in the final dashboard, just to help your understanding.
#st.caption("Predict a salary using job title, company, skills, and ZIP code, with confidence intervals and key drivers.")
#st.markdown("""
#**What this does**
#- Trains/loads a regression model to estimate base salary.
#- Explains drivers (feature importance / partial effects) and shows model fit (MAE, R², residuals).
#- Lets you compare scenarios by tweaking inputs.

#**Great for**
#- Benchmarking offers or expectations.
#- Testing which skills shift pay the most.
#- Exploring location/company effects by ZIP and employer.

#*Note:* This is an estimate from historical postings—actual compensation may vary.
#""")

col_1, col_2, col_3 = st.columns([4, 4, 4], gap="small")  # ≈ 33% / 33% / 33%



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
#from regression_task.regression_algo import run_regression_algorithm

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

# page set up for regression
st.set_page_config(page_title="Job Regression Dashboard", layout="wide")
st.title("Salary Predictor — Salary Regression Explorer")

tabs = st.tabs(
    ["Input Parameters", "Scatter Plot", "Regression Summary"]
)


with tabs[0]:
    # set title of the tab.
    st.header("Regression Inputs")
    col1, col2, col3, col4 = st.columns(4) # Set up 4 columns for each input.

    # Job skill for column 1.
    with col1:
        skill = st.selectbox("Type of Skill", ["Python", "C++", "Java"])

    # Company for column 2.
    with col2:
        company = st.selectbox("Company Name", ["Boeing", "Northrop Grumman", "Microsoft"])

    # Job title for column 3.
    with col3:
        job_title = st.selectbox("Job Title", ["Manager", "Software Tester", "Quality Control"])

    # Zip code for column 4.
    with col4:
        zip_code = st.number_input("Zip Code", 11111, 99999, 73019, step=1)

with tabs[1]:
    st.header("Job Data Scatterplot")

with tabs[2]:
    st.header("Regression Summary")


    
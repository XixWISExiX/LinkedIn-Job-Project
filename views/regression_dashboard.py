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

# Get all skills and get the entire dataframe for all the skills.
skill_cols = set(st.session_state.job_market_filtered_df.columns[31:])
skill_cols.add('normalized_salary')
skill_df = st.session_state.job_market_filtered_df.loc[:, st.session_state.job_market_filtered_df.columns.isin(skill_cols)]

# Get dataframe of the entire dataset
entire_df = st.session_state.job_market_filtered_df


# Extract the columns for the regression.
job_title_column = entire_df['title']
company_column = entire_df['company_name']
zip_code_column = entire_df['zip_code']
job_location_column = entire_df['location']
normalized_salary_column = entire_df['normalized_salary']
max_salary_column = entire_df['max_salary']
med_salary_column = entire_df['med_salary']
min_salary_column = entire_df['min_salary']

# Get a unique list of the job title, company, zip code, and the location for the user input.
job_title_list = list(set(job_title_column.tolist()))
company_list = list(set(company_column.tolist()))
zip_code_list = list(set(zip_code_column.tolist()))
job_location_list = list(set(job_location_column.tolist()))



# page set up for regression
st.set_page_config(page_title="Job Regression Dashboard", layout="wide")
st.title("Salary Predictor — Salary Regression Explorer")

tabs = st.tabs(
    ["Input Parameters", "Scatter Plot", "Regression Summary"]
)


with tabs[0]:
    # set title of the tab.
    st.header("Regression Inputs")
    col1, col2, col3, col4, col5 = st.columns(5) # Set up 4 columns for each input.

    # Job skill for column 1.
    with col1:
        skill_input = st.selectbox("Type of Skill", sorted(skill_cols))

    # Company for column 2.
    with col2:
        company_input = st.selectbox("Company Name", company_list)

    # Job title for column 3.
    with col3:
        job_title_input = st.selectbox("Job Title", sorted(job_title_list))

    # Zip code for column 4.
    with col4:
        zip_code_input = st.selectbox("Zip Code", sorted(zip_code_list))

    # location for column 5.
    with col5:
        location_input = st.selectbox("Location", sorted(job_location_list))

with tabs[1]:
    st.header("Job Data Scatterplot")

with tabs[2]:
    st.header("Regression Summary")


    
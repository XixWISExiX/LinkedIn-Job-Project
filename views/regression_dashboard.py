import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import math
#from regression_task.regression_algo import run_regression_algorithm

st.title("Salary Predictor — Estimate Pay from Skills, Company, and location")

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

# Extract the following columns for the inputs and the X regression values.
company_column = entire_df['company_name']
job_location_column = entire_df['location']
job_title_column = entire_df['title']

# Extract the following for the Y regressions.
normalized_salary_column = entire_df['normalized_salary']

# Get a unique list of the company, the title and the location for the user input.
company_list = list(set(company_column.tolist()))
job_location_list = list(set(job_location_column.tolist()))
job_title_list = list(set(job_title_column.tolist()))

# remove any floating point NaN's.
company_list = [item for item in company_list if not (isinstance(item, float) and math.isnan(item))]
job_location_list = [item for item in job_location_list if not (isinstance(item, float) and math.isnan(item))]
job_title_list = [item for item in job_title_list if not (isinstance(item, float) and math.isnan(item))]


# page set up for regression
st.set_page_config(page_title="Job Regression Dashboard", layout="wide")
st.title("Salary Predictor — Salary Regression Explorer")

tabs = st.tabs(
    ["Input Parameters", "Scatter Plot", "Regression Summary"]
)


with tabs[0]:
    # set title of the tab.
    st.header("Regression Inputs")
    col1, col2, col3, col4 = st.columns(4) # Set up 3 columns for each input.

    # Job skill for column 1.
    with col1:
        skill_input = st.multiselect("Type of Skill", sorted(skill_cols), help = "Pick one or more skills via dropdown or typing. " \
        "Select the x in the red selection to delete a selection. Press the x next to the dropdown to delete the whole selection ")

    # Company for column 2.
    with col2:
        company_input = st.selectbox("Company Name", sorted(company_list), help = "Click the dropdown box or type to find the company.")

    # location for column 3.
    with col3:
        location_input = st.selectbox("Location", sorted(job_location_list), help = 
                                      "Click the dropdown box or type to find the job location.")
        
    # title for column 4.
    with col3:
        title_input = st.selectbox("Title", sorted(job_title_list), help = 
                                      "Click the dropdown box or type to find the job title.")
        

with tabs[1]:
    st.header("Job Data Scatterplot")

with tabs[2]:
    st.header("Regression Summary")


    
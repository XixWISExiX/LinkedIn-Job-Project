import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import math
from regression_task.regression_algo import perform_regression, predict_salary, mean_squared_error, perform_r2_score
from sklearn.preprocessing import StandardScaler
import altair as alt

st.title("Salary Predictor — Estimate Pay from Skills, Company, and location")

# NOTE: This part doesn't need to be in the final dashboard, just to help your understanding.
#st.caption("Predict a salary using job title, company, location skills, with confidence intervals and key drivers.")
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

# Get all skills and get the entire dataframe for all the skills and the required variables for regression.
regression_columns = set(st.session_state.job_market_filtered_df.columns[31:])
regression_columns.add('company_name')
regression_columns.add('location')
regression_columns.add('title')
regression_columns.add('normalized_salary')
regression_df = st.session_state.job_market_filtered_df.loc[:, st.session_state.job_market_filtered_df.columns.isin(regression_columns)]

# Extract the following columns for the inputs.
skill_selection = set(st.session_state.job_market_filtered_df.columns[31:])
company_column = regression_df['company_name']
job_location_column = regression_df['location']
job_title_column = regression_df['title']


# Clean df by dropping rows that have empty values. Use this for the scatterplots also.
regression_df_cleaned = regression_df.dropna()

# Make the x values for regression
x_values = regression_df_cleaned[["company_name", "location", "title"]]

# Extract the following for the Y regressions.
y_values = regression_df_cleaned[['normalized_salary']]

# intiialized and prepare the encoder for regression and predictions.
data_encoder = {}

for column in x_values.columns:
    x_values[column], catagories = pd.factorize(x_values[column]) #encode each column
    data_encoder[column] = catagories  # Store the mappings

# Addd column for the intercept.
X_matrix = np.c_[np.ones(len(x_values)), x_values.values]

y_vector = y_values.values.reshape(-1, 1)


# Calculate the regression slops using the perform_regression from the regression_algo.py
beta = perform_regression(X_matrix, y_vector)

# Obtain the slope and the intercept.
intercept = beta[0][0]
coefficients = beta[1:].flatten()

y_pred = predict_salary(beta, X_matrix)

mse = mean_squared_error(y_vector, y_pred)

r2 = perform_r2_score(y_vector, y_pred)


# Get a unique list of the company, the title and the location for the user input.
company_list = list(set(company_column.tolist()))
job_location_list = list(set(job_location_column.tolist()))
job_title_list = list(set(job_title_column.tolist()))

# remove any floating point NaN's.
company_selection = [item for item in company_list if not (isinstance(item, float) and math.isnan(item))]
job_location_selection = [item for item in job_location_list if not (isinstance(item, float) and math.isnan(item))]
job_title_selection = [item for item in job_title_list if not (isinstance(item, float) and math.isnan(item))]


# page set up for regression
st.set_page_config(page_title="Job Regression Dashboard", layout="wide")
st.title("Salary Predictor — Salary Regression Explorer")

tabs = st.tabs(
    ["Input Parameters", "Scatter Plots", "Regression Summary"]
)


with tabs[0]:
    # set title of the tab.
    st.header("Regression Inputs")
    col1, col2, col3, col4 = st.columns(4) # Set up 3 columns for each input.

    # Job skill for column 1.
    with col1:
        skill_inputs = st.multiselect("Type of Skill", sorted(skill_selection), help = "Pick one or more skills via dropdown or typing. " \
        "Select the x in the red selection to delete a selection. Press the x next to the dropdown to delete the whole selection ")

    # Company for column 2.
    with col2:
        company_input = st.selectbox("Company Name", sorted(company_selection), help = "Click the dropdown box or type to find the company.")

    # location for column 3.
    with col3:
        location_input = st.selectbox("Location", sorted(job_location_selection), help = 
                                      "Click the dropdown box or type to find the job location.")
        
    # title for column 4.
    with col4:
        title_input = st.selectbox("Title", sorted(job_title_selection), help = 
                                      "Click the dropdown box or type to find the job title.")
        
    

with tabs[1]:
    st.header("Job Data Scatterplots")

    scatter_graph_df = regression_df
    
    scatter_graph_df["Company_name-Title-Location"] = scatter_graph_df.apply(lambda x: f'{x["company_name"]} {x["title"]} {x["location"]}', 
                                                                             axis = 1)

    chart = (alt.Chart(scatter_graph_df).mark_point().encode(x = alt.X("Company_name-Title-Location:O", 
                                                                       title = "Company_name-Title-Location"), 
                                                                       y = "normalized_salary:Q", 
                                                                       tooltip = ["company_name", 
                                                                                  "title", "location", "normalized_salary"])
                                                                       .properties(title = "Scattered plot of salary over the " \
                                                                       "Company Name, Title, and Location tuple."))
    
    st.altair_chart(chart, use_container_width = True)
    



with tabs[2]:
    st.header("Regression Summary")

    col_a, col_b, col_c = st.columns(3)

    with col_a: 
        st.header("Regression Formaula")
        st.write("Intercept of working regression dataset (b0):", round(intercept, 2))
        st.write("Coefficients of working regression dataset ordered from 'company_name', 'location', and 'job_title' :", coefficients)

    with col_b:
        st.header("Mean Squared Error of regression vs dataset")
        st.write("MSE of working regression dataset:", round(mse, 2))

    with col_c:
        st.header("R2 score of the dataset")
        st.write("R2 score of the working dataset:", round(r2, 2))

    
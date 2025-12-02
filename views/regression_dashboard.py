import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from regression_task.regression_algo import perform_regression, predict_salary, mean_squared_error, perform_r2_score
import altair as alt

st.title("Salary Predictor — Estimate Pay from Skills, Company, and location")

# Get all skills and get the entire dataframe for all the skills and the required variables for regression.
regression_columns = set(st.session_state.job_market_filtered_df.columns[31:])
regression_columns.add('company_name')
regression_columns.add('location')
regression_columns.add('title')
regression_columns.add('normalized_salary')
regression_df = st.session_state.job_market_filtered_df.loc[:, st.session_state.job_market_filtered_df.columns.isin(regression_columns)]

# Extract the the columns containing the skills for the skills input.
skill_selection = set(st.session_state.job_market_filtered_df.columns[31:])

# Clean df by dropping rows that have empty values. Use this for the scatterplots also.
regression_df_cleaned = regression_df.dropna()

# Also sort the cleaned dataframe to give a better scatter and regression.
regression_df_cleaned = regression_df_cleaned.sort_values(by = 'normalized_salary')

# Extract the x values for regression
x_values = regression_df_cleaned[["company_name", "location", "title"]]

# Extract the y calues for regression
y_values = regression_df_cleaned[['normalized_salary']]

# intiialized and prepare the encoder for regression and predictions.
data_encoder = {}

for column in x_values.columns:
    x_values[column], categories = pd.factorize(x_values[column]) #encode each column
    data_encoder[column] = categories  # Store the mappings

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


# page set up for regression
st.set_page_config(page_title="Job Regression Dashboard", layout="wide")
st.title("Salary Predictor — Salary Regression Explorer")

# Add each tab for each topic of the regression section.
tabs = st.tabs(
    ["Input Parameters to output the predicted salary", "Bar Plots of average normalized salaries with filtering", 
     "Regression Summary for the entire dataset", "Top-K Job Companies, Job Locations, Job Titles, and Job Skills"]
)


with tabs[0]:
    # set title of the tab.
    st.header("Regression Inputs and the processed output")

    skill_inputs = st.multiselect("Types of Skills", sorted(skill_selection), default = [], placeholder= "All skills", 
                                  help = "Select one or more skills to filter via dropdown or typing. " \
    "Select the x in the red selection to delete a filter. Press the x next to the dropdown to delete all filters")

    if not skill_inputs:
        filtered_df = regression_df

    else:
        filtered_df = regression_df[regression_df[skill_inputs].all(axis = 1)]

    # If the filtered skills results in an empty data, user cannot impurt jobs to find the salary!!
    if (filtered_df.empty):
        st.error("Sorry!! I don't have any jobs that match your skills!!")
    else:
        # Get all the comanies, locations, and titles for user inputs.
        company_column = filtered_df['company_name']
        job_location_column = filtered_df['location']
        job_title_column = filtered_df['title']

        # Get a unique list of the company, the title and the location for the user input.
        company_list = list(set(company_column.tolist()))
        job_location_list = list(set(job_location_column.tolist()))
        job_title_list = list(set(job_title_column.tolist()))

        # remove any floating point NaN's.
        company_selection = [item for item in company_list if not (isinstance(item, float) and math.isnan(item))]
        job_location_selection = [item for item in job_location_list if not (isinstance(item, float) and math.isnan(item))]
        job_title_selection = [item for item in job_title_list if not (isinstance(item, float) and math.isnan(item))]

        col1, col2, col3 = st.columns(3) # Set up 3 columns for the company, job title, and location after the user picks the skills.

        # Company for column 2.
        with col1:
            company_input = st.selectbox("Company Name", sorted(company_selection), index = None, placeholder = "Select a company...",
                                            help = "Click the dropdown box or type to find the company.")

        # location for column 3.
        with col2:
            location_input = st.selectbox("Location", sorted(job_location_selection), 
                                            index = None, placeholder = "Select a location...", help = 
                                    "Click the dropdown box or type to find the job location.")
    
        # title for column 4.
        with col3:
            title_input = st.selectbox("Title", sorted(job_title_selection), index = None, 
                                        placeholder = "Select a job title...", help = 
                                    "Click the dropdown box or type to find the job title.")
    
        # All inputs must be selected to calculate the predicted salary. 
        if not company_input or not location_input or not title_input:
            st.warning("Please select a company, a location, and a job title and we will give you an estimated salary.")

        # You have the inputs, then create a new dataframe to predict a new salary.
        else:
            new_data = {
                'company_name': company_input,
                'location': location_input,
                'title': title_input
            }

            values_encoded = []

            for column in ['company_name', 'location', 'title']:
                ctgories = data_encoder[column]
                if new_data[column] in ctgories:
                    indx = np.where(ctgories == new_data[column])[0][0]
                else:
                    indx = -1

                values_encoded.append(indx)

            new_x = np.array([1] + values_encoded)

            estimated_salary = predict_salary(beta, new_x)

            st.write("Predicted normalized salary for these inputs:", round(float(estimated_salary), 2))



with tabs[1]:
    st.header("Job Data Barplot with filtering options")

    st.subheader("Filtering Options:")

    # use the regression dataframe to perform the graph plotting. We will also used this for the scatterplots.
    scatter_graph_df = regression_df_cleaned
    
    company_scatter_column = scatter_graph_df['company_name']
    job_location_scatter_column = scatter_graph_df['location']
    job_title_scatter_column = scatter_graph_df['title']
    
    # Get a unique list of the company, the title and the location for the user input.
    company_scatter_list = list(set(company_scatter_column.tolist()))
    job_location_scatter_list = list(set(job_location_scatter_column.tolist()))
    job_title_scatter_list = list(set(job_title_scatter_column.tolist()))

    # remove any floating point NaN's.
    company_scatter_selection = [item for item in company_scatter_list if not (isinstance(item, float) and math.isnan(item))]
    job_location_scatter_selection = [item for item in job_location_scatter_list if not (isinstance(item, float) and math.isnan(item))]
    job_title_scatter_selection = [item for item in job_title_scatter_list if not (isinstance(item, float) and math.isnan(item))]
        
    # Define columns
    col_a, col_b, col_c, col_d = st.columns(4)

    # Multiselect to filter out the barplots of job companies.
    with col_a:
        selected_job_companies = st.multiselect("Job companies", sorted(company_scatter_selection), default = [], 
                                                placeholder= "All companies", 
                                  help = "Select one or more companies to filter via dropdown or typing. " \
        "Select the x in the red selection to delete a comapny filter. Press the x next to the dropdown to delete all companies filters")

    # Multiselect to filter out the barplots of job locations.
    with col_b: 
        selected_job_locations = st.multiselect("Job locations", sorted(job_location_scatter_selection), default = [], 
                                                placeholder= "All locations", 
                                  help = "Select one or more locations to filter via dropdown or typing. " \
        "Select the x in the red selection to delete a location filter. Press the x next to the dropdown to delete all locations filters")

    # Multiselect to filter out the barplots of job titles
    with col_c:
        selected_job_titles = st.multiselect("Job titles", sorted(job_title_scatter_selection), default = [], placeholder= "All titles", 
                                  help = "Select one or more titles to filter via dropdown or typing. " \
        "Select the x in the red selection to delete a title filter. Press the x next to the dropdown to delete all titles filters")

    # Multiselect to filter out the barplots of skills
    with col_d:
        selected_job_skills = st.multiselect("Types of Skills", sorted(skill_selection), default = [], placeholder= "All skills", 
                                  help = "Select one or more skills to filter via dropdown or typing. " \
        "Select the x in the red selection to delete a skill filter. Press the x next to the dropdown to delete all skills filters")


    # apply filtering if selected.
    filtered_scatter_graph_df = scatter_graph_df
    if selected_job_companies:
        filtered_scatter_graph_df = filtered_scatter_graph_df[filtered_scatter_graph_df['company_name'].isin(selected_job_companies)]
    if selected_job_locations:
        filtered_scatter_graph_df = filtered_scatter_graph_df[filtered_scatter_graph_df['location'].isin(selected_job_locations)]
    if selected_job_titles:
        filtered_scatter_graph_df = filtered_scatter_graph_df[filtered_scatter_graph_df['title'].isin(selected_job_titles)]
    if selected_job_skills:
        filtered_scatter_graph_df = filtered_scatter_graph_df[filtered_scatter_graph_df[selected_job_skills].all(axis = 1)]

    # Display Dataframe and barplot.
    st.subheader("Filtered Dataset and barplot")

    # If the filering results in an empty dataframe, we display error to user about no data exists.
    if filtered_scatter_graph_df.empty:
        st.error("Cannot generate datatable and barplot for the selected filters due to an empty dataset!!")

    else:

        st.dataframe(filtered_scatter_graph_df, use_container_width=True)

       
        # Barplot on average normal salary over company name.
        
        plot_data = filtered_scatter_graph_df.groupby('company_name')['normalized_salary'].mean().sort_values()

        # Create the Matplotlib figure and axes using the Object-Oriented API
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_data.plot(kind='barh', ax=ax, color='skyblue')

        ax.set_title("Average Normalized Salaries by Company Name")
        ax.set_xlabel("Average Normalized Salary ($)")
        ax.set_ylabel("Company Name")
        ax.set_xlim(xmin=min(plot_data) * 0.9, xmax=max(plot_data) * 1.1)
        
        # Add value labels to the bars
        for index, value in enumerate(plot_data):
            ax.text(value, index, f'${value:,.0f}', va='center', ha='left')

        # Display the barplot in Streamlit using st.pyplot()
        st.pyplot(fig)

        #------------------------------------------------------------------------------

        # Barplot on average normal salary over location.
        
        plot_data2 = filtered_scatter_graph_df.groupby('location')['normalized_salary'].mean().sort_values()

        # Create the Matplotlib figure and axes using the Object-Oriented API
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        plot_data2.plot(kind='barh', ax=ax2, color='skyblue')

        ax2.set_title("Average Normalized Salaries by Location")
        ax2.set_xlabel("Average Normalized Salary ($)")
        ax2.set_ylabel("Location")
        ax2.set_xlim(xmin=min(plot_data2) * 0.9, xmax=max(plot_data2) * 1.1)
        
        # Add value labels to the bars
        for index, value in enumerate(plot_data2):
            ax2.text(value, index, f'${value:,.0f}', va='center', ha='left')

        # Display the barplot in Streamlit using st.pyplot()
        st.pyplot(fig2)

        #-------------------------------------------------------------------------------------------

        # Barplot on average normal salary over job title.

        plot_data3 = filtered_scatter_graph_df.groupby('title')['normalized_salary'].mean().sort_values()

        # Create the Matplotlib figure and axes using the Object-Oriented API
        fig3, ax3 = plt.subplots(figsize=(8, 8))
        plot_data3.plot(kind='barh', ax=ax3, color='skyblue')

        ax3.set_title("Average Normalized Salaries by Job Title")
        ax3.set_xlabel("Average Normalized Salary ($)")
        ax3.set_ylabel("Job Title")
        ax3.set_xlim(xmin=min(plot_data3) * 0.9, xmax=max(plot_data3) * 1.1)
        
        # Add value labels to the bars
        for index, value in enumerate(plot_data3):
            ax3.text(value, index, f'${value:,.0f}', va='center', ha='left')

        # Display the barplot in Streamlit using st.pyplot()
        st.pyplot(fig3)

        #-------------------------------------------------------------------------------------------

        # Barplot on average normal salary over job skills.

        # If the user did not filter any skills, then the options selected will be all skills by default.
        if not selected_job_skills:
            skill_options = skill_selection

        else:
            skill_options = selected_job_skills

        # Initilize array of skill salaries
        skill_salary = {}

        # Add all the normalized salaries for each skill
        for skill in skill_options:
            if skill in filtered_scatter_graph_df.columns:
                average_salary = (filtered_scatter_graph_df.loc[filtered_scatter_graph_df[skill] == 1, 'normalized_salary'].mean())
                skill_salary[skill] = average_salary

        # Only show barplots of skill that has salary.
        skill_salary = {skill: salary for skill, salary in skill_salary.items() if not math.isnan(salary)}

        # Sort by the highest salary on top of the bar plot.
        plot_data4 = pd.Series(skill_salary).sort_values()

        # Create the Matplotlib figure and axes using the Object-Oriented API
        fig4, ax4 = plt.subplots(figsize=(8, 8))
        plot_data4.plot(kind='barh', ax=ax4, color='skyblue')

        ax4.set_title("Average Normalized Salaries by Job Skills")
        ax4.set_xlabel("Average Normalized Salary ($)")
        ax4.set_ylabel("Type of Skill")
        ax4.set_xlim(xmin=min(plot_data4) * 0.9, xmax=max(plot_data4) * 1.1)
        
        # Add value labels to the bars
        for index, value in enumerate(plot_data4):
            ax4.text(value, index, f'${value:,.0f}', va='center', ha='left')

        # Display the barplot in Streamlit using st.pyplot()
        st.pyplot(fig4)


# Tab for the regression summary of entire dataset.
with tabs[2]:
    st.header("Regression Summary")

    # Define columns
    col_a, col_b, col_c = st.columns(3)

    # Column shows the regression formula with slope and intercept
    with col_a: 
        st.header("Regression Formaula")
        st.write("Intercept of working regression dataset (b0):", round(intercept, 2))
        st.write("Coefficients of working regression dataset ordered from 'company_name', 'location', and 'job_title' :", coefficients)

    # Column will show the mean squared error of the dataset from the line.
    with col_b:
        st.header("Mean Squared Error of regression vs dataset")
        st.write("MSE of working regression dataset:", round(mse, 2))

    # Coefficient of determination.
    with col_c:
        st.header("R2 score of the dataset")
        st.write("R2 score of the working dataset:", round(r2, 2))

    
    # Display the scatterplot of all available salaries over job tuple group.
    st.header("Job Data Scatterplot of unfilterized entire dataset")
    
    # Combine 3 x columns into one column
    scatter_graph_df["Company_name_Title_Location"] = scatter_graph_df.apply(lambda x: 
                                                                             f'{x["company_name"]} {x["title"]} {x["location"]}', 
                                                                             axis = 1)
    
    # Need a numeric index for regression.
    scatter_graph_df = scatter_graph_df.reset_index().rename(columns={"index": "x_numeric"})

    #Make sure x is in a sorted order.
    scatter_graph_df = scatter_graph_df.sort_values("Company_name_Title_Location")
    
    # Add plots to the data.
    plots = (alt.Chart(scatter_graph_df).mark_point().encode(x = alt.X("Company_name_Title_Location:O", 
                                                                       sort = scatter_graph_df["Company_name_Title_Location"].tolist(),
                                                                       title = "Company / Title / Location"), 
                                                                       y = "normalized_salary:Q", 
                                                                       tooltip = ["company_name", 
                                                                                  "title", "location", "normalized_salary"])
                                                                       .properties(title = "Scattered plot of salary over the " \
                                                                       "Company Name, Title, and Location tuple.")).interactive()
    

    # Add a scatter line through the scatterplot.
    line = alt.Chart(scatter_graph_df).mark_line(color ='red').encode(x = "Company_name_Title_Location:O", 
                                                                      y = "mean(normalized_salary):Q").interactive()

    # If user inputed a job to predict the salary, add the point into the scatterplot.
    if company_input and location_input and title_input:
        new_point = pd.DataFrame([{
            "company_name": company_input,
            "location": location_input,
            "title": title_input,
            "normalized_salary": estimated_salary
        }])

        # Combine the inputs into one tuple
        new_point["Company_name_Title_Location"] = new_point.apply(lambda x: f'{x["company_name"]} {x["title"]} {x["location"]}', axis = 1)

        # Make a numeric index to match the category order if it exists, otherwise add at the end of the dataset.

        # Check if the user input exists in the dataset, otherwise, add it at the very end of the dataset.
        label = new_point["Company_name_Title_Location"].iloc[0]
        labels = scatter_graph_df["Company_name_Title_Location"].tolist()

        if label in labels:
            new_idx = labels.index(label)
        
        else:
            new_idx = len(labels)

        new_point["x_numeric"] = new_idx

        # Make the new point red so that the user will know.
        new_point_highlight = (alt.Chart(new_point).mark_point(size=200, color="red").encode(
            x="Company_name-Title-Location:O", y="normalized_salary:Q"))
        
        # Combine with the chart.
        chart = plots + line + new_point_highlight  

    # Otherwise, display with the original points and regression line.
    else:
        chart = plots + line

    # Display through the streamlit browser.
    st.altair_chart(chart, use_container_width= True)

    
    # Parity plot compares the predicted values to the actual values.
    fig_parity, ax_parity = plt.subplots()
    ax_parity.scatter(y_vector, y_pred)
    ax_parity.plot([min(y_vector), max(y_vector)], [min(y_vector), max(y_vector)], 
                   'r--', label='Ideal Salary Regression Line') # Ideal parity line
    ax_parity.set_xlabel("Actual Normalized Salaries")
    ax_parity.set_ylabel("Predicted Normalized Salaries")
    ax_parity.set_title("Parity Plot of the Predicted Salaries vs the Normalized Salaries")
    ax_parity.legend()
    ax_parity.grid(True)
    st.pyplot(fig_parity)

# Tab for the top k companies, locations, titles, and skills.
with tabs[3]:

    # Create a slider to determine the value of k for the top-k results
    K = st.slider("Select the value of K to produce the top K results", min_value = 1, max_value = 50, value = 5)

    # Definition to make the barplots for the top k companies, locations, and title.
    def top_k_barplot(df, column_name, k):
        top_k = df[column_name].value_counts().head(k)

        fig_top_k, ax_top_k = plt.subplots(figsize=(10, 6))
        top_k.plot(kind='bar', ax = ax_top_k)

        ax_top_k.set_title(f"Top {k} {column_name}")
        ax_top_k.set_xlabel(column_name)
        ax_top_k.set_ylabel("Count")
        ax_top_k.tick_params(axis = 'x', rotation = 90)
        st.pyplot(fig_top_k)

    # Call using the original dataframe.
    top_k_barplot(regression_df, "company_name", K)
    top_k_barplot(regression_df, "location", K)
    top_k_barplot(regression_df, "title", K)

    st.warning("Top K job skills barplot will come soon.")
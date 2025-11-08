'''
streamlit run dashboard.py 
'''

import streamlit as st
import pandas as pd


# NOTE: The one important session state.
# st.session_state.job_market_filtered_df


# -- Update DF Func ---
def update_df():
    print('update')

    valid_csv = "datasets/clean_job_postings.csv"

    df = pd.read_csv(valid_csv, low_memory=False)

    # Only store df values for first run.
    if st.session_state.job_market_filtered_df is None:
        st.session_state.multi_options['company_name'] = set(df['company_name'])
        st.session_state.multi_options['title'] = set(df['title'])
        st.session_state.multi_options['location'] = set(df['location'])
        st.session_state.multi_options['work_type'] = set(df['work_type'])
        st.session_state.multi_options['formatted_experience_level'] = set(df['formatted_experience_level'])
        st.session_state.multi_options['skills'] = set(df.columns[31:])

    for key, vals in st.session_state.filter_map.items():
        if key == 'skills' and vals != []:
            mask = df[st.session_state.filter_map['skills']].ge(1).any(axis=1)
            df = df[mask]
        elif vals != []:
            df = df[df[key].isin(vals)]

    st.session_state.job_market_filtered_df = df

# ---------------------



# ------- MAIN --------

st.session_state.setdefault("filter_map", {'company_name':[], 'title':['Software Engineer'], 'location':[], 'work_type':[], 'formatted_experience_level':[], 'skills':[]})

st.session_state.setdefault("multi_options", {'company_name':[], 'title':[''], 'location':[], 'work_type':[], 'formatted_experience_level':[], 'skills':[]})

st.session_state.setdefault("job_market_filtered_df", None)

st.set_page_config(layout="wide", page_title="Job Market Explorer",
                   initial_sidebar_state="expanded")

# WEBSITE OPEN
if not st.session_state.get("_app_inited", False):
    update_df()
st.session_state["_app_inited"] = True

# Pages of the application
home  = st.Page("views/home.py",                  title="Home (Job Exploration)", icon="🏠")
dummy  = st.Page("views/dummy_dashboard.py",                  title="Dummy", icon="👤")
assoc = st.Page("views/associative_dashboard.py", title="Skill Bundles (Association Rules)", icon="🔗")
cluster = st.Page("views/cluster_dashboard.py", title="Job Landscape (Clustering)", icon="🧩")
regression = st.Page("views/regression_dashboard.py", title="Salary Predictor (Regression)", icon="📈")

# Filters for the dataframe used.
st.sidebar.subheader("Filters")
filter_option = st.sidebar.selectbox('Choose Column:', ['company_name', 'title', 'location', 'work_type', 'formatted_experience_level', 'skills'], key="filter_option")

# Unique key for each text input field
ti_key = f"main_filter_{filter_option}"

# Filter for the given option
st.sidebar.multiselect(
    f"{filter_option.capitalize()} Filter",
    options=st.session_state.multi_options[filter_option],
    default=st.session_state.filter_map[filter_option],
    key=ti_key,         # bind to session state
    placeholder="Start typing...",
)

# Update the option in the map with the current text
st.session_state.filter_map[filter_option] = st.session_state.get(ti_key, "")

if st.sidebar.button('Run Filter'):
    update_df()



nav = st.navigation([home, dummy, assoc, cluster, regression])

nav.run()

# ---------------------

'''
streamlit run dashboard.py 
'''

import streamlit as st
import pandas as pd


# NOTE: There are 3 main sessions states.
# st.session_state.job_filter
# st.session_state.company_filter
# st.session_state.job_market_filtered_df


# -- Update DF Func ---
def update_df():
    print('update')
    def standardize_posting_id(df: pd.DataFrame) -> pd.DataFrame:
        if "posting_id" not in df.columns:
            for c in ["id", "job_id", "postingId", "postingID", "PostingID"]:
                if c in df.columns:
                    df = df.rename(columns={c: "posting_id"})
                    break
        return df

    postings_csv="datasets/archive/postings.csv"
    lexicon_csv="datasets/universal_skills_catalog.csv"
    salaries_csv="datasets/archive/jobs/salaries.csv"

    df = pd.read_csv(postings_csv, low_memory=False)
    df = standardize_posting_id(df)

    title = st.session_state["job_filter"]
    company = st.session_state["company_filter"]

    if company:
        df = df[df["company_name"].astype(str).str.contains(company, case=False, na=False, regex=False)]
    if title:
        df = df[df["title"].astype(str).str.contains(title, case=False, na=False, regex=False)]

    st.session_state.job_market_filtered_df = df

# ---------------------



# ------- MAIN --------

st.session_state.setdefault("job_filter", "software")
st.session_state.setdefault("company_filter", "")
st.session_state.setdefault("job_market_filtered_df", None)

st.set_page_config(layout="wide", page_title="Job Market Explorer",
                   initial_sidebar_state="expanded")

# WEBSITE OPEN
if not st.session_state.get("_app_inited", False):
    update_df()
st.session_state["_app_inited"] = True

# Pages of the application
home  = st.Page("views/home.py",                  title="Home", icon="🏠")
dummy  = st.Page("views/dummy_dashboard.py",                  title="Dummy", icon="👤")
assoc = st.Page("views/associative_dashboard.py", title="Skill Bundles (Association Rules)", icon="🔗")
cluster = st.Page("views/cluster_dashboard.py", title="Job Landscape (Clustering)", icon="🧩")
regression = st.Page("views/regression_dashboard.py", title="Salary Predictor (Regression)", icon="📈")

# Filters for the dataframe used.
st.sidebar.subheader("Filters")
st.sidebar.text_input("Job filter", key="job_filter", value=st.session_state.job_filter, on_change=update_df)
st.sidebar.text_input("Company filter", key="company_filter", value=st.session_state.company_filter, on_change=update_df)

nav = st.navigation([home, dummy, assoc, cluster, regression])

nav.run()

# ---------------------

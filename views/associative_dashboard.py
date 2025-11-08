import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#st.sidebar.page_link("dashboard.py", label="🏠 Home")
#st.sidebar.page_link("pages/associative_dashboard.py", label="Association Rule Mining")
#st.set_page_config(layout="wide", initial_sidebar_state="expanded")
#st.title("Title")


st.title("Skill Bundles — What Skills Go Together")

st.caption("Discover frequently co-occurring skills across job postings to inform learning paths, resume bundling, and course planning.")
st.markdown("""
**What this does**
- Mines association rules between skills and roles (support, confidence, lift, interest).
- Surfaces common bundles and rare-but-high-lift combos.
- Lets you filter by company, role, location, and date range.

**Great for**
- Finding complementary skills to add.
- Spotting company- or role-specific bundles.
- Prioritizing high-impact upskilling.
""")

#df = st.session_state.job_market_filtered_df
#
st.subheader('Charts')
tab1, tab2 = st.tabs(['Tab 1', 'Tab 2'])

with tab1:
    #st.write('Content for tab1')
    st.subheader('Filtered Job Description DataFrame')
    st.write(st.session_state.job_market_filtered_df)
with tab2:
    st.write('Content for tab2')


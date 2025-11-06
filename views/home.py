import streamlit as st
import pandas as pd



# ------- EDA ---------

st.subheader('Exporatory Data Analysis (EDA) Dashboard')
filtered_df, tab2 = st.tabs(['Filtered DataFrame', 'Tab 2'])

with filtered_df:
    st.subheader('Filtered Job Description DataFrame')
    st.write(st.session_state.job_market_filtered_df)
with tab2:
    st.write('Content for tab2')

# ---------------------

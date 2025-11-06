import streamlit as st

st.title("Job Landscape — What Kinds of Jobs Exist")

# NOTE: This part doesn't need to be in the final dashboard, just to help your understanding.
st.caption("Cluster similar job postings by required skills to reveal role families, sub-specialties, and outliers.")
st.markdown("""
**What this does**
- Groups jobs into clusters based on skill vectors / embeddings.
- Profiles each cluster with top skills, example titles, and representative companies.
- Visualizes the market map (e.g., 2D projection) with filters.

**Great for**
- Understanding role neighborhoods (e.g., Data Eng vs ML Eng vs Analytics).
- Finding adjacent clusters to pivot into.
- Spotting niche pockets with unique skill mixes.
""")

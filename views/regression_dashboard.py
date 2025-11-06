import streamlit as st

st.title("Salary Predictor — Estimate Pay from Role, Company & Skills")

# NOTE: This part doesn't need to be in the final dashboard, just to help your understanding.
st.caption("Predict a salary using job title, company, skills, and ZIP code, with confidence intervals and key drivers.")
st.markdown("""
**What this does**
- Trains/loads a regression model to estimate base salary.
- Explains drivers (feature importance / partial effects) and shows model fit (MAE, R², residuals).
- Lets you compare scenarios by tweaking inputs.

**Great for**
- Benchmarking offers or expectations.
- Testing which skills shift pay the most.
- Exploring location/company effects by ZIP and employer.

*Note:* This is an estimate from historical postings—actual compensation may vary.
""")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


# ------- EDA ---------



st.title('Job Exploration Dashboard')
st.caption(
    "Interactively explore filtered job postings: "
    "view the table, see how common different categories are, "
    "and compare salary distributions."
)

with st.expander("ℹ️ How to use this page", expanded=False):
    st.markdown(
        """
        **Tabs:**
        - **Tabular Job Postings Data** – see the filtered job postings table.
        - **Number of Job Postings Bar Plot** – count how many postings fall into each category.
        - **Salary Box Plots** – compare salary distributions across categories or skills.

        **Tips:**
        - Use the **bin slider** to control how many top categories/skills are shown.
        - Try switching between **company, title, location, experience level, work type, skills**, etc.
        """
    )



bin_count = st.slider('Number of top categories / skills to show', min_value=1, max_value=20, value=3, help="Controls how many top items (companies, titles, locations, skills, etc.) are shown in the plots below.")


tab1, tab2, tab3 = st.tabs(['📋 Tabular Job Postings Data', '📊 Number of Job Postings Bar Plot', '💵 Salary Box Plots'])



with tab1:
    st.subheader('Filtered Job Postings Table')
    st.caption(
            "This table shows the current filtered view of the job postings. "
            "Use the filters on other pages to change which rows appear here."
        )
    st.write(f'Number of Entries: {len(st.session_state.job_market_filtered_df)}')
    #st.write(st.session_state.job_market_filtered_df)
    st.write(st.session_state.job_market_filtered_df[['posting_id', 'company_id', 'company_name', 'title', 'formatted_experience_level', 'description', 'normalized_salary', 'max_salary', 'med_salary', 'min_salary', 'location', 'zip_code', 'work_type', 'applies', 'listed_time', 'expiry', 'remote_allowed', 'application_type', 'job_posting_url']])



with tab2:
    st.subheader('Number of Job Postings by Category')
    st.caption(
            "Select how you want to group the postings (e.g., by company, title, location, "
            "experience level, or skills). The bar chart shows how many postings fall into "
            "each of the top categories."
        )
    options = ['company_name', 'title', 'location', 'work_type', 'formatted_experience_level', 'normalized_salary', 'skills']
    option = st.selectbox('Group jobs by...', options, key="tab2_selector", help=(
        "Choose which column to group by.\n"
        "For example, 'Company' will show the top companies by number of postings."))

    if option == 'normalized_salary':
        binned = pd.cut(st.session_state.job_market_filtered_df[option], bins=bin_count)
        counts = binned.value_counts(sort=False)  # keep bin order
    elif option == 'skills':
        drop_list = set(st.session_state.job_market_filtered_df.columns[:31])
        skills_df = st.session_state.job_market_filtered_df.loc[:, ~st.session_state.job_market_filtered_df.columns.isin(drop_list)]
        counts = skills_df.sum(axis=0, skipna=True).nlargest(bin_count)
    else:
        counts = (
            st.session_state.job_market_filtered_df[option]
            .astype(str)
            .value_counts(dropna=False)
            .head(bin_count)
        )

    # --- tidy DataFrame for Plotly, with nice labels ---
    idx = counts.index
    if pd.api.types.is_interval_dtype(idx):  # from pd.cut
        labels = [f"{iv.left:,.0f}–{iv.right:,.0f}" for iv in idx]
    else:
        labels = [str(x) for x in idx]

    df_bar = pd.DataFrame({"label": labels, "count": counts.values})

    # Preserve current order exactly
    category_order = list(df_bar["label"])

    # --- Plotly bar ---
    fig = px.bar(
        df_bar, x="label", y="count", text="count",
        labels={"label": option, "count": "Count"},
    )

    # Format numbers, put text above bars, rotate ticks, add headroom
    top = df_bar["count"].max() if len(df_bar) else 0
    fig.update_traces(texttemplate="%{y:,}", textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=500, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_tickangle=25,
        xaxis={"categoryorder": "array", "categoryarray": category_order},
        yaxis_range=[0, top * 1.10 if top else 1],
    )

    st.plotly_chart(fig, use_container_width=True)



with tab3:
    st.subheader('Salary Distributions by Category')
    st.caption(
            "Compare salary distributions across different categories or skills. "
            "Each box shows the spread of normalized salaries for postings in that group."
        )

    option = st.selectbox('Group salary by...', ['company_name', 'title', 'location', 'work_type', 'formatted_experience_level', 'skills'], key="tab3_selector", help='Choose which column to use as groups for the salary box plots.')

    if option == 'skills':
        # choose skill columns and get top-N by frequency
        skill_cols = list(st.session_state.job_market_filtered_df.columns[31:])
        skills_df = st.session_state.job_market_filtered_df.loc[:, ['normalized_salary', *skill_cols]]
        order = (skills_df[skill_cols] > 0).sum().sort_values(ascending=False).head(bin_count).index.tolist()

        # build list of arrays and labels
        arrays = [skills_df.loc[skills_df[c] > 0, 'normalized_salary'].dropna().to_numpy(float) for c in order]
        labels = order
    else:
        # non-skill categorical column
        data = st.session_state.job_market_filtered_df.dropna(subset=[option, 'normalized_salary'])
        order = data[option].value_counts().head(bin_count).index.tolist()
        arrays = [data.loc[data[option] == c, 'normalized_salary'].to_numpy(float) for c in order]
        labels = order

    # ---- build tidy DataFrame for Plotly ----
    # filter out empty groups to avoid shape errors
    arrays, labels = zip(*[(a, l) for a, l in zip(arrays, labels) if len(a) > 0]) if arrays else ([], [])
    if arrays:
        lengths = [len(a) for a in arrays]
        tidy = pd.DataFrame({
            "label": np.repeat(labels, lengths),
            "normalized_salary": np.concatenate(arrays)
        })
    else:
        tidy = pd.DataFrame(columns=["label", "normalized_salary"])

    # ---- plot ----
    fig = px.box(
        tidy, x='label', y='normalized_salary',
        points='all',
        labels={"label": option, "normalized_salary": "Salary"},
        #color_discrete_sequence=["#636EFA"],
    )
    fig.update_traces(jitter=0.24, pointpos=0.0, marker_size=6, marker_opacity=0.6, selector=dict(type="box"))
    fig.update_layout(width=900, height=500, xaxis_tickangle=25, margin=dict(l=10, r=10, t=30, b=10))

    st.plotly_chart(fig, use_container_width=True)


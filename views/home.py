import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ------- EDA ---------

st.title('Exporatory Data Analysis (EDA) Dashboard')

bin_count = st.slider('Select the number of bins', min_value=1, max_value=20, value=3)
tab1, tab2, tab3 = st.tabs(['Tabular Job Data', 'Bar Plot', 'Box Plots'])



with tab1:
    st.subheader('Filtered Job DataFrame')
    st.write(f'Number of Entries: {len(st.session_state.job_market_filtered_df)}')
    #st.write(st.session_state.job_market_filtered_df)
    st.write(st.session_state.job_market_filtered_df[['posting_id', 'company_id', 'company_name', 'title', 'formatted_experience_level', 'description', 'normalized_salary', 'max_salary', 'med_salary', 'min_salary', 'location', 'zip_code', 'work_type', 'applies', 'listed_time', 'expiry', 'remote_allowed', 'application_type', 'job_posting_url']])



with tab2:
    st.subheader('Column Item Frequencies')
    options = ['company_name', 'title', 'location', 'work_type', 'formatted_experience_level', 'normalized_salary', 'skills']
    option = st.selectbox('Choose Column:', options, key="tab2_selector")

    if option == 'normalized_salary':
        binned = pd.cut(st.session_state.job_market_filtered_df[[option]].iloc[:,0], bins=bin_count)
        counts = binned.value_counts(sort=False)
    elif option == 'skills':
        drop_list = set(st.session_state.job_market_filtered_df.columns[:31])
        skills_df = st.session_state.job_market_filtered_df.loc[:, ~st.session_state.job_market_filtered_df.columns.isin(drop_list)]
        col_sums = skills_df.sum(axis=0, skipna=True)
        counts = col_sums.nlargest(bin_count)
    else:
        counts = (
            st.session_state.job_market_filtered_df[option]
            .astype(str)                       # avoid issues with non-strings
            .value_counts(dropna=False)
            .head(bin_count)
        )

    # Make Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(f"Top {len(counts)} values in '{option}'")
    ax.set_xlabel(option)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", labelrotation=25)
    plt.tight_layout()
    labels = [f"{v:,}" for v in counts.values]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=9)
    top = counts.max()
    ax.set_ylim(0, top * 1.10)   # 20% extra space above tallest label
    st.pyplot(fig, clear_figure=True)



with tab3:
    st.subheader('Column Item Salary Distributions')
    option = st.selectbox('Choose Column:', ['company_name', 'title', 'location', 'work_type', 'formatted_experience_level', 'skills'], key="tab3_selector")

    if option == 'skills':
        skill_cols = set(st.session_state.job_market_filtered_df.columns[31:])
        skill_cols.add('normalized_salary')
        skills_df = st.session_state.job_market_filtered_df.loc[:, st.session_state.job_market_filtered_df.columns.isin(skill_cols)]
        #st.write(skills_df['c++'])
        skill_cols.remove('normalized_salary')
        order = (skills_df[list(skill_cols)] > 0).sum().sort_values(ascending=False).head(bin_count).index
        data_dict = {col: skills_df.loc[skills_df[col] > 0, "normalized_salary"].dropna().to_numpy(float) for col in order}
        labels = list(order)
        data   = [data_dict[c] for c in labels]
    else:
        data = st.session_state.job_market_filtered_df[[option, 'normalized_salary']]
        data = st.session_state.job_market_filtered_df.dropna(subset=[option, 'normalized_salary'])
        order = data[option].value_counts().index.tolist()
        data = [data.loc[data[option] == c, 'normalized_salary'].to_numpy() for c in order][:bin_count]
        labels = order[:bin_count]
        

    # Make Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(data, labels=labels)

    xpos = np.arange(1, len(data) + 1)  # boxplot positions start at 1
    point_handles = []
    for i, vals in enumerate(data, start=1):
        if len(vals) == 0:
            continue
        # jitter width: narrower when fewer categories
        jitter = (np.random.rand(len(vals)) - 0.5) * 0.24
        h = ax.scatter(
            np.full_like(vals, i, dtype=float) + jitter,
            vals,
            s=14, alpha=0.6, zorder=3, edgecolors="none"
        )
        point_handles.append(h)

    ax.set_xlabel(option)
    ax.set_ylabel("Salary")
    ax.tick_params(axis="x", labelrotation=25)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

# ---------------------

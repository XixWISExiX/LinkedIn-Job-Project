import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from associative_task.associative_algo import run_association_rule_mining



st.title("Skill Bundles — What Skills Go Together")

#st.caption("Discover frequently co-occurring skills across job postings to inform learning paths, resume bundling, and course planning.")
#st.markdown("""
#**What this does**
#- Mines association rules between skills and roles (support, confidence, lift, interest).
#- Surfaces common bundles and rare-but-high-lift combos.
#- Lets you filter by company, role, location, and date range.
#
#**Great for**
#- Finding complementary skills to add.
#- Spotting company- or role-specific bundles.
#- Prioritizing high-impact upskilling.
#""")

skill_cols = set(st.session_state.job_market_filtered_df.columns[31:])
skill_cols.add('normalized_salary')
skills_df = st.session_state.job_market_filtered_df.loc[:, st.session_state.job_market_filtered_df.columns.isin(skill_cols)]


col_1, col_2, col_3 = st.columns([4, 4, 4], gap="small")  # ≈ 33% / 33% / 33%

with col_1:
    topk = st.number_input('Topk Filter', min_value=1, value=100, step=25)

with col_2:
    max_rule_size = st.number_input('Max Rule Size', min_value=1, value=3)

with col_3:
    sort_col = st.selectbox('Sort by', ['support', 'confidence', 'lift'], index=1)
    #sort_col = st.selectbox('Filter', ['support', 'confidence', 'lift'], index=1)


min_support = st.slider('Min Support Threashold', min_value=0.0, value=0.05, max_value=1.0, step=0.01, format="%.2f")

rules_df = run_association_rule_mining(skills_df, min_support, max_rule_size, topk, sort_col)

def update_rules_df():
    rules_df = run_association_rule_mining(skills_df, min_support, max_rule_size, topk, sort_col)

tab1, tab2, tab3, tab4 = st.tabs(['Association Rules Table', 'Heatmap Scatterplot', 'Heatmap (LHS vs RHS)', 'Help'])

with tab1:
    st.subheader('Association Rules Table')
    st.write('An association rule is denoted as (antecedent -> consequent)')
    st.write(rules_df)

with tab2:
    st.subheader('Skill Association Heatmap Scatterplot')

    fig = px.scatter(
        rules_df, x="support", y="lift",
        color="confidence", color_continuous_scale="Inferno",
        size="support", size_max=18,
        hover_data=["antecedent","consequent","support","lift","confidence"],
        labels={"support":"Support", "lift":"Lift", "confidence":"Confidence"},
    )

    # reference line at lift=1
    fig.add_hline(y=1.0, line_color="gray", opacity=0.6)

    fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)


with tab3:
    st.subheader('Lift Heatmap (LHS vs RHS)')
    grid_size = st.slider("Matrix Size", min_value=1, max_value=100, value=10, step=1)

    AGG = "max"           # how to combine duplicate (LHS,RHS): "max" or "mean"
    DROP_OVERLAP = True   # drop rules where LHS ∩ RHS ≠ ∅

    skill_cols = [c for c in skills_df.columns if c != "normalized_salary"]

    def norm_itemset(x):
        """Return sorted tuple for strings/tuples/lists; None if contains unknown skills."""
        if isinstance(x, (list, tuple)):
            t = tuple(sorted(x))
        else:  # string (single item)
            t = (x,)
        # guard: make sure all items are known skill columns
        return t if all(it in skill_cols for it in t) else None

    r = rules_df.copy()
    r["lhs_set"] = r["antecedent"].apply(norm_itemset)
    r["rhs_set"] = r["consequent"].apply(norm_itemset)
    r = r.dropna(subset=["lhs_set", "rhs_set"])

    # optional: drop overlaps
    if DROP_OVERLAP:
        r = r[[set(a).isdisjoint(set(b)) for a, b in zip(r["lhs_set"], r["rhs_set"])]]

    # bound sizes
    r = r[r["lhs_set"].apply(len) <= max_rule_size]
    r = r[r["rhs_set"].apply(len) <= max_rule_size]

    if r.empty:
        st.info("No rules match the chosen LHS/RHS size constraints.")
    else:
        # aggregate duplicates
        if AGG == "mean":
            pair_lift = r.groupby(["lhs_set", "rhs_set"], as_index=False)["lift"].mean()
        else:
            pair_lift = r.groupby(["lhs_set", "rhs_set"], as_index=False)["lift"].max()

        # Choose which LHS/RHS combos to keep (top by best lift)
        lhs_best = pair_lift.groupby("lhs_set")["lift"].max().sort_values(ascending=False)
        rhs_best = pair_lift.groupby("rhs_set")["lift"].max().sort_values(ascending=False)
        lhs_keep = list(lhs_best.head(grid_size).index)
        rhs_keep = list(rhs_best.head(grid_size).index)

        sub = pair_lift[pair_lift["lhs_set"].isin(lhs_keep) & pair_lift["rhs_set"].isin(rhs_keep)]

        # pivot to matrix
        heat = sub.pivot_table(index="lhs_set", columns="rhs_set", values="lift", aggfunc="max")

        # to readable labels like "python & sql"
        def label(t): return " & ".join(t)

        heat.index = [label(t) for t in heat.index]
        heat.columns = [label(t) for t in heat.columns]

        # optional: fill missing with 1 (independence) so the colormap centers at 1
        heat = heat.fillna(1.0)

        # plot with separating gridlines
        fig = px.imshow(
            heat,
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=1.0,
            labels=dict(x="RHS skills", y="LHS skills", color="Lift"),
            aspect="auto",
        )
        fig.update_traces(
            hovertemplate="LHS=%{y}<br>RHS=%{x}<br>Lift=%{z:.3f}<extra></extra>",
            xgap=1, ygap=1
        )
        fig.update_layout(
            height=700,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="RHS (consequent) skills",
            yaxis_title="LHS (antecedent) skills",
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.write('Support = P(A∩B). Fraction of transactions containing A and B together (0–1).')
    st.write('Confidence = P(B|A) = support(A∩B)/support(A). Directional reliability of A→B.')
    st.write('Lift = confidence(A→B)/P(B). >1 positive association; =1 independent; <1 negative.')

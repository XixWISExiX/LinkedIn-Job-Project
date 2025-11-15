import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from associative_task.associative_algo import run_association_rule_mining



st.title("Skill Bundles — What Skills Go Together")

st.caption(
    "Mine association rules between skills to discover which skills frequently appear together "
    "in job postings. Use this to inform upskilling, resume bundling, and course planning."
)

with st.expander("ℹ️ What is this page showing?", expanded=False):
    st.markdown(
        """
        This page runs **association rule mining** over the skill columns in your filtered job postings.

        **You can:**
        - Control the **minimum support** (how common a pattern must be).
        - Limit the **maximum size** of each rule (how many skills on each side).
        - Restrict to the **top-K** rules sorted by support, confidence, or lift.

        **Tabs:**
        - **Association Rules Table** – view the raw rules (antecedent → consequent) with metrics.
        - **Heatmap Scatterplot** – visualize how rules trade off support and lift.
        - **Heatmap (LHS vs RHS)** – see lift for top left-hand vs right-hand skill bundles as a matrix.
        - **Help** – quick definitions and guidance.
        """
    )


skill_cols = set(st.session_state.job_market_filtered_df.columns[31:])
skill_cols.add('normalized_salary')
skills_df = st.session_state.job_market_filtered_df.loc[:, st.session_state.job_market_filtered_df.columns.isin(skill_cols)]


col_1, col_2, col_3 = st.columns([4, 4, 4], gap="small")  # ≈ 33% / 33% / 33%

with col_1:
    topk = st.number_input('Top-K rules to keep', min_value=1, value=100, step=25, help="After mining, keep only the top K rules based on the chosen sort metric.")

with col_2:
    max_rule_size = st.number_input('Maximum rule size', min_value=1, value=3, help="Maximum number of skills allowed on each side of a rule (e.g., 3 allows up to triples like {Python, SQL, Spark} → {AWS}).")

with col_3:
    sort_col = st.selectbox('Sort rules by', ['support', 'confidence', 'lift'], index=1, help=(
            "- **support**: prioritize more common patterns\n"
            "- **confidence**: prioritize more reliable A→B rules\n"
            "- **lift**: prioritize patterns that are unexpectedly strong"
        ))
    #sort_col = st.selectbox('Filter', ['support', 'confidence', 'lift'], index=1)


min_support = st.slider('Minimum support threshold', min_value=0.0, value=0.05, max_value=1.0, step=0.01, format="%.2f", help="Rules must appear in at least this fraction of postings (e.g., 0.05 means 5% of all filtered job postings).")

rules_df = run_association_rule_mining(skills_df, min_support, max_rule_size, topk, sort_col)

def update_rules_df():
    rules_df = run_association_rule_mining(skills_df, min_support, max_rule_size, topk, sort_col)

tab1, tab2, tab3, tab4 = st.tabs(["📋 Association Rules Table", "📈 Heatmap Scatterplot", "🧊 Heatmap (LHS vs RHS)", "❓ Help"])

with tab1:
    st.subheader('Association Rules Table')
    st.caption(
            "Each row is a rule of the form **(antecedent → consequent)**, "
            "with support, confidence, and lift."
        )
    #st.write(rules_df[['antecedent', 'consequent', 'support', 'confidence', 'lift', 'correlation']])

# Pretty string representations for itemsets if needed
    def format_itemset(x):
        if isinstance(x, (list, tuple, set)):
            return ", ".join(sorted(map(str, x)))
        return str(x)

    display_df = rules_df.copy()
    display_df["antecedent"] = display_df["antecedent"].apply(format_itemset)
    display_df["consequent"] = display_df["consequent"].apply(format_itemset)

    st.dataframe(
        display_df[['antecedent', 'consequent', 'support', 'confidence', 'lift', 'correlation']],
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.subheader('Skill Association Heatmap Scatterplot')
    st.caption(
            "Each point is a rule (antecedent → consequent). "
            "The x-axis is **support**, the y-axis is **lift**, "
            "and the color encodes **confidence**."
        )

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
    st.subheader('Lift Heatmap (LHS vs RHS skill bundles)')
    st.caption(
            "Shows the **lift** between top left-hand-side (LHS) and right-hand-side (RHS) "
            "skill bundles. Darker colors indicate stronger positive association."
        )
    grid_size = st.slider("Number of LHS/RHS bundles to show", min_value=1, max_value=100, value=10, step=1, help="Controls how many top LHS and RHS bundles (by best lift) appear in the matrix.")

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
    st.subheader("How to read these metrics")
    st.markdown(
        r"""
**Support**

$$
\text{support}(A \rightarrow B) = P(A \cap B)
$$
- Interpretation: fraction of all job postings that contain **both** the skills in A and B.
- Range: 0–1. Higher means the pattern is more common.

**Confidence**

$$
\text{confidence}(A \rightarrow B) = P(B \mid A)
= \frac{P(A \cap B)}{P(A)}
$$

- Interpretation: if a posting has skills A, how likely is it to also have skills B?
- Directional: (A $\rightarrow$ B) and (B $\rightarrow$ A) can have different confidence.

**Lift**

$$
\text{lift}(A \rightarrow B)
= \frac{\text{confidence}(A \rightarrow B)}{P(B)}
$$

- Interpretation: how much more likely B is when A is present, compared to B appearing at random.
- Lift > 1: positive association (A and B appear together more than expected).
- Lift = 1: independence (no special association).
- Lift < 1: negative association (seeing A makes B less likely).


---
**Typical use-cases:**
- Find **bundles of skills** to group on a resume (e.g., Python & SQL & Pandas).
- See which skills tend to appear together for a given **job title or company** (via filters on the main page).
- Prioritize **upskilling** toward skills that frequently appear with ones you already know.
        """,
    )

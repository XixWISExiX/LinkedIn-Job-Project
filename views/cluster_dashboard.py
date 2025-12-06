import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from clustering_task.clustering_algo import KMeansScratch, pca_transform
from scipy.stats import ttest_ind
from scipy.spatial.distance import pdist, squareform


# getting skills data
skill_cols = set(st.session_state.job_market_filtered_df.columns[31:])
skill_cols.add('normalized_salary')

skills_df = st.session_state.job_market_filtered_df.loc[:, st.session_state.job_market_filtered_df.columns.isin(skill_cols)]
skills_df["company_name"] = st.session_state.job_market_filtered_df["company_name"]
skills_df["title"] = st.session_state.job_market_filtered_df["title"]
skills_df["posting_id"] = st.session_state.job_market_filtered_df["posting_id"]

max_sample = skills_df.shape[0]


# build X using only real skill columns
feature_cols = [c for c in skills_df.columns if c not in ("company_name", "title", "posting_id", "normalized_salary")]
X = skills_df[feature_cols].to_numpy(float)

st.set_page_config(page_title="Job Clustering Dashboard", layout="wide")
st.title("Job Landscape — What Jobs Are Similar")

st.caption(
    "Explore job clusters built from skill vectors using PCA and K-Means. "
    "Use this to understand how skill sets group together across the job market."
)

with st.expander("ℹ️ What is this page showing?", expanded=False):
    st.markdown(
        """
        This page runs **unsupervised clustering** on job postings using their **skill vectors**.

        **Pipeline (high level):**
        1. Skills → high-dimensional vectors (already one-hot).
        2. **PCA** reduces dimensionality to preserve the most important variance.
        3. **K-Means** groups similar skill profiles into clusters.

        **You can:**
        - Choose **k (clusters)** to control how granular the job groups are.
        - Select a **distance metric** that influences how similarity is measured.
        - Adjust **PCA components** to trade off interpretability vs information retention.
        - Evaluate results with **Silhouette**, **Davies–Bouldin**, or **Calinski–Harabasz**.

        **Tabs:**
        - **3D Cluster Map** – visualize distinct skill neighborhoods in PCA space.
        - **Similarity Matrix** – inspect pairwise similarity between postings.
        - **Evaluation** – validate cluster quality + salary separation signals.
        - **Help** – definitions and interpretation ranges.
        """
    )

col1, col2, col3, col4 = st.columns(4)
with col1:
    num_clusters = st.number_input("k (clusters)", 2, 100, 5, 1, help=("Controls how many skill groups K-Means will form. Smaller k gives broader categories; larger k creates more specialized clusters. If k is too large, clusters may become tiny and less meaningful.")
)

with col2:
    num_pca_components = st.number_input("PCA components", 3, 50, 3, help="Number of principal components used before clustering. More components preserve more variance but reduce interpretability. 3–8 is usually a good balance for visualization.")

with col3:
    num_iterations = st.number_input("K-Means iterations", 10, 1000, 100, step=10, help="How long K-Means refines centroids. Higher values can help stability but increase runtime.")

with col4:
    num_samples = st.number_input(
        "Max samples for clustering",
        min_value=5,
        max_value=max_sample,
        value=min(500, max_sample),
        step=(((max_sample//40 + 25) // 50) * 50),
        help="Limits rows used for PCA/K-Means for responsiveness. "
            "If the filtered set is larger, a random sample is used."
    )

col5, col6, col7 = st.columns([4, 4, 4], gap="small")

with col5:
    evaluation_function = st.selectbox("Evaluation Function", ["silhouette", "davies-bouldin", "calinski-harabasz"], help="Determines which evaluation score to use of the K-Means output.")

with col6:
    distance_function = st.selectbox("Distance Function", ["euclidean", "manhattan", "minkowski"], help="The distance metric which is used to compare points to centroids.")

with col7:
    if distance_function == "minkowski":
        minkowski_p = st.slider("Minkowski Power (p)", min_value=3.0, max_value=5.0, value=3.0, step=0.1, help="Exponent that determines sensitivity to larger differences.")



if skills_df.shape[0] > 200:
    skills_df = skills_df.sample(n=200, random_state=42).reset_index(drop=True)
    X = skills_df[feature_cols].to_numpy(float)


# tab set up
tabs = st.tabs(["🧭 3D Cluster Map", "🧊 Similarity Matrix", "📏 Evaluation", "❓ Help"])


# KMeans algorithm
Xpca, comps, mu = pca_transform(X, n_components=num_pca_components)

eigenvalues = np.var(Xpca, axis=0)
total_variance = np.var(X, axis=0).sum()
percent_variance = (eigenvalues / total_variance) * 100
cumulative_variance = np.cumsum(percent_variance)

pca_table = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(num_pca_components)],
    "Eigenvalue (Total)": np.round(eigenvalues, 3),
    "% of Variance": np.round(percent_variance, 3),
    "Cumulative %": np.round(cumulative_variance, 3)
})

km = KMeansScratch(
    n_clusters=num_clusters,
    max_iters=num_iterations,
    distance=distance_function
).fit(Xpca)

labels, centroids = km.labels, km.centroids
centroids_pca = centroids
pca_labels = [f"PC{i+1}" for i in range(num_pca_components)]


# salary summary
salary_df = skills_df.assign(Cluster=labels)
salary_df = salary_df[salary_df["normalized_salary"] > 0]
cost_summary = (
    salary_df.groupby("Cluster")["normalized_salary"]
    .agg(["mean", "median", "min", "max", "count"])
    .round(2)
    .reset_index()
    .rename(columns={"count": "Size"})
)

# cluster map
with tabs[0]:
    st.subheader("Skill Clusters in PCA Space")
    st.caption("Jobs projected into PCA space and colored by cluster. Centroids represent the 'average' skill profile for each group.")

    ix, iy, iz = 0, 1, 2
    ax_x, ax_y, ax_z = pca_labels[ix], pca_labels[iy], pca_labels[iz]

    hover_texts = []
    for i in range(X.shape[0]):
        job_skills = [feature_cols[j] for j in np.where(X[i] > 0)[0]]

        company = skills_df.iloc[i]["company_name"]
        title = skills_df.iloc[i]["title"]

        hover_texts.append(
            f"Company: {company}<br>"
            f"Title: {title}<br>"
            f"Top Skills: {', '.join(job_skills[:5])}"
        )

    df3d = pd.DataFrame({
        "x": Xpca[:, ix],
        "y": Xpca[:, iy],
        "z": Xpca[:, iz],
        "Cluster": labels,
        "Skills": hover_texts
    })

    fig3d = px.scatter_3d(
        df3d,
        x="x", y="y", z="z",
        color=df3d["Cluster"].astype(str),
        hover_name="Skills",
        opacity=0.8,
        color_discrete_sequence=px.colors.qualitative.Vivid
    )

    fig3d.add_trace(
        go.Scatter3d(
            x=centroids_pca[:, ix],
            y=centroids_pca[:, iy],
            z=centroids_pca[:, iz],
            mode="markers+text",
            text=[f"C{i}" for i in range(num_clusters)],
            textposition="top center",
            marker=dict(size=9, color="black", line=dict(width=4, color="white")),
            name="Centroids"
        )
    )

    fig3d.update_layout(
        height=700,
        scene=dict(xaxis_title=ax_x, yaxis_title=ax_y, zaxis_title=ax_z)
    )

    st.plotly_chart(fig3d, width='stretch')



# similarity matrix tab

# Proper job labels pulled directly from skills_df
job_labels = [
    f"{skills_df.iloc[i]['company_name']} — {skills_df.iloc[i]['title']} — {skills_df.iloc[i]['posting_id']}"
    for i in range(X.shape[0])
]

with tabs[1]:
    st.subheader("Pairwise Similarity Heatmap")
    #st.caption("Distance-based similarity between postings in PCA space.")
    st.caption("Similarity between postings in PCA space. Use the slider to limit the view for readability.")

    ## cosine similarity
    sim = 1 - squareform(pdist(Xpca, metric=distance_function))
    #sim = 1 - squareform(pdist(Xpca, metric="euclidean"))

    sim_df = pd.DataFrame(sim, index=job_labels, columns=job_labels)

    # --- Slider to limit heatmap size for readability ---
    N = len(job_labels)
    max_show = min(100, N)  # keep this sane for UI performance
    show_n = st.slider(
        "Items to display",
        min_value=10,
        max_value=max_show,
        value=min(25, max_show),
        step=1,
        help="Limits the heatmap size for readability and faster rendering."
    )

    # Deterministic subset so the view doesn't jump around each rerun
    subset_idx = list(range(show_n))
    labels_sub = [job_labels[i] for i in subset_idx]
    sim_sub = sim[np.ix_(subset_idx, subset_idx)]
    sim_df_sub = pd.DataFrame(sim_sub, index=labels_sub, columns=labels_sub)

    fig = px.imshow(
        sim_df_sub,
        color_continuous_scale="Viridis",
        aspect="auto",
        labels=dict(color="Similarity")
    )


    # --- Dynamic height based on items shown ---
    base_height = 350
    per_item = 22
    max_height = 1200
    dynamic_height = min(max_height, base_height + per_item * show_n)

    fig.update_layout(
        height=dynamic_height,
        margin=dict(l=30, r=10, t=40, b=80)
    )

    st.plotly_chart(fig, use_container_width=True)


# evaluation tab
with tabs[2]:
    c1, c2 = st.columns([3, 1], gap="large", vertical_alignment="center")
    with c1:
        st.subheader("Cluster Quality & Salary Differences")
        st.caption("Check cluster compactness/separation, PCA variance retention, and whether clusters show meaningful salary differences.")

    with c2:
        # Evaluation score
        if evaluation_function == "silhouette":
            score = silhouette_score(X, labels)
        elif evaluation_function == "davies-bouldin":
            score = davies_bouldin_score(X, labels)
        else:
            score = calinski_harabasz_score(X, labels)
        st.metric(label=f"{evaluation_function.title()} Score", value=round(score, 4))

    metric_tabs = st.tabs(["Cluster Sizes", "PCA Component Distribution", "Cluster Cost Summary", "T-Test"])

    with metric_tabs[0]:
        # Cluster size distribution
        st.subheader("Cluster Sizes")
        st.caption("Shows how many postings fall into each cluster. Very tiny clusters may indicate over-fragmentation (k too high).")
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        size_df = pd.DataFrame({"Cluster": cluster_sizes.index, "Size": cluster_sizes.values})
        st.bar_chart(size_df.set_index("Cluster"))

    with metric_tabs[1]:
        # PCA table
        st.subheader("PCA Component Distribution")
        st.caption("A scree-style view of variance captured by each principal component. If the first few PCs capture most variance, clustering in PCA space is more trustworthy.")

        # keep only the displayed components (or you can plot full if you have it)
        plot_df = pca_table.copy()

        # Bar: % variance
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=plot_df["Component"],
            y=plot_df["% of Variance"],
            name="% of Variance"
        ))

        # Line: cumulative %
        fig.add_trace(go.Scatter(
            x=plot_df["Component"],
            y=plot_df["Cumulative %"],
            mode="lines+markers",
            name="Cumulative %",
            yaxis="y2"
        ))

        fig.update_layout(
            title="PCA Scree Plot with Cumulative Variance",
            xaxis_title="Principal Components",
            yaxis=dict(title="% of Variance"),
            yaxis2=dict(
                title="Cumulative %",
                overlaying="y",
                side="right",
                range=[0, 100]
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
            margin=dict(l=40, r=40, t=60, b=40),
        )

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show PCA table"):
            st.dataframe(pca_table, use_container_width=True)



    with metric_tabs[2]:
        # Salary summary
        st.subheader("Cluster Cost Summary")
        st.caption("Compares salary statistics by cluster. Large gaps in mean/median suggest clusters reflect economically distinct roles.")


        cs = cost_summary.copy()
        cols = {c.lower(): c for c in cs.columns}

        cluster_col = cols.get("cluster")
        mean_col = cols.get("mean") or cols.get("avg") or cols.get("average")
        median_col = cols.get("median")
        min_col = cols.get("min")
        max_col = cols.get("max")

        if not (cluster_col and mean_col):
            st.info("Cost summary table format didn't match expected columns for plotting.")
            st.dataframe(cost_summary, use_container_width=True)
        else:
            cs = cs.sort_values(mean_col)

            fig = go.Figure()

            # --- Mean bars ---
            fig.add_trace(go.Bar(
                x=cs[cluster_col],
                y=cs[mean_col],
                name="Mean"
            ))

            # --- Min/Max as error bars around mean (if available) ---
            if min_col and max_col:
                err_plus = (cs[max_col] - cs[mean_col]).clip(lower=0)
                err_minus = (cs[mean_col] - cs[min_col]).clip(lower=0)

                fig.update_traces(
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=err_plus,
                        arrayminus=err_minus
                    )
                )

            # --- Median markers (if available) ---
            if median_col:
                fig.add_trace(go.Scatter(
                    x=cs[cluster_col],
                    y=cs[median_col],
                    mode="markers",
                    name="Median",
                    marker=dict(size=10, symbol="diamond")
                ))

            fig.update_layout(
                title="Cluster Salary Summary (Mean with Min/Max Range + Median)",
                xaxis_title="Cluster",
                yaxis_title="Normalized Salary",
                barmode="overlay",
                height=500,
                margin=dict(l=40, r=20, t=60, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Show cost summary table"):
                st.dataframe(cost_summary, use_container_width=True)


    with metric_tabs[3]:
        st.subheader("Cluster Compactness Comparison (t-test)")
        st.caption("Tests whether each cluster’s distance-to-centroid distribution differs from the rest of the dataset. Lower mean distance with a low p-value suggests a tighter, more distinct cluster.")

        # distances from each point to each centroid
        D = np.linalg.norm(Xpca[:, None, :] - centroids_pca[:, :Xpca.shape[1]], axis=2)

        ttest_results = []
        for c in range(num_clusters):
            # distances of points in cluster c to centroid c
            cluster_dists = D[labels == c, c]

            # distances of points in other clusters to their own centroids
            rest_idx = np.where(labels != c)[0]
            rest_dists = D[rest_idx, labels[rest_idx]]

            # guard against tiny clusters
            if len(cluster_dists) < 2 or len(rest_dists) < 2:
                ttest_results.append({
                    "Cluster": c,
                    "Mean Distance": np.nan,
                    "Rest Mean Distance": np.nan,
                    "t-statistic": np.nan,
                    "p-value": np.nan
                })
                continue

            t_stat, p_value = ttest_ind(cluster_dists, rest_dists, equal_var=False)

            ttest_results.append({
                "Cluster": c,
                "Mean Distance": round(cluster_dists.mean(), 4),
                "Rest Mean Distance": round(rest_dists.mean(), 4),
                "t-statistic": round(t_stat, 4),
                "p-value": round(p_value, 6)
            })

        ttest_df = pd.DataFrame(ttest_results)
        st.dataframe(ttest_df, use_container_width=True)

        # ---- Add a plot for interpretability ----
        st.caption(
            "Visual comparison of each cluster’s average distance-to-centroid "
            "vs the rest of the dataset. Use p-values to judge whether differences "
            "are likely meaningful."
        )

        plot_df = ttest_df.melt(
            id_vars=["Cluster", "t-statistic", "p-value"],
            value_vars=["Mean Distance", "Rest Mean Distance"],
            var_name="Group",
            value_name="Avg Distance"
        )

        fig_t = px.bar(
            plot_df,
            x="Cluster",
            y="Avg Distance",
            color="Group",
            barmode="group",
            hover_data={
                "t-statistic": True,
                "p-value": True,
                "Avg Distance": ":.4f"
            },
            title="Cluster Compactness: Mean Distance vs Rest"
        )

        fig_t.update_layout(
            xaxis_title="Cluster",
            yaxis_title="Average Distance to Assigned Centroid",
            height=450,
            margin=dict(l=40, r=20, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        st.plotly_chart(fig_t, use_container_width=True)




# help section
with tabs[3]:
    st.subheader("How to read these metrics")
    st.markdown(
        r"""
**Silhouette Coefficient**

The silhouette value for a point *i* is defined as:

$$
s = \frac{b - a}{\max(a, b)}
$$

Where:
- \(a\): the average distance of *i* to points in the same cluster 
- \(b\): the minimum average distance of *i* to points in other clusters 

Interpretation
- The closer $s$ is to 1 the better
- Negative $s$ value is undesirable

The silhouette coefficient is found by calculating $s$ for all points and averaging them out.

---

**Davies–Bouldin Index (DBI)**

*Where $X$ is the dataset*
$$
DB = \frac{1}{k} \sum_{i=1}^{k}
\max_{j \ne i}
\left(
\frac{\Delta(X_i) + \Delta(X_j)}{\delta(X_i, X_j)}
\right)
$$

This index uses cluster scatter (spread) and separation (centroid distances).

Interpretation
- \($DB$ < 1.0\): Good clustering  
- \(1.0 < $DB$ < 2.0\): Moderate clustering
- \($DB$ > 2.0\): Bad clustering

---

**Calinski–Harabasz Index (CH Index)**

Based on ratio of separation (BSS) to cohesion (WSS):

$$
CH = \frac{BSS / (K - 1)}{WSS / (N - K)}
$$

Interpretation
- Higher CH index indicates dense and well-separated clusters.
---

**PCA Component Distribution**

Variance explained by each principal component.

Interpretation
- First Principal Component should capture most of variance
- Every component after should progressively capture less 

---

**Cluster Cost Summary (Salary)**

Salary data of each of the clusters \(with cost being salary\).

Interpretation
- Large differences in mean/median salary in clusters show that job-skill patterns have been captured in clusters
- Overlapping salary ranges show clusters may be structurally fine but economically similar 

---

**t-Test for Cluster Distinctiveness**

Tests whether cluster \(i\)'s members are significantly closer to  
their centroid than non-members.

Interpretation
- \($p$ < 0.01\): highly distinct cluster  
- \($p$ < 0.05\): statistically distinct cluster  
- \(0.05 < $p$ < 0.15\): borderline separation  
- \(0.15 > $p$\): weak or no meaningful separation  
---
"""
    )


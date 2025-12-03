import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from clustering_task.clustering_algo import KMeansScratch, pca_transform
from scipy.stats import ttest_ind
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import plotly.express as px


# getting skills data
skill_cols = set(st.session_state.job_market_filtered_df.columns[31:])
skill_cols.add('normalized_salary')

skills_df = st.session_state.job_market_filtered_df.loc[:, st.session_state.job_market_filtered_df.columns.isin(skill_cols)]
skills_df["company_name"] = st.session_state.job_market_filtered_df["company_name"]
skills_df["title"] = st.session_state.job_market_filtered_df["title"]
skills_df["posting_id"] = st.session_state.job_market_filtered_df["posting_id"]

# build X using only real skill columns
feature_cols = [c for c in skills_df.columns if c not in ("company_name", "title", "posting_id", "normalized_salary")]
X = skills_df[feature_cols].to_numpy(float)

st.set_page_config(page_title="Job Clustering Dashboard", layout="wide")
st.title("Job Landscape — What Jobs Are Similar")

st.caption(
    "Explore job clusters built from skill vectors using PCA and K-Means. "
    "Use this to understand how skill sets group together across the job market."
)

with st.expander("What is this page showing?", expanded=False):
    st.markdown(
        """
        This page performs **unsupervised clustering** on the skill columns of your filtered job postings,
        using PCA for dimensionality reduction and K-Means for cluster formation.

        **You can:**
        - Choose the **number of clusters (k)** to create different granularities of job grouping.
        - Select a **distance function** (Euclidean, Manhattan, Minkowski) for K-Means.
        - Control the number of **PCA components** used to project job-skill vectors.
        - Pick an **evaluation metric** (Silhouette, Davies–Bouldin, Calinski–Harabasz).

        **Tabs:**
        - **3D Cluster Map** – view clusters in 3D PCA space, with centroids and job skill hover labels.  
        - **Similarity Matrix** – inspect pairwise PCA similarity between job postings as a heatmap.  
        - **Evaluation** – cluster scores, PCA variance breakdown, centroid distances, and t-test results.  
        - **Help** – formal definitions and interpretation ranges for all metrics.
        """
    )


# inputs
st.header("Input Parameters")

col1, col2, col3 = st.columns(3)
with col1:
    num_clusters = st.number_input(
        "k",
        2, 100, 5, step=1,
        help="The number of clusters made by the K-Means algorithm."
    )

with col2:
    num_pca_components = st.number_input(
        "PCA Components",
        3, 50, 3,
        help="Choose amount of PCA components. (More PCA components preserve more variance but result in less interpretability)."
    )

with col3:
    num_iterations = st.number_input(
        "Number of Iterations",
        10, 1000, 100, step=10,
        help="The amount of iterations the K-Means algorithm will run, mostly for the purpose of refining centroids."
    )

col4, col5 = st.columns(2)
with col4:
    distance_function = st.selectbox(
        "Distance Function",
        ["euclidean", "manhattan", "minkowski"],
        help="The distance metric which is used to compare points to centroids."
    )

    minkowski_p = None
    if distance_function == "minkowski":
        minkowski_p = st.slider(
            "Minkowski Power (p)",
            min_value=3.0, max_value=5.0, value=3.0, step=0.1,
            help="Exponent that determines sensitivity to larger differences."
        )

with col5:
    evaluation_function = st.selectbox(
        "Evaluation Function",
        ["silhouette", "davies-bouldin", "calinski-harabasz"],
        help="Determines which evaluation score to use of the K-Means output."
    )

st.markdown("---")


# tab set up
tabs = st.tabs(["3D Cluster Map", "Similarity Matrix", "Evaluation", "Help"])


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
    st.header("Skill Clusters in PCA Feature Space")
    st.caption("Jobs projected into 3D PCA space, colored based on the cluster they are apart of.")

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
    # cosine similarity
    cos_sim = 1 - squareform(pdist(Xpca, metric='euclidean'))

    sim_df = pd.DataFrame(cos_sim, index=job_labels, columns=job_labels)

    fig = px.imshow(
        sim_df,
        color_continuous_scale="Viridis",
        aspect="auto",
        labels=dict(color="Similarity")
    )

    # make labels small
    fig.update_xaxes(tickfont=dict(size=6))
    fig.update_yaxes(tickfont=dict(size=6))

    fig.update_layout(
        height=800, 
        margin=dict(l=10, r=10, t=30, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)


# evaluation tab
with tabs[2]:
    st.header("Evaluation Metrics & Distances")

    # Evaluation score
    if evaluation_function == "silhouette":
        score = silhouette_score(X, labels)
    elif evaluation_function == "davies-bouldin":
        score = davies_bouldin_score(X, labels)
    else:
        score = calinski_harabasz_score(X, labels)

    st.metric(label=f"{evaluation_function.title()} Score", value=round(score, 4))

    # Cluster size distribution
    st.subheader("Cluster Sizes")
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    size_df = pd.DataFrame({"Cluster": cluster_sizes.index, "Size": cluster_sizes.values})
    st.bar_chart(size_df.set_index("Cluster"))
    
    # PCA table
    st.subheader("PCA Component Distribution")
    st.dataframe(pca_table, width='stretch')

    # Salary summary
    st.subheader("Cluster Cost Summary")
    st.dataframe(cost_summary, width='stretch')

    # t-test
    st.subheader("t-Test")

    D = np.linalg.norm(Xpca[:, None, :] - centroids_pca[:, :Xpca.shape[1]], axis=2)
    all_distances = np.array([D[i, lbl] for i, lbl in enumerate(labels)])

    ttest_results = []
    for c in range(num_clusters):
        cluster_dists = all_distances[np.where(labels == c)]
        rest_dists = all_distances[np.where(labels != c)]

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
- \(0,15 > $p$\): weak or no meaningful separation  
---
"""
    )


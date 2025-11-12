import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from clustering_task.clustering_algo import KMeansScratch, pca_transform

# getting skills data
skill_cols = list(st.session_state.job_market_filtered_df.columns[31:])
if "normalized_salary" in st.session_state.job_market_filtered_df.columns:
    skill_cols.append("normalized_salary")

skills_df = st.session_state.job_market_filtered_df[skill_cols].fillna(0)

# get skills
feature_cols = [c for c in skills_df.columns if c != "normalized_salary"]
# data for KMeans
X = skills_df[feature_cols].to_numpy(dtype=float)

# page set up
st.set_page_config(page_title="Job Clustering Dashboard", layout="wide")
st.title("Job Landscape — Skill Clustering Explorer")

tabs = st.tabs(
    ["Input Parameters", "3D Cluster Map", "Contour Map", "Evaluation"]
)

with tabs[0]:
    st.header("Input Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        num_clusters = st.number_input("Number of Clusters (k)", 2, 20, 5, step=1)
    with col2:
        num_pca_components = st.number_input("Number of PCA Components", 3, 10, 3)
    with col3:
        num_iterations = st.number_input("Number of Iterations", 10, 1000, 100, step=10)
    col4, col5 = st.columns(2)
    with col4:
        distance_function = st.selectbox("Distance Function", ["euclidean", "manhattan", "minkowski"])
        # allow user to set p parameter for minkowski
        minkowski_p = None
        if distance_function == "minkowski":
            minkowski_p = st.slider(
                "Minkowski Power (p)",
                min_value=3.0, max_value=5.0, value=3.0, step=0.1,
            )
    with col5:
        evaluation_function = st.selectbox(
            "Evaluation Function", ["silhouette", "davies-bouldin", "calinski-harabasz"]
        )

## K-means algorithm

# PCA
Xpca, comps, mu = pca_transform(X, n_components=num_pca_components)
Xpca, comps, mu = pca_transform(X, n_components=num_pca_components)

# pca data for table
eigenvalues = np.var(Xpca, axis=0)
total_variance = np.var(X, axis=0).sum()
percent_variance = (eigenvalues / total_variance) * 100
cumulative_variance = np.cumsum(percent_variance)

# create a PCA distribution table
pca_table = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(num_pca_components)],
    "Eigenvalue (Total)": np.round(eigenvalues, 3),
    "% of Variance": np.round(percent_variance, 3),
    "Cumulative %": np.round(cumulative_variance, 3)
})

# kmeans algorithm
km = KMeansScratch(
    n_clusters=num_clusters,
    max_iters=num_iterations,
    distance=distance_function 
).fit(X)
labels, centroids = km.labels, km.centroids
centroids_pca = (centroids - mu) @ comps[:, :num_pca_components]
pca_labels = [f"PC{i+1}" for i in range(num_pca_components)]

# salary summary
if "normalized_salary" in skills_df.columns:
    salary_df = skills_df.assign(Cluster=labels)
    salary_df = salary_df[salary_df["normalized_salary"] > 0]
    cost_summary = (
        salary_df.groupby("Cluster")["normalized_salary"]
        .agg(["mean", "median", "min", "max", "count"])
        .round(2)
        .reset_index()
        .rename(columns={"count": "Size"})
    )
else:
    cost_summary = pd.DataFrame()

# top skills in a cluster
cluster_skill_strength = (
    skills_df[feature_cols]
    .assign(Cluster=labels)
    .groupby("Cluster")
    .mean()
)
top_skills_per_cluster = {
    c: cluster_skill_strength.loc[c].sort_values(ascending=False).head(5).index.tolist()
    for c in cluster_skill_strength.index
}

with tabs[1]:
    st.header("Skill Clusters in PCA Feature Space")
    # pca axises
    ix, iy, iz = 0, 1, 2
    # axises labels
    ax_x, ax_y, ax_z = pca_labels[ix], pca_labels[iy], pca_labels[iz]

    # Create hover text with top skills for each job
    hover_texts = []
    for i in range(X.shape[0]):
        job_skills = [feature_cols[j] for j in np.where(X[i] > 0)[0]]
        # limit to top 5 to avoid clutter
        hover_texts.append(", ".join(job_skills[:5]) if job_skills else "(No skills listed)")

    # Build the DataFrame for 3D scatter
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
        hover_name="Skills",  # 👈 shows skills on hover
        opacity=0.8,
        color_discrete_sequence=px.colors.qualitative.Vivid
    )

    # centroids
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


with tabs[2]:
    st.header("Contour Map")

    ix2, iy2 = 0, 1
    ax_x2, ax_y2 = pca_labels[ix2], pca_labels[iy2]

    df2d = pd.DataFrame({
        "x": Xpca[:, ix2],
        "y": Xpca[:, iy2],
        "Cluster": labels
    })

    fig2d = px.density_contour(
        df2d, x="x", y="y", color=df2d["Cluster"].astype(str),
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )

    # Only modify contour traces
    fig2d.update_traces(
        selector=dict(type='contour'),
        contours_coloring="fill",
        contours_showlabels=True
    )

    st.plotly_chart(fig2d, width='stretch')

with tabs[3]:
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
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    size_df = pd.DataFrame({"Cluster": cluster_sizes.index, "Size": cluster_sizes.values})
    st.subheader("Cluster Sizes")
    st.bar_chart(size_df.set_index("Cluster"))

    # Distance-to-centroid distribution (Euclidean)
    D = np.linalg.norm(Xpca[:, None, :] - centroids_pca[:, :Xpca.shape[1]], axis=2)
    dist_df = pd.DataFrame({
        "Cluster": labels,
        "DistanceToCentroid": [D[i, lbl] for i, lbl in enumerate(labels)]
    })

    # PCA 
    st.subheader("PCA Component Distribution")
    st.dataframe(pca_table, width='stretch')

    # Cluster cost
    st.subheader("Cluster Cost Summary")
    st.dataframe(cost_summary, width='stretch')

    
    # What are the top skills in a cluster
    st.subheader("Top Skills per Cluster")
    for cluster_id, skills in top_skills_per_cluster.items():
        st.markdown(f"**Cluster {cluster_id}** — Top Skills:")
        st.write(", ".join(skills))
        st.markdown("---")


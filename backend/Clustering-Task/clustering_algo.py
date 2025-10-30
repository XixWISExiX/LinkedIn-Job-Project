"""
Clustering Task 

Usage:
    python Clustering-Task/clustering_algo.py --config=Clustering-Task/clustering_args.yml
"""

import os
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Any, Dict, Tuple
from sklearn.metrics import silhouette_score  # For cluster quality evaluation

# avoid latex errors
mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.default'] = 'regular'

from pre_processing import *


# argument parsing
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Job Clustering Dashboard", fromfile_prefix_chars='@')
    p.add_argument("--company-filter", type=str, default=None)
    p.add_argument("--title-filter", type=str, default=None)
    p.add_argument("--job-postings-csv-in", type=str, default="datasets/archive/postings.csv")
    p.add_argument("--skills-lexicon-csv-in", type=str, default="datasets/universal_skills_catalog.csv")
    p.add_argument("--max-phrase-len", type=int, default=3)
    p.add_argument("--min-skill-df", type=int, default=10)
    p.add_argument("--k-values", type=int, nargs="+", default=[3,5,7])
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--normalize", type=str, default="true", choices=["true","false"])
    p.add_argument("--plot-out-dir", type=str, default="datasets/output/plots")
    p.add_argument("--plot-prefix", type=str, default="clusters")
    p.add_argument("--show-plots", action="store_true")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config")
    return p


def flatten(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def yaml_keys_to_arg_names(flat_cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in flat_cfg.items():
        key = k.split(".")[-1].replace("-", "_")
        out[key] = v
    return out


def pca_transform(X: np.ndarray, n_components: int = 2, center: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0) if center else np.zeros(X.shape[1])
    Xc = X - mu
    C = np.cov(Xc, rowvar=False)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    comps = vecs[:, order][:, :n_components]
    return Xc @ comps, comps, mu


class KMeans:
    def __init__(self, n_clusters, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None

    def fit(self, X):
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iters):
            # Assign each data point to the nearest centroid
            labels = self._assign_labels(X)

            # Update centroids
            new_centroids = self._update_centroids(X, labels)

            # Check for convergence
            if np.allclose(self.centroids, new_centroids, atol=1e-6):
                break

            self.centroids = new_centroids

        self.labels = labels
        return self

    def _assign_labels(self, X):
        # Compute distances from each data point to centroids
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        # Assign labels based on the nearest centroid
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else self.centroids[i]
            for i in range(self.n_clusters)
        ])
        return new_centroids
    
# plotters
def plot_3d_clusters(X3, labels, centers3, out_path, title, cluster_names=None):
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X3[:,0], X3[:,1], X3[:,2], c=labels, s=20, alpha=0.8, cmap='tab10')
    if centers3 is not None:
        ax.scatter(centers3[:,0], centers3[:,1], centers3[:,2], c="black", s=100, marker="x")
        for i, (x, y, z) in enumerate(centers3):
            ax.text(x, y, z, f"C{i}", color="black", fontsize=9, weight="bold")
    if cluster_names:
        patches = [mpatches.Patch(color=scatter.cmap(scatter.norm(i)),
                                  label=f"Cluster {i}: {cluster_names.get(i,'N/A')}") for i in sorted(set(labels))]
        ax.legend(handles=patches, loc="best")
    ax.set_title(title)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    return out_path


def plot_contour_regions(X2, labels, centers2, out_path, title, cluster_names=None):
    x_min, x_max = X2[:,0].min()-0.5, X2[:,0].max()+0.5
    y_min, y_max = X2[:,1].min()-0.5, X2[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    dists = np.stack([np.sum((grid - c)**2, axis=1) for c in centers2], axis=1)
    Z = np.argmin(dists, axis=1).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(centers2.shape[0]+1)-0.5, cmap='tab10')
    scatter = ax.scatter(X2[:,0], X2[:,1], c=labels, s=12, edgecolor="none", cmap='tab10')
    ax.scatter(centers2[:,0], centers2[:,1], c="black", s=80, marker="x")
    for i, (x, y) in enumerate(centers2):
        ax.text(x, y, f"C{i}", color="black", fontsize=9, weight="bold")
    if cluster_names:
        patches = [mpatches.Patch(color=scatter.cmap(scatter.norm(i)),
                                  label=f"Cluster {i}: {cluster_names.get(i,'N/A')}") for i in sorted(set(labels))]
        ax.legend(handles=patches, loc="best")
    ax.set_title(title)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    return out_path

if __name__ == "__main__":
    parser = build_arg_parser()
    args, _ = parser.parse_known_args()
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        parser.set_defaults(**yaml_keys_to_arg_names(flatten(cfg)))
    args = parser.parse_args()

    X, vocab, df_filtered = build_skill_matrix(
        args.job_postings_csv_in, args.skills_lexicon_csv_in,
        args.company_filter, args.title_filter,
        args.max_phrase_len, args.min_skill_df
    )

    normalize_flag = str(args.normalize).lower() == "true" if not isinstance(args.normalize, bool) else args.normalize
    Xn, _, _ = zscore(X.astype(float)) if normalize_flag else (X.astype(float), None, None)
    X2, comps2, mu2 = pca_transform(Xn, 2)
    X3, comps3, mu3 = pca_transform(Xn, 3)
    os.makedirs(args.plot_out_dir, exist_ok=True)

    for k in args.k_values:
        km = KMeans(n_clusters=k, max_iters=args.max_iter).fit(Xn)
        labels, centers = km.labels, km.centroids
        centers2 = (centers - mu3) @ comps2
        centers3 = (centers - mu3) @ comps3

        cluster_names = get_cluster_skill_labels(Xn, labels, vocab, top_n=3)

        out3d = os.path.join(args.plot_out_dir, f"{args.plot_prefix}_k{k}_3d.png")
        plot_3d_clusters(X3, labels, centers3, out3d, f"Job Similarity Clusters (k={k}) — 3D PCA", cluster_names)

        out2d = os.path.join(args.plot_out_dir, f"{args.plot_prefix}_k{k}_contour.png")
        plot_contour_regions(X2, labels, centers2, out2d, f"Cluster Regions (k={k}) — PCA Contour", cluster_names)

        # Silhouette score
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(Xn, labels)
            print(f"Silhouette Score (k={k}): {sil:.4f}")
        else:
            print(f"Silhouette Score (k={k}): N/A (only one cluster)")

    for k in args.k_values:
        print(f" - {args.plot_out_dir}/{args.plot_prefix}_k{k}_3d.png")
        print(f" - {args.plot_out_dir}/{args.plot_prefix}_k{k}_contour.png")

    if args.show_plots:
        plt.show()

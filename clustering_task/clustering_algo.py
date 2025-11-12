import numpy as np
from typing import Tuple


def pca_transform(X: np.ndarray, n_components: int = 2, center: bool = True):
    mu = X.mean(axis=0) if center else np.zeros(X.shape[1])
    Xc = X - mu
    C = np.cov(Xc, rowvar=False)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    comps = vecs[:, order][:, :n_components]
    return Xc @ comps, comps, mu

class KMeansScratch:
    def __init__(self, n_clusters: int, max_iters: int = 100, distance: str = "euclidean", p: int = 3):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.distance = distance.lower()
        self.p = p

    def fit(self, X: np.ndarray):
        rng = np.random.default_rng(42)
        self.centroids = X[rng.choice(X.shape[0], self.n_clusters, replace=False)]
        for _ in range(self.max_iters):
            labels = self._assign_labels(X)
            new_centroids = self._update_centroids(X, labels)
            if np.allclose(self.centroids, new_centroids, atol=1e-6):
                break
            self.centroids = new_centroids

        self.labels = labels
        return self

    # distance calculator
    def _pairwise_distances(self, X: np.ndarray, C: np.ndarray):
        if self.distance == "euclidean":
            return np.sqrt(((X[:, None] - C) ** 2).sum(axis=2))

        elif self.distance == "manhattan":
            return np.abs(X[:, None] - C).sum(axis=2)

        elif self.distance == "minkowski":
            return (((np.abs(X[:, None] - C)) ** self.p).sum(axis=2)) ** (1 / self.p)

    def _assign_labels(self, X: np.ndarray):
        distances = self._pairwise_distances(X, self.centroids)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray):
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else self.centroids[i]
            for i in range(self.n_clusters)
        ])
        return new_centroids

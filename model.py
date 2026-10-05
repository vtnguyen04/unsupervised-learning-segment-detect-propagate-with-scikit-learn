"""
Unsupervised Learning: Segment, Detect, Propagate with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - blobs_data
import numpy as np
from sklearn.datasets import make_blobs


def blobs_data(random_state=42):
    centers = np.array(
        [[0.2, 2.3], [-1.5, 2.3], [-2.8, 1.8], [-2.8, 2.8], [-2.8, 1.3]]
    )
    cluster_std = [0.4, 0.3, 0.1, 0.1, 0.1]

    X, y = make_blobs(
        n_samples=2000,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    return X, y

# Step 2 - fit_kmeans
from sklearn.cluster import KMeans


def fit_kmeans(X, k, random_state=42):
    return KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)

# Step 3 - inertia_curve
def inertia_curve(X, ks):
    return {k: float(fit_kmeans(X, k).inertia_) for k in ks}

# Step 4 - silhouette_curve
from sklearn.metrics import silhouette_score


def silhouette_curve(X, ks):
    return {
        k: float(silhouette_score(X, fit_kmeans(X, k).labels_))
        for k in ks
        if k >= 2
    }


def best_k_by_silhouette(curve):
    return max(curve, key=lambda k: (curve[k], -k))

# Step 5 - fit_dbscan
from sklearn.cluster import DBSCAN


def fit_dbscan(X, eps=0.2, min_samples=5):
    return DBSCAN(eps=eps, min_samples=min_samples).fit(X)


def dbscan_summary(dbscan):
    labels = dbscan.labels_
    unique_clusters = set(labels) - {-1}
    n_clusters = int(len(unique_clusters))
    n_noise = int((labels == -1).sum())
    n_core = int(len(dbscan.core_sample_indices_))

    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "n_core": n_core,
    }

# Step 6 - dbscan_predict
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def dbscan_predict(dbscan, X_new, n_neighbors=50):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    X_core = dbscan.components_
    y_core = dbscan.labels_[dbscan.core_sample_indices_]

    knn.fit(X_core, y_core)
    return knn.predict(X_new).astype(int)

# Step 7 - fit_gmm
from sklearn.mixture import GaussianMixture


def fit_gmm(X, n_components, random_state=42):
    return GaussianMixture(
        n_components=n_components, n_init=10, random_state=random_state
    ).fit(X)


def bic_curve(X, ks):
    return {k: float(fit_gmm(X, k).bic(X)) for k in ks}

# Step 8 - flag_anomalies
from sklearn.mixture import GaussianMixture


def fit_gmm(X, n_components, random_state=42):
    return GaussianMixture(
        n_components=n_components, n_init=10, random_state=random_state
    ).fit(X)


def bic_curve(X, ks):
    return {k: float(fit_gmm(X, k).bic(X)) for k in ks}

# Step 9 - digits_data (not yet solved)
# TODO: implement

# Step 10 - baseline_50_random (not yet solved)
# TODO: implement

# Step 11 - representative_digits (not yet solved)
# TODO: implement

# Step 12 - train_on_representatives (not yet solved)
# TODO: implement

# Step 13 - propagate_and_train (not yet solved)
# TODO: implement

# Step 14 - synthetic_image (not yet solved)
# TODO: implement

# Step 15 - segment_colors (not yet solved)
# TODO: implement

# Step 16 - save_and_reload_clusterer (not yet solved)
# TODO: implement

# Step 17 - predict_digit_labels (not yet solved)
# TODO: implement


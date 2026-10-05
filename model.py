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
import numpy as np


def flag_anomalies(gmm, X, contamination=0.04):
    densities = gmm.score_samples(X)
    threshold = np.percentile(densities, 100 * contamination)
    return densities < threshold

# Step 9 - digits_data
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def digits_data(test_size=0.25, random_state=42):
    digits = load_digits()
    X, y = digits.data, digits.target
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

# Step 10 - baseline_50_random
def baseline_50_random(
    X_train, y_train, X_test, y_test, n_labeled=50, random_state=42
):
    clf = LogisticRegression(max_iter=10000, random_state=random_state)
    clf.fit(X_train[:n_labeled], y_train[:n_labeled])
    return float(clf.score(X_test, y_test))

# Step 11 - representative_digits
def representative_digits(X_train, k=50, random_state=42):
    kmeans = fit_kmeans(X_train, k, random_state=random_state)
    distances = kmeans.transform(X_train)
    rep_idx = np.argmin(distances, axis=0)
    return kmeans, rep_idx.astype(int)

# Step 12 - train_on_representatives
import numpy as np
from sklearn.linear_model import LogisticRegression


def train_on_representatives(X_train, y_train, rep_idx, X_test, y_test):
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train[rep_idx], y_train[rep_idx])
    return float(clf.score(X_test, y_test))

# Step 13 - propagate_and_train
def propagate_and_train(
    X_train,
    y_train,
    kmeans,
    rep_idx,
    X_test,
    y_test,
    percentile=20,
):
    cluster_labels = kmeans.labels_
    distances = kmeans.transform(X_train)

    sample_distances = distances[np.arange(len(X_train)), cluster_labels]

    rep_labels = y_train[rep_idx]
    y_train_propagated = rep_labels[cluster_labels]

    selected_mask = np.zeros(len(X_train), dtype=bool)

    # Filter points within each cluster based on distance percentile
    for j in range(kmeans.n_clusters):
        in_cluster = cluster_labels == j
        cluster_dists = sample_distances[in_cluster]
        if len(cluster_dists) > 0:
            cutoff = np.percentile(cluster_dists, percentile)
            selected_mask[in_cluster] = cluster_dists <= cutoff

    X_train_prop = X_train[selected_mask]
    y_prop = y_train_propagated[selected_mask]
    y_true_selected = y_train[selected_mask]

    n_propagated = int(selected_mask.sum())
    label_accuracy = float(np.mean(y_prop == y_true_selected))

    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train_prop, y_prop)
    test_accuracy = float(clf.score(X_test, y_test))

    return {
        "n_propagated": n_propagated,
        "label_accuracy": label_accuracy,
        "test_accuracy": test_accuracy,
    }

# Step 14 - synthetic_image
def synthetic_image(size=48):
    img = np.zeros((size, size, 3), dtype=np.float64)
    mid = size // 2

    g_ramp = np.linspace(0.0, 1.0, mid)

    img[:, :mid, 0] = 1.0
    img[:, :mid, 1] = g_ramp
    img[:, :mid, 2] = 0.0

    img[:, mid:, 0] = 0.0
    img[:, mid:, 1] = g_ramp
    img[:, mid:, 2] = 1.0

    sq = size // 4
    img[:sq, :sq, :] = 1.0

    return img

# Step 15 - segment_colors (not yet solved)
# TODO: implement

# Step 16 - save_and_reload_clusterer (not yet solved)
# TODO: implement

# Step 17 - predict_digit_labels (not yet solved)
# TODO: implement


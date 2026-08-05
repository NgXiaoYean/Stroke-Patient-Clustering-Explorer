# Shared files for Evaluation 
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score


@dataclass
class ClusteringResult:
    data: pd.DataFrame
    features: np.ndarray
    projection_2d: np.ndarray
    labels: np.ndarray
    n_components: int
    n_clusters: int
    noise_ratio: float
    silhouette: float
    davies_bouldin: float
    cluster_summary: pd.DataFrame
    pca_full_variance_ratio: np.ndarray
    pca_selected_loadings: pd.DataFrame
    preprocessed_feature_names: list[str]
    preprocessed_data: np.ndarray
    
    # DBSCAN-specific configuration placeholders (K-Means/MeanShift may Ignore)
    suggested_eps: float | None = None
    selected_eps: float | None = None
    min_samples: int | None = None
    k_distances: np.ndarray | None = None
    parameter_results: pd.DataFrame | None = None


def calculate_clustering_metrics(features: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    # SIlhouette & Davies-Bouldin index 
    non_noise = labels != -1
    clusters = len(set(labels[non_noise]))
    noise = float(np.mean(~non_noise))
    if clusters < 2 or non_noise.sum() <= clusters:
        return {
            "n_clusters": clusters,
            "noise_ratio": noise,
            "silhouette": np.nan,
            "davies_bouldin": np.nan
        }
    return {
        "n_clusters": clusters,
        "noise_ratio": noise,
        "silhouette": float(silhouette_score(features[non_noise], labels[non_noise])),
        "davies_bouldin": float(davies_bouldin_score(features[non_noise], labels[non_noise])),
    }

"""K-Means clustering module for Stroke Patient Clustering Explorer.

The stroke outcome is excluded from clustering and is used only afterwards
for clinical interpretation of the discovered patient groups.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

from data_processing import clean_data, preprocess_data, apply_pca, calculate_cluster_summary
from evaluation import ClusteringResult

RANDOM_STATE = 42
KMeansResult = ClusteringResult


@dataclass(frozen=True)
class KMeansConfig:
    n_clusters: Optional[int] = None
    min_k: int = 2
    max_k: int = 10
    pca_variance: float = 0.90
    risk_multiplier: float = 1.5
    n_init: int = 20


def _evaluate_k(features: np.ndarray, k: int, n_init: int) -> dict[str, float | int]:
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=n_init)
    labels = model.fit_predict(features)
    return {
        "k": k,
        "inertia": float(model.inertia_),
        "silhouette": float(silhouette_score(features, labels)),
        "davies_bouldin": float(davies_bouldin_score(features, labels)),
    }


def run_kmeans(data: pd.DataFrame, config: KMeansConfig = KMeansConfig()) -> KMeansResult:
    """Fit K-Means on PCA-transformed patient features."""
    if not 0 < config.pca_variance <= 1:
        raise ValueError("pca_variance must be between 0 and 1.")
    if config.min_k < 2:
        raise ValueError("min_k must be at least 2.")
    if config.max_k < config.min_k:
        raise ValueError("max_k must be greater than or equal to min_k.")

    cleaned = clean_data(data)
    processed, feature_names = preprocess_data(cleaned)
    features, projection_2d, pca_full_variance_ratio, pca_selected_loadings = apply_pca(
        processed, feature_names, config.pca_variance
    )

    upper_k = min(config.max_k, len(features) - 1)
    if upper_k < config.min_k:
        raise ValueError("Not enough patients to evaluate the requested K range.")

    rows = [_evaluate_k(features, k, config.n_init) for k in range(config.min_k, upper_k + 1)]
    search = pd.DataFrame(rows)

    if config.n_clusters is None:
        # Maximise Silhouette; use Davies-Bouldin and smaller K as tie-breakers.
        selected_k = int(
            search.sort_values(
                ["silhouette", "davies_bouldin", "k"],
                ascending=[False, True, True],
            ).iloc[0]["k"]
        )
    else:
        selected_k = int(config.n_clusters)
        if selected_k < 2 or selected_k >= len(features):
            raise ValueError("n_clusters must be at least 2 and smaller than the number of patients.")

    model = KMeans(n_clusters=selected_k, random_state=RANDOM_STATE, n_init=config.n_init)
    labels = model.fit_predict(features)
    silhouette = float(silhouette_score(features, labels))
    db_index = float(davies_bouldin_score(features, labels))

    result_data = cleaned.copy()
    result_data["cluster"] = labels
    cluster_summary = calculate_cluster_summary(result_data, config.risk_multiplier)

    return KMeansResult(
        data=result_data,
        features=features,
        projection_2d=projection_2d,
        labels=labels,
        n_components=features.shape[1],
        n_clusters=selected_k,
        noise_ratio=0.0,
        silhouette=silhouette,
        davies_bouldin=db_index,
        cluster_summary=cluster_summary,
        pca_full_variance_ratio=pca_full_variance_ratio,
        pca_selected_loadings=pca_selected_loadings,
        preprocessed_feature_names=feature_names,
        preprocessed_data=processed,
        parameter_results=search,
        selected_k=selected_k,
        inertia=float(model.inertia_),
    )


def main() -> None:
    path = Path(__file__).with_name("brain_stroke.csv")
    result = run_kmeans(pd.read_csv(path))
    print(f"Selected K         : {result.selected_k}")
    print(f"PCA components     : {result.n_components}")
    print(f"Silhouette score   : {result.silhouette:.4f}")
    print(f"Davies-Bouldin     : {result.davies_bouldin:.4f}")
    print(f"Inertia            : {result.inertia:.4f}")


if __name__ == "__main__":
    main()

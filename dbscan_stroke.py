from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from data_processing import (
    clean_data,
    preprocess_data,
    apply_pca,
    calculate_cluster_summary
)

RANDOM_STATE = 42


@dataclass(frozen=True)
class DBSCANConfig:
    # Let Machine set the eps automatically 
    eps: Optional[float] = None
    min_samples: Optional[int] = None
    pca_variance: float = 0.90
    max_noise_ratio: float = 0.60
    max_clusters: int = 12
    risk_multiplier: float = 1.5


from evaluation import ClusteringResult

DBSCANResult = ClusteringResult


def load_stroke_data(path: str | Path) -> pd.DataFrame:
    return clean_data(pd.read_csv(path))


def _metrics(features: np.ndarray, labels: np.ndarray) -> tuple[int, float, float, float]:
    non_noise = labels != -1
    clusters = len(set(labels[non_noise]))
    noise = float(np.mean(~non_noise))
    if clusters < 2 or non_noise.sum() <= clusters:
        return clusters, noise, np.nan, np.nan
    return (
        clusters,
        noise,
        float(silhouette_score(features[non_noise], labels[non_noise])),
        float(davies_bouldin_score(features[non_noise], labels[non_noise])),
    )


def k_distance_diagnostics(features: np.ndarray, min_samples: int) -> tuple[np.ndarray, float]:
    """Use a robust 90th-percentile start point, not the outlier-tail jump."""
    if len(features) <= min_samples:
        raise ValueError("min_samples must be smaller than the number of patients.")
    distances, _ = NearestNeighbors(n_neighbors=min_samples).fit(features).kneighbors(features)
    values = np.sort(distances[:, -1])
    return values, float(np.quantile(values, 0.90))


def evaluate_eps_candidates(features: np.ndarray, min_samples: int, suggested_eps: float, config: DBSCANConfig) -> pd.DataFrame:
    rows = []
    for eps in np.unique(np.round(suggested_eps * np.linspace(0.55, 1.45, 19), 4)):
        labels = DBSCAN(eps=float(eps), min_samples=min_samples).fit_predict(features)
        clusters, noise, silhouette, db_index = _metrics(features, labels)
        rows.append({
            "eps": float(eps), "clusters": clusters, "noise_ratio": noise,
            "silhouette": silhouette, "davies_bouldin": db_index,
            "valid": 2 <= clusters <= config.max_clusters and noise <= config.max_noise_ratio and np.isfinite(silhouette),
        })
    return pd.DataFrame(rows)


def _select_eps(search: pd.DataFrame, suggested_eps: float) -> float:
    candidates = search[search["valid"]].copy()
    if candidates.empty:
        candidates = search.dropna(subset=["silhouette"]).copy()
    if candidates.empty:
        return suggested_eps
    candidates["distance_from_suggestion"] = (candidates["eps"] - suggested_eps).abs()
    return float(candidates.sort_values(
        ["silhouette", "noise_ratio", "distance_from_suggestion"],
        ascending=[False, True, True],
    ).iloc[0]["eps"])


def run_dbscan(data: pd.DataFrame, config: DBSCANConfig = DBSCANConfig()) -> DBSCANResult:
    """Fit DBSCAN; stroke is held out until clusters are interpreted."""
    if not 0 < config.pca_variance <= 1:
        raise ValueError("pca_variance must be between 0 and 1.")
    
    cleaned = clean_data(data)
    processed, feature_names = preprocess_data(cleaned)
    features, projection_2d, pca_full_variance_ratio, pca_selected_loadings = apply_pca(
        processed, feature_names, config.pca_variance
    )
    
    min_samples = config.min_samples or max(4, 2 * features.shape[1])
    k_distances, suggested_eps = k_distance_diagnostics(features, min_samples)
    search = evaluate_eps_candidates(features, min_samples, suggested_eps, config)
    selected_eps = float(config.eps) if config.eps is not None else _select_eps(search, suggested_eps)
    labels = DBSCAN(eps=selected_eps, min_samples=min_samples).fit_predict(features)
    clusters, noise, silhouette, db_index = _metrics(features, labels)
    
    result_data = cleaned.copy()
    result_data["cluster"] = labels
    cluster_summary = calculate_cluster_summary(result_data, config.risk_multiplier)
    
    return DBSCANResult(
        data=result_data,
        features=features,
        projection_2d=projection_2d,
        labels=labels,
        parameter_results=search,
        k_distances=k_distances,
        suggested_eps=suggested_eps,
        selected_eps=selected_eps,
        min_samples=min_samples,
        n_components=features.shape[1],
        n_clusters=clusters,
        noise_ratio=noise,
        silhouette=silhouette,
        davies_bouldin=db_index,
        cluster_summary=cluster_summary,
        pca_full_variance_ratio=pca_full_variance_ratio,
        pca_selected_loadings=pca_selected_loadings,
        preprocessed_feature_names=feature_names,
        preprocessed_data=processed
    )


def main() -> None:
    result = run_dbscan(load_stroke_data("brain_stroke.csv"))
    print(f"Selected eps: {result.selected_eps:.4f} (suggested: {result.suggested_eps:.4f})")
    print(f"min_samples: {result.min_samples}; PCA components: {result.n_components}")
    print(f"Clusters: {result.n_clusters}; noise: {result.noise_ratio:.1%}")
    print(f"Silhouette: {result.silhouette:.3f}; Davies-Bouldin: {result.davies_bouldin:.3f}")


if __name__ == "__main__":
    main()

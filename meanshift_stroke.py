"""MeanShift clustering module for Stroke Patient Clustering Explorer.

MeanShift is a non-parametric, centroid-based algorithm that automatically
discovers the number of clusters by locating high-density regions in the
feature space. No cluster count needs to be specified upfront - the bandwidth
(kernel radius) controls granularity instead.

Pipeline
--------
1. Preprocess & scale clinical features (via data_processing.py)
2. Apply PCA for dimensionality reduction
3. Estimate optimal bandwidth with ``estimate_bandwidth`` (quantile sweep)
4. Fit MeanShift and assign cluster labels
5. Compute Silhouette & Davies-Bouldin quality metrics
6. Build cluster summary with stroke-risk profiling
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import davies_bouldin_score, silhouette_score

from data_processing import (
    clean_data,
    preprocess_data,
    apply_pca,
    calculate_cluster_summary,
)
from evaluation import ClusteringResult

RANDOM_STATE = 42

MeanShiftResult = ClusteringResult


@dataclass(frozen=True)
class MeanShiftConfig:
    """Tunable knobs exposed to the Streamlit sidebar."""

    bandwidth: Optional[float] = None
    quantile: float = 0.25
    pca_variance: float = 0.90
    risk_multiplier: float = 1.5
    min_bin_freq: int = 5
    cluster_all: bool = True


def _metrics(
    features: np.ndarray, labels: np.ndarray
) -> tuple[int, float, float, float]:
    """Compute cluster count, noise ratio and quality scores."""
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


def _auto_bandwidth(features: np.ndarray, quantile: float) -> float:
    """Return sklearn's bandwidth estimate, clamped to a sensible range."""
    bw = estimate_bandwidth(features, quantile=quantile, random_state=RANDOM_STATE)
    return float(max(bw, 0.1))


def bandwidth_sweep(
    features: np.ndarray,
    config: MeanShiftConfig,
) -> pd.DataFrame:
    """Evaluate a range of bandwidth values around the automatic estimate."""
    base_bw = _auto_bandwidth(features, config.quantile)
    candidates = np.unique(
        np.round(base_bw * np.linspace(0.50, 1.80, 14), 4)
    )
    rows = []
    for bw in candidates:
        ms = MeanShift(
            bandwidth=float(bw),
            min_bin_freq=config.min_bin_freq,
            cluster_all=config.cluster_all,
            bin_seeding=True,
        )
        labels = ms.fit_predict(features)
        clusters, noise, silhouette, db_index = _metrics(features, labels)
        rows.append(
            {
                "bandwidth": float(bw),
                "clusters": clusters,
                "noise_ratio": noise,
                "silhouette": silhouette,
                "davies_bouldin": db_index,
                "valid": clusters >= 2 and np.isfinite(silhouette),
            }
        )
    return pd.DataFrame(rows)


def _select_bandwidth(sweep: pd.DataFrame, suggested_bw: float) -> float:
    """Pick the best bandwidth from the sweep table (highest silhouette)."""
    candidates = sweep[sweep["valid"]].copy()
    if candidates.empty:
        candidates = sweep.dropna(subset=["silhouette"]).copy()
    if candidates.empty:
        return suggested_bw
    candidates["dist"] = (candidates["bandwidth"] - suggested_bw).abs()
    return float(
        candidates.sort_values(
            ["silhouette", "noise_ratio", "dist"],
            ascending=[False, True, True],
        ).iloc[0]["bandwidth"]
    )


def run_meanshift(
    data: pd.DataFrame,
    config: MeanShiftConfig = MeanShiftConfig(),
) -> MeanShiftResult:
    """Full MeanShift pipeline: clean -> preprocess -> PCA -> cluster -> evaluate."""
    if not 0 < config.pca_variance <= 1:
        raise ValueError("pca_variance must be between 0 and 1.")

    cleaned = clean_data(data)
    processed, feature_names = preprocess_data(cleaned)
    features, projection_2d, pca_full_variance_ratio, pca_selected_loadings = (
        apply_pca(processed, feature_names, config.pca_variance)
    )

    suggested_bw = _auto_bandwidth(features, config.quantile)
    sweep = bandwidth_sweep(features, config)
    selected_bw = (
        float(config.bandwidth)
        if config.bandwidth is not None
        else _select_bandwidth(sweep, suggested_bw)
    )

    ms = MeanShift(
        bandwidth=selected_bw,
        min_bin_freq=config.min_bin_freq,
        cluster_all=config.cluster_all,
        bin_seeding=True,
    )
    labels = ms.fit_predict(features)

    clusters, noise, silhouette, db_index = _metrics(features, labels)

    result_data = cleaned.copy()
    result_data["cluster"] = labels
    cluster_summary = calculate_cluster_summary(result_data, config.risk_multiplier)

    return MeanShiftResult(
        data=result_data,
        features=features,
        projection_2d=projection_2d,
        labels=labels,
        n_components=features.shape[1],
        n_clusters=clusters,
        noise_ratio=noise,
        silhouette=silhouette,
        davies_bouldin=db_index,
        cluster_summary=cluster_summary,
        pca_full_variance_ratio=pca_full_variance_ratio,
        pca_selected_loadings=pca_selected_loadings,
        preprocessed_feature_names=feature_names,
        preprocessed_data=processed,
        suggested_eps=suggested_bw,
        selected_eps=selected_bw,
        parameter_results=sweep,
    )


def main() -> None:
    path = Path(__file__).with_name("brain_stroke.csv")
    result = run_meanshift(pd.read_csv(path))
    print(f"Suggested bandwidth : {result.suggested_eps:.4f}")
    print(f"Selected bandwidth  : {result.selected_eps:.4f}")
    print(f"PCA components      : {result.n_components}")
    print(f"Clusters found      : {result.n_clusters}")
    print(f"Noise ratio         : {result.noise_ratio:.1%}")
    print(f"Silhouette score    : {result.silhouette:.3f}" if not np.isnan(result.silhouette) else "Silhouette: N/A")
    print(f"Davies-Bouldin      : {result.davies_bouldin:.3f}" if not np.isnan(result.davies_bouldin) else "Davies-Bouldin: N/A")


if __name__ == "__main__":
    main()
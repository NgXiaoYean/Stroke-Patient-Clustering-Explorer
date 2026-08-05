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

def generate_algorithm_comparison(results_dict: dict[str, ClusteringResult]) -> pd.DataFrame:
    """Compiles internal metrics and stroke risk capture across all algorithms."""
    comparison_rows = []
    
    for model_name, res in results_dict.items():
        # Exclude noise points (-1) when calculating clean cluster metrics
        clean_df = res.data[res.data["cluster"] != -1]
        stroke_rates = clean_df.groupby("cluster")["stroke"].mean()
        max_stroke_rate = stroke_rates.max() if not stroke_rates.empty else 0.0
        
        comparison_rows.append({
            "Algorithm": model_name,
            "Clusters Found": res.n_clusters,
            "Noise Ratio": f"{res.noise_ratio:.2%}",
            "Silhouette Score": round(res.silhouette, 4) if np.isfinite(res.silhouette) else "N/A",
            "Davies-Bouldin Index": round(res.davies_bouldin, 4) if np.isfinite(res.davies_bouldin) else "N/A",
            "Max Cluster Stroke Rate": f"{max_stroke_rate:.2%}",
            "Baseline Stroke Rate": f"{res.data['stroke'].mean():.2%}"
        })
    return pd.DataFrame(comparison_rows)


def calculate_feature_contributions(res: ClusteringResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculates attribute contribution via PCA loadings and cluster mean percentage deviations."""
    # Method A: Overall feature weight through mean absolute PCA loadings
    pca_weights = res.pca_selected_loadings.abs().mean(axis=1).sort_values(ascending=False)
    pca_importance = pd.DataFrame({
        "Attribute": pca_weights.index,
        "Mean Absolute PCA Weight": pca_weights.values.round(4)
    })
    
    # Method B: Relative deviation of cluster means from overall dataset population mean
    numeric_cols = ["age", "avg_glucose_level", "bmi"]
    global_means = res.data[numeric_cols].mean()
    cluster_means = res.data[res.data["cluster"] != -1].groupby("cluster")[numeric_cols].mean()
    percentage_deviation = (((cluster_means - global_means) / global_means) * 100).round(2)
    
    return pca_importance, percentage_deviation
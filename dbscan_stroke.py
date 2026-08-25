from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from data_processing import (
    build_preprocessor,
    clean_data,
    preprocess_data,
    apply_pca,
    calculate_cluster_summary,
    NUMERIC_COLUMNS,
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
)

RANDOM_STATE = 42

RAW_INPUT_COLUMNS = NUMERIC_COLUMNS + BINARY_COLUMNS + CATEGORICAL_COLUMNS


@dataclass(frozen=True)
class DBSCANConfig:
    eps: Optional[float] = 0.760
    min_samples: Optional[int] = 30
    pca_variance: float = 0.90
    max_noise_ratio: float = 0.60
    max_clusters: int = 12
    risk_multiplier: float = 1.5


from evaluation import ClusteringResult

DBSCANResult = ClusteringResult


@dataclass
class PredictionArtifacts:
    preprocessor: ColumnTransformer
    pca_selected: PCA
    pca_scaler: StandardScaler
    nearest_neighbor_index: NearestNeighbors
    training_labels: np.ndarray
    cluster_summary: pd.DataFrame


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


def run_dbscan_with_artifacts(
    data: pd.DataFrame,
    config: DBSCANConfig = DBSCANConfig(),
) -> tuple[DBSCANResult, PredictionArtifacts]:
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

    artifact_preprocessor = build_preprocessor().fit(cleaned[RAW_INPUT_COLUMNS])
    artifact_processed = artifact_preprocessor.transform(cleaned[RAW_INPUT_COLUMNS])
    if hasattr(artifact_processed, "toarray"):
        artifact_processed = artifact_processed.toarray()

    artifact_pca = PCA(
        n_components=config.pca_variance, svd_solver="full", random_state=RANDOM_STATE
    ).fit(artifact_processed)
    artifact_pca_features = artifact_pca.transform(artifact_processed)

    artifact_scaler = StandardScaler().fit(artifact_pca_features)

    nn_index = NearestNeighbors(n_neighbors=1).fit(features)

    result = DBSCANResult(
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
        preprocessed_data=processed,
    )

    artifacts = PredictionArtifacts(
        preprocessor=artifact_preprocessor,
        pca_selected=artifact_pca,
        pca_scaler=artifact_scaler,
        nearest_neighbor_index=nn_index,
        training_labels=labels,
        cluster_summary=cluster_summary,
    )

    return result, artifacts


def run_dbscan(data: pd.DataFrame, config: DBSCANConfig = DBSCANConfig()) -> DBSCANResult:
    result, _artifacts = run_dbscan_with_artifacts(data, config)
    return result


def predict_new_patient(
    raw_patient: dict,
    artifacts: PredictionArtifacts,
) -> dict:
    missing = sorted(set(RAW_INPUT_COLUMNS) - set(raw_patient.keys()))
    if missing:
        raise ValueError(f"Missing required input columns: {', '.join(missing)}")

    patient_df = pd.DataFrame([{col: raw_patient[col] for col in RAW_INPUT_COLUMNS}])

    processed = artifacts.preprocessor.transform(patient_df)
    if hasattr(processed, "toarray"):
        processed = processed.toarray()
    pca_features = artifacts.pca_selected.transform(processed)
    scaled_features = artifacts.pca_scaler.transform(pca_features)

    _, neighbor_indices = artifacts.nearest_neighbor_index.kneighbors(
        scaled_features, n_neighbors=1
    )
    neighbor_idx = int(neighbor_indices[0, 0])
    predicted_cluster = int(artifacts.training_labels[neighbor_idx])

    if predicted_cluster == -1:
        return {
            "predicted_cluster": -1,
            "matched_profile": "No confident match (falls in a sparse/noise region)",
            "cluster_patients": 0,
            "cluster_stroke_rate_pct": 0.0,
            "cluster_age_mean": 0.0,
            "cluster_glucose_mean": 0.0,
            "cluster_bmi_mean": 0.0,
            "cluster_elevated_risk": False,
        }

    summary = artifacts.cluster_summary
    row = summary[summary["cluster"] == predicted_cluster]

    if row.empty:
        return {
            "predicted_cluster": predicted_cluster,
            "matched_profile": "Unknown",
            "cluster_patients": 0,
            "cluster_stroke_rate_pct": 0.0,
            "cluster_age_mean": 0.0,
            "cluster_glucose_mean": 0.0,
            "cluster_bmi_mean": 0.0,
            "cluster_elevated_risk": False,
        }

    row = row.iloc[0]
    elevated = bool(row["elevated_risk"])
    return {
        "predicted_cluster": predicted_cluster,
        "matched_profile": "Elevated Risk" if elevated else "Typical Risk",
        "cluster_patients": int(row["patients"]),
        "cluster_stroke_rate_pct": float(row["stroke_rate_pct"]),
        "cluster_age_mean": float(row["age_mean"]),
        "cluster_glucose_mean": float(row["glucose_mean"]),
        "cluster_bmi_mean": float(row["bmi_mean"]),
        "cluster_elevated_risk": elevated,
    }


def main() -> None:
    result = run_dbscan(load_stroke_data("brain_stroke.csv"))
    print(f"Selected eps: {result.selected_eps:.4f} (suggested: {result.suggested_eps:.4f})")
    print(f"min_samples: {result.min_samples}; PCA components: {result.n_components}")
    print(f"Clusters: {result.n_clusters}; noise: {result.noise_ratio:.1%}")
    print(f"Silhouette: {result.silhouette:.3f}; Davies-Bouldin: {result.davies_bouldin:.3f}")


if __name__ == "__main__":
    main()

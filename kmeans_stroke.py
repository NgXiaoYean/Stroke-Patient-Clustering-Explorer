"""K-Means clustering module for Stroke Patient Clustering Explorer.

The stroke outcome is excluded from clustering and is used only afterwards
for clinical interpretation of the discovered patient groups.

Pipeline
--------
1. Preprocess & scale clinical features (via data_processing.py)
2. Apply PCA for dimensionality reduction (with a downstream StandardScaler)
3. Search K=min_k..max_k and select the K with the highest Silhouette score
4. Fit KMeans with the selected K
5. Compute Silhouette & Davies-Bouldin quality metrics
6. Build cluster summary with stroke-risk profiling
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
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
from evaluation import ClusteringResult

RANDOM_STATE = 42
KMeansResult = ClusteringResult

# Columns accepted by predict_new_patient() — must match meanshift_stroke.py exactly.
RAW_INPUT_COLUMNS = NUMERIC_COLUMNS + BINARY_COLUMNS + CATEGORICAL_COLUMNS


@dataclass
class PredictionArtifacts:
    preprocessor: ColumnTransformer
    pca_selected: PCA
    pca_scaler: StandardScaler
    model: KMeans
    cluster_summary: pd.DataFrame


@dataclass(frozen=True)
class KMeansConfig:
    n_clusters: Optional[int] = None
    min_k: int = 2
    max_k: int = 15
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


def run_kmeans_with_artifacts(
    data: pd.DataFrame,
    config: KMeansConfig = KMeansConfig(),
) -> tuple[KMeansResult, PredictionArtifacts]:
    """Full K-Means pipeline that also returns fitted prediction artifacts.

    The artifacts contain fitted copies of the preprocessor, PCA, scaler,
    and the same KMeans model used for clustering, so that new patients can
    be scored via ``predict_new_patient()`` without re-fitting anything.
    """
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

    # ── Fit dedicated copies for prediction artifacts ─────────────────────────
    # These are fitted on cleaned[RAW_INPUT_COLUMNS] only — never on the stroke label.
    # Transform order must match apply_pca():  preprocessor → pca → StandardScaler.
    artifact_preprocessor = build_preprocessor().fit(cleaned[RAW_INPUT_COLUMNS])
    artifact_processed = artifact_preprocessor.transform(cleaned[RAW_INPUT_COLUMNS])
    if hasattr(artifact_processed, "toarray"):
        artifact_processed = artifact_processed.toarray()

    artifact_pca = PCA(
        n_components=config.pca_variance, svd_solver="full", random_state=RANDOM_STATE
    ).fit(artifact_processed)
    artifact_pca_features = artifact_pca.transform(artifact_processed)

    artifact_scaler = StandardScaler().fit(artifact_pca_features)

    result = KMeansResult(
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

    artifacts = PredictionArtifacts(
        preprocessor=artifact_preprocessor,
        pca_selected=artifact_pca,
        pca_scaler=artifact_scaler,
        model=model,
        cluster_summary=cluster_summary,
    )

    return result, artifacts


def run_kmeans(data: pd.DataFrame, config: KMeansConfig = KMeansConfig()) -> KMeansResult:
    """Fit K-Means on PCA-transformed patient features."""
    result, _artifacts = run_kmeans_with_artifacts(data, config)
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

    predicted_cluster = int(artifacts.model.predict(scaled_features)[0])

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
    path = Path(__file__).with_name("brain_stroke.csv")
    result = run_kmeans(pd.read_csv(path))
    print(f"Selected K         : {result.selected_k}")
    print(f"PCA components     : {result.n_components}")
    print(f"Silhouette score   : {result.silhouette:.4f}")
    print(f"Davies-Bouldin     : {result.davies_bouldin:.4f}")
    print(f"Inertia            : {result.inertia:.4f}")


if __name__ == "__main__":
    main()

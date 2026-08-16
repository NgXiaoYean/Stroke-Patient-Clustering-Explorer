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

MeanShiftResult = ClusteringResult

RAW_INPUT_COLUMNS = NUMERIC_COLUMNS + BINARY_COLUMNS + CATEGORICAL_COLUMNS


@dataclass
class PredictionArtifacts:
    preprocessor: ColumnTransformer
    pca_selected: PCA
    pca_scaler: StandardScaler
    model: MeanShift
    cluster_summary: pd.DataFrame


@dataclass(frozen=True)
class MeanShiftConfig:
    bandwidth: Optional[float] = None
    quantile: float = 0.25
    pca_variance: float = 0.90
    risk_multiplier: float = 1.5
    min_bin_freq: int = 5
    cluster_all: bool = True


def _metrics(
    features: np.ndarray, labels: np.ndarray
) -> tuple[int, float, float, float]:
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
        np.round(base_bw * np.linspace(0.30, 1.80, 16), 4)
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


def run_meanshift_with_artifacts(
    data: pd.DataFrame,
    config: MeanShiftConfig = MeanShiftConfig(),
) -> tuple[MeanShiftResult, PredictionArtifacts]:
    """Full MeanShift pipeline that also returns fitted prediction artifacts.

    The artifacts contain fitted copies of the preprocessor, PCA, scaler,
    and MeanShift model so that new patients can be scored via
    ``predict_new_patient()`` without re-fitting anything.
    """
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

    # ── Fit dedicated copies for prediction artifacts ────────────────────
    artifact_preprocessor = build_preprocessor().fit(cleaned[RAW_INPUT_COLUMNS])
    artifact_processed = artifact_preprocessor.transform(cleaned[RAW_INPUT_COLUMNS])
    if hasattr(artifact_processed, "toarray"):
        artifact_processed = artifact_processed.toarray()

    artifact_pca = PCA(
        n_components=config.pca_variance, svd_solver="full", random_state=RANDOM_STATE
    ).fit(artifact_processed)
    artifact_pca_features = artifact_pca.transform(artifact_processed)

    artifact_scaler = StandardScaler().fit(artifact_pca_features)

    result = MeanShiftResult(
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

    artifacts = PredictionArtifacts(
        preprocessor=artifact_preprocessor,
        pca_selected=artifact_pca,
        pca_scaler=artifact_scaler,
        model=ms,
        cluster_summary=cluster_summary,
    )

    return result, artifacts


def run_meanshift(
    data: pd.DataFrame,
    config: MeanShiftConfig = MeanShiftConfig(),
) -> MeanShiftResult:
    """Full MeanShift pipeline: clean -> preprocess -> PCA -> cluster -> evaluate."""
    result, _artifacts = run_meanshift_with_artifacts(data, config)
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


from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd

from data_processing import load_dataset
from dbscan_stroke import DBSCANConfig, run_dbscan_with_artifacts
from meanshift_stroke import MeanShiftConfig, run_meanshift_with_artifacts

try:
    from kmeans_stroke import KMeansConfig, run_kmeans_with_artifacts
    KMEANS_AVAILABLE = True
except ImportError:
    KMEANS_AVAILABLE = False


DATA_PATH = Path(__file__).with_name("brain_stroke.csv")
OUTPUT_DIR = Path(__file__).with_name("04_Trained_Model")


def main() -> None:
    if not DATA_PATH.exists():
        print(f"ERROR: '{DATA_PATH.name}' not found next to save_models.py. "
              "Place the dataset there and re-run.")
        sys.exit(1)

    print(f"Loading dataset from {DATA_PATH} ...")
    data: pd.DataFrame = load_dataset(str(DATA_PATH))
    print(f"Loaded {len(data):,} rows.")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── K-Means ────────────────────────────────────────────────────────
    if KMEANS_AVAILABLE:
        print("Training K-Means (n_clusters=auto, pca_variance=0.90, "
              "risk_multiplier=1.5, max_k=15) ...")
        km_result, km_artifacts = run_kmeans_with_artifacts(
            data,
            KMeansConfig(n_clusters=None, pca_variance=0.90, risk_multiplier=1.5, max_k=15),
        )
        out_path = OUTPUT_DIR / "kmeans_artifacts.joblib"
        joblib.dump((km_result, km_artifacts), out_path)
        print(f"  Saved -> {out_path}")
    else:
        print("K-Means module not available (import failed) — skipped.")

    # ── DBSCAN ─────────────────────────────────────────────────────────
    print("Training DBSCAN (eps=auto, min_samples=30, pca_variance=0.90, "
          "risk_multiplier=1.5) ...")
    db_result, db_artifacts = run_dbscan_with_artifacts(
        data,
        DBSCANConfig(eps=None, min_samples=30, pca_variance=0.90, risk_multiplier=1.5),
    )
    out_path = OUTPUT_DIR / "dbscan_artifacts.joblib"
    joblib.dump((db_result, db_artifacts), out_path)
    print(f"  Saved -> {out_path}")

    # ── MeanShift ──────────────────────────────────────────────────────
    print("Training MeanShift (bandwidth=auto, quantile=0.25, pca_variance=0.90, "
          "risk_multiplier=1.5, min_bin_freq=5) ...")
    ms_result, ms_artifacts = run_meanshift_with_artifacts(
        data,
        MeanShiftConfig(bandwidth=None, quantile=0.25, pca_variance=0.90,
                         risk_multiplier=1.5, min_bin_freq=5),
    )
    out_path = OUTPUT_DIR / "meanshift_artifacts.joblib"
    joblib.dump((ms_result, ms_artifacts), out_path)
    print(f"  Saved -> {out_path}")

    print("\nDone. Trained model files are in:", OUTPUT_DIR.resolve())
    print("Copy the '04_Trained_Model' folder next to app.py (or into your "
          "submission's 04_Trained_Model/ folder) so app.py can find it.")


if __name__ == "__main__":
    main()
# Stroke Patient Clustering Explorer

BMCS2074 Artificial Intelligence Assignment (Unsupervised Machine Learning). Applies K-Means, DBSCAN, and MeanShift to the Brain Stroke Dataset to find patient groups with similar clinical/lifestyle profiles, then checks how stroke cases are distributed across the groups found.

## Files Introduction
- `app.py` — Streamlit dashboard, entry point for the app.
- `data_processing.py` — cleaning, preprocessing (RobustScaler/StandardScaler + one-hot encoding), PCA, chi-square/likelihood-ratio tests, EDA helpers.
- `kmeans_stroke.py` — K-Means pipeline, K search (K=2–15), prediction for new patients.
- `dbscan_stroke.py` — DBSCAN pipeline, k-distance diagnostics, eps grid search.
- `meanshift_stroke.py` — MeanShift pipeline, bandwidth estimation + sweep.
- `evaluation.py` — shared `ClusteringResult` dataclass and cross-algorithm metrics.
- `visualizations.py` — all Plotly chart functions used by the dashboard.
- `brain_stroke.csv` — dataset (Kaggle, Jillani SofTech, 2022).

## Setup & Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens the dashboard in your browser. Use the sidebar to pick an algorithm (K-Means / DBSCAN / MeanShift), adjust parameters, and view cluster plots, stroke-rate breakdowns, and PCA diagnostics. There's also a "predict new patient" form that assigns a hypothetical patient to the nearest existing cluster.

## Notes
- The stroke label is never used during clustering — only afterwards, to profile the clusters that were found.
- K-Means K is selected by Silhouette Score (inertia/elbow shown as a secondary check, not the selection criterion).
- DBSCAN's eps was manually chosen from the grid search (`eps=0.76`, `min_samples=30`) to balance noise ratio against Silhouette Score — see report Section 3.4.2.2 for the reasoning.
- Random state fixed at 42 throughout for reproducibility.

# Stroke Patient Clustering Explorer (Streamlit)

Interactive app for your assignment goal: *discover groups of patients with
similar lifestyle/clinical characteristics, then analyze how stroke cases are
distributed among those groups.*

## Run it locally

1. Put `app.py` and `brain_stroke.csv` in the same folder.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Launch:
   ```
   streamlit run app.py
   ```
4. It opens in your browser at http://localhost:8501

## What it does

- **Sidebar**: pick the algorithm (DBSCAN / K-Means / MeanShift), tune its
  parameters live (eps, min_samples, k, bandwidth), adjust how much PCA
  variance to keep, and set the "elevated risk" threshold. Everything below
  recalculates instantly (results are cached so it stays fast).
- **📊 Data Overview**: dataset preview, stroke class balance, feature
  distributions, PCA variance curve.
- **🧩 Clusters**: 2D PCA scatter of the clusters (color by cluster or by
  stroke status), cluster size chart, and — for DBSCAN — the k-distance
  elbow graph so you can see exactly where your `eps` sits.
- **❤️ Stroke Risk by Cluster**: stroke rate per cluster vs. the overall
  rate, auto-flags "elevated-risk" clusters, plus the full crosstab.
- **🧬 Cluster Profiles**: mean age/glucose/BMI/etc. per cluster, a radar
  chart comparing clusters, and each cluster's most common work type /
  smoking status.
- **🧑‍⚕️ Try a Patient**: enter a hypothetical patient's details and the
  app runs them through the same preprocessing pipeline, finds their nearest
  cluster, and shows whether that cluster is elevated-risk — a nice live
  demo for your presentation.

Preprocessing mirrors your notebook: `RobustScaler` for outlier-prone
numeric features, `StandardScaler` for the binary flags, `OneHotEncoder`
for categoricals, then PCA before clustering.

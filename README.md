# Stroke Patient Clustering Explorer

Interactive Streamlit application for discovering patient profiles with DBSCAN,
then comparing their stroke rates. The `stroke` column is **not** used to build
clusters: it remains an outcome for interpreting the profiles afterwards.

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The app opens with DBSCAN ready to use. K-Means and MeanShift already have
navigation placeholders, so teammates can connect their modules without
rebuilding the UI.

## DBSCAN settings

| Setting | What it changes | Typical effect |
| --- | --- | --- |
| `eps` | Neighbourhood radius | Lower EPS makes tight/smaller groups and more noise; higher EPS merges nearby groups. Too high can put most patients in one cluster. |
| `min_samples` | Neighbours required to be a dense core | Higher values make DBSCAN stricter, usually producing more noise and fewer/tighter clusters. |
| PCA variance | Information kept before distance clustering | Higher values retain more detail but make distances less dense; use a stable value such as 0.85–0.95. |
| Elevated-risk multiplier | Interpretation only | Flags a cluster when its stroke rate exceeds the overall rate by this multiplier. It does not alter clustering. |

Use automatic EPS first. It uses the 90th percentile of the k-distance curve
and scores nearby values, avoiding the old approach of using the final largest
jump, which is often caused by isolated outliers. Compare candidates using:

- higher silhouette score (better separation),
- lower Davies–Bouldin index (better separation),
- a noise percentage that is understandable for the project, and
- clusters large enough to describe meaningfully.

There is no universally “best” high or low number of clusters. More than two
clusters is completely valid even though `stroke` has only `0` and `1`: clusters
represent different patient profiles, while stroke is a separate outcome. The
`-1` label means a patient is noise/unassigned, not a third stroke category.

## Report files

Run the DBSCAN module directly to create reusable report artefacts:

```powershell
python dbscan_stroke.py
```

It creates `reports/dbscan_report.png`, the EPS search table, a cluster summary,
and patient-level cluster labels.

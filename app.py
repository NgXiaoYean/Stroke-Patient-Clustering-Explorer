"""
Stroke Patient Clustering Explorer
-----------------------------------
Interactive Streamlit app that lets the user discover groups of patients with
similar lifestyle and clinical characteristics (via DBSCAN / K-Means /
MeanShift) and explore how stroke cases are distributed among those groups.

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

RANDOM_STATE = 42
DATA_PATH = "brain_stroke.csv"

NUMERIC_COLS = ["age", "avg_glucose_level", "bmi"]
BINARY_COLS = ["hypertension", "heart_disease"]
CATEGORICAL_COLS = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
PROFILE_COLS = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "stroke"]

st.set_page_config(
    page_title="Stroke Patient Clustering Explorer",
    page_icon="🧠",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
PRIMARY = "#1B4B66"
ACCENT = "#D9534F"
BG_CARD = "#F5F8FA"

st.markdown(
    f"""
    <style>
    .metric-card {{
        background-color: {BG_CARD};
        border-radius: 10px;
        padding: 14px 18px;
        border-left: 5px solid {PRIMARY};
    }}
    .risk-card {{
        background-color: #FCEEEE;
        border-radius: 10px;
        padding: 14px 18px;
        border-left: 5px solid {ACCENT};
    }}
    h1, h2, h3 {{
        color: {PRIMARY};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Data loading & caching
# ----------------------------------------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df


@st.cache_resource
def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), NUMERIC_COLS),
            ("bin", StandardScaler(), BINARY_COLS),
            ("cat", OneHotEncoder(drop=None, handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )


@st.cache_data
def preprocess(df):
    stroke_label = df["stroke"].copy()
    X_raw = df.drop(columns=["stroke"])
    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X_raw)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    return X_processed, stroke_label


@st.cache_data
def run_pca(X_processed, variance_threshold):
    pca_full = PCA(random_state=RANDOM_STATE).fit(X_processed)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cum_var >= variance_threshold) + 1)
    n_components = max(n_components, 2)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_processed)
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca_2d.fit_transform(X_processed)
    return X_pca, X_2d, cum_var, n_components


@st.cache_data
def suggest_eps(X_pca, min_samples):
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X_pca)
    distances, _ = neighbors.kneighbors(X_pca)
    k_distances = np.sort(distances[:, -1])
    diffs = np.diff(k_distances)
    elbow_idx = np.argmax(diffs[len(diffs) // 2 :]) + len(diffs) // 2
    return float(k_distances[elbow_idx]), k_distances


@st.cache_data
def run_clustering(algo, X_pca, params):
    if algo == "DBSCAN":
        model = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
        labels = model.fit_predict(X_pca)
    elif algo == "K-Means":
        model = KMeans(n_clusters=params["n_clusters"], random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X_pca)
    elif algo == "MeanShift":
        model = MeanShift(bandwidth=params["bandwidth"], bin_seeding=True)
        labels = model.fit_predict(X_pca)
    else:
        raise ValueError(algo)
    return labels


def cluster_metrics(X_pca, labels):
    mask = labels != -1
    n_clusters = len(set(labels[mask])) if mask.any() else 0
    noise_ratio = float(np.mean(labels == -1))
    sil = dbi = np.nan
    if n_clusters >= 2 and mask.sum() > 1:
        sil = silhouette_score(X_pca[mask], labels[mask])
        dbi = davies_bouldin_score(X_pca[mask], labels[mask])
    return n_clusters, noise_ratio, sil, dbi


# ----------------------------------------------------------------------------
# Load & prep data
# ----------------------------------------------------------------------------
df_original = load_data(DATA_PATH)
X_processed, stroke_label = preprocess(df_original)

# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------
st.sidebar.title("🧠 Controls")
st.sidebar.caption("Tune the pipeline and watch every tab update live.")

variance_threshold = st.sidebar.slider(
    "PCA variance to retain", min_value=0.70, max_value=0.99, value=0.90, step=0.01,
    help="Number of principal components is chosen automatically to explain at least this much variance."
)
X_pca, X_2d, cum_var, n_components = run_pca(X_processed, variance_threshold)
st.sidebar.caption(f"→ Using **{n_components}** principal components.")

st.sidebar.markdown("---")
algo = st.sidebar.radio("Clustering algorithm", ["DBSCAN", "K-Means", "MeanShift"], index=0)

params = {}
if algo == "DBSCAN":
    default_min_samples = 2 * n_components
    min_samples = st.sidebar.slider("min_samples", 2, 40, default_min_samples)
    eps_suggestion, k_distances = suggest_eps(X_pca, min_samples)

    if "eps_value" not in st.session_state:
        st.session_state.eps_value = round(eps_suggestion, 2)

    col_a, col_b = st.sidebar.columns([2, 1])
    with col_b:
        st.write("")
        st.write("")
        if st.button("Auto"):
            st.session_state.eps_value = round(eps_suggestion, 2)
    with col_a:
        eps = st.slider(
            "eps (neighborhood radius)", 0.1, 5.0,
            value=st.session_state.eps_value, step=0.05, key="eps_slider"
        )
    st.session_state.eps_value = eps
    st.sidebar.caption(f"Elbow-suggested eps ≈ {eps_suggestion:.2f} (click **Auto** to apply)")
    params = {"eps": eps, "min_samples": min_samples}

elif algo == "K-Means":
    n_clusters = st.sidebar.slider("Number of clusters (k)", 2, 10, 4)
    params = {"n_clusters": n_clusters}

elif algo == "MeanShift":
    auto_bw = estimate_bandwidth(X_pca, quantile=0.2, random_state=RANDOM_STATE)
    bandwidth = st.sidebar.slider(
        "Bandwidth", max(0.1, float(auto_bw) * 0.3), float(auto_bw) * 2.5,
        value=float(auto_bw), step=0.05
    )
    st.sidebar.caption(f"Auto-estimated bandwidth ≈ {auto_bw:.2f}")
    params = {"bandwidth": bandwidth}

st.sidebar.markdown("---")
risk_multiplier = st.sidebar.slider(
    "Elevated-risk threshold (× overall stroke rate)", 1.1, 3.0, 1.5, 0.1
)

# ----------------------------------------------------------------------------
# Run clustering
# ----------------------------------------------------------------------------
labels = run_clustering(algo, X_pca, params)
df = df_original.copy()
df["cluster"] = labels
n_clusters, noise_ratio, sil, dbi = cluster_metrics(X_pca, labels)
overall_rate = df["stroke"].mean() * 100

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🧠 Stroke Patient Clustering Explorer")
st.markdown(
    "Discover groups of patients with similar **lifestyle and clinical characteristics**, "
    "then examine how **stroke cases** are distributed among those groups. "
    "Adjust the controls in the sidebar — every chart below updates instantly."
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Algorithm", algo)
m2.metric("Clusters found", n_clusters)
m3.metric("Noise points", f"{noise_ratio:.1%}" if algo == "DBSCAN" else "—")
m4.metric("Silhouette score", f"{sil:.3f}" if not np.isnan(sil) else "n/a")
m5.metric("Davies-Bouldin", f"{dbi:.3f}" if not np.isnan(dbi) else "n/a")

tabs = st.tabs(
    ["📊 Data Overview", "🧩 Clusters", "❤️ Stroke Risk by Cluster",
     "🧬 Cluster Profiles", "🧑‍⚕️ Try a Patient"]
)

# ----------------------------------------------------------------------------
# TAB 1 — Data overview / EDA
# ----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Dataset snapshot")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(df_original.head(10), use_container_width=True)
        st.caption(f"{df_original.shape[0]} patients · {df_original.shape[1]} columns")
    with c2:
        stroke_counts = df_original["stroke"].value_counts(normalize=True).rename(
            {0: "No Stroke", 1: "Stroke"}
        )
        fig = px.pie(
            values=stroke_counts.values, names=stroke_counts.index,
            title="Stroke class balance", color=stroke_counts.index,
            color_discrete_map={"No Stroke": PRIMARY, "Stroke": ACCENT}, hole=0.45
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Distribution of key clinical features")
    feat = st.selectbox("Feature", NUMERIC_COLS, key="eda_feat")
    fig = px.histogram(
        df_original, x=feat, color=df_original["stroke"].map({0: "No Stroke", 1: "Stroke"}),
        nbins=40, barmode="overlay", opacity=0.7,
        color_discrete_map={"No Stroke": PRIMARY, "Stroke": ACCENT},
        labels={"color": "Stroke status"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Categorical feature breakdown")
    cat_feat = st.selectbox("Category", CATEGORICAL_COLS, key="eda_cat")
    counts = df_original[cat_feat].value_counts().reset_index()
    counts.columns = [cat_feat, "count"]
    fig = px.bar(counts, x="count", y=cat_feat, orientation="h", color_discrete_sequence=[PRIMARY])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### PCA cumulative explained variance")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=cum_var, mode="lines+markers", name="Cumulative variance"))
    fig.add_hline(y=variance_threshold, line_dash="dash", line_color=ACCENT,
                  annotation_text=f"{variance_threshold:.0%} threshold")
    fig.add_vline(x=n_components - 1, line_dash="dot", line_color=PRIMARY,
                  annotation_text=f"{n_components} components")
    fig.update_layout(xaxis_title="Number of components", yaxis_title="Cumulative variance explained")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 2 — Cluster visualization
# ----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Patient clusters (PCA 2D projection)")
    plot_df = pd.DataFrame({
        "PC1": X_2d[:, 0], "PC2": X_2d[:, 1],
        "Cluster": np.where(labels == -1, "Noise", "Cluster " + labels.astype(str)),
        "Stroke": df_original["stroke"].map({0: "No Stroke", 1: "Stroke"}),
        "Age": df_original["age"],
    })
    color_view = st.radio("Color points by", ["Cluster", "Stroke"], horizontal=True)
    fig = px.scatter(
        plot_df, x="PC1", y="PC2", color=color_view,
        hover_data=["Age", "Stroke", "Cluster"],
        opacity=0.65,
        color_discrete_map={"No Stroke": PRIMARY, "Stroke": ACCENT} if color_view == "Stroke" else None,
    )
    fig.update_traces(marker=dict(size=6))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Cluster sizes")
    size_df = df["cluster"].value_counts().sort_index().reset_index()
    size_df.columns = ["Cluster", "Patients"]
    size_df["Cluster"] = size_df["Cluster"].map(lambda x: "Noise" if x == -1 else f"Cluster {x}")
    fig = px.bar(size_df, x="Cluster", y="Patients", color_discrete_sequence=[PRIMARY])
    st.plotly_chart(fig, use_container_width=True)

    if algo == "DBSCAN":
        st.markdown("#### K-distance graph (for eps tuning)")
        _, k_distances = suggest_eps(X_pca, params["min_samples"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=k_distances, mode="lines", name="k-distance"))
        fig.add_hline(y=params["eps"], line_dash="dash", line_color=ACCENT,
                      annotation_text=f"current eps = {params['eps']:.2f}")
        fig.update_layout(xaxis_title="Points sorted by distance",
                           yaxis_title=f"Distance to {params['min_samples']}-th neighbor")
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 3 — Stroke risk analysis
# ----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("How is stroke distributed across the discovered groups?")
    valid_df = df[df["cluster"] != -1] if -1 in df["cluster"].values else df
    cluster_rates = valid_df.groupby("cluster")["stroke"].mean().mul(100).round(2)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Cluster {c}" for c in cluster_rates.index], y=cluster_rates.values,
        marker_color=[ACCENT if r > overall_rate * risk_multiplier else PRIMARY for r in cluster_rates.values],
        text=[f"{r:.1f}%" for r in cluster_rates.values], textposition="outside",
    ))
    fig.add_hline(y=overall_rate, line_dash="dash", line_color="gray",
                  annotation_text=f"Overall rate ({overall_rate:.2f}%)")
    fig.update_layout(yaxis_title="Stroke rate (%)", xaxis_title="Cluster")
    st.plotly_chart(fig, use_container_width=True)

    elevated = cluster_rates[cluster_rates > overall_rate * risk_multiplier]
    if len(elevated):
        st.markdown(
            f"<div class='risk-card'>⚠️ <b>Elevated-risk cluster(s):</b> "
            f"{', '.join(f'Cluster {c} ({r:.1f}%)' for c, r in elevated.items())} "
            f"— stroke rate exceeds {risk_multiplier}× the overall rate of {overall_rate:.2f}%.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No cluster currently exceeds the elevated-risk threshold. Try lowering the multiplier or adjusting clustering parameters.")

    st.markdown("#### Cross-tabulation")
    cross = pd.crosstab(df["cluster"], df["stroke"], margins=True)
    cross.columns = ["No Stroke", "Stroke", "Total"] if cross.shape[1] == 3 else cross.columns
    cross["Stroke Rate (%)"] = (cross["Stroke"] / cross["Total"] * 100).round(2) if "Stroke" in cross.columns else np.nan
    st.dataframe(cross, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 4 — Cluster profiles
# ----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Average characteristics per cluster")
    cluster_profile = df.groupby("cluster")[PROFILE_COLS].mean().round(2)
    cluster_profile["size"] = df.groupby("cluster").size()
    st.dataframe(cluster_profile, use_container_width=True)

    st.markdown("#### Compare clusters visually")
    numeric_profile_cols = ["age", "avg_glucose_level", "bmi"]
    radar_source = df.groupby("cluster")[numeric_profile_cols].mean()
    radar_norm = (radar_source - radar_source.min()) / (radar_source.max() - radar_source.min() + 1e-9)

    fig = go.Figure()
    for c in radar_norm.index:
        label = "Noise" if c == -1 else f"Cluster {c}"
        fig.add_trace(go.Scatterpolar(
            r=radar_norm.loc[c].values.tolist() + [radar_norm.loc[c].values[0]],
            theta=numeric_profile_cols + [numeric_profile_cols[0]],
            fill="toself", name=label, opacity=0.6
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Most common lifestyle traits per cluster")
    c1, c2 = st.columns(2)
    for col, container in zip(["work_type", "smoking_status"], [c1, c2]):
        mode_table = df.groupby("cluster")[col].agg(lambda x: x.value_counts().idxmax())
        mode_table.index = ["Noise" if i == -1 else f"Cluster {i}" for i in mode_table.index]
        container.markdown(f"**Most common {col.replace('_', ' ')}**")
        container.dataframe(mode_table.rename("Most common value"), use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 5 — Try a patient (interactive "what-if")
# ----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("See which cluster a hypothetical patient belongs to")
    st.caption(
        "Enter a patient's lifestyle and clinical details. The app transforms them through the "
        "same pipeline used for clustering, then assigns them to the nearest cluster."
    )

    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            p_gender = st.selectbox("Gender", sorted(df_original["gender"].unique()))
            p_age = st.slider("Age", 0, 100, 55)
            p_married = st.selectbox("Ever married", sorted(df_original["ever_married"].unique()))
        with c2:
            p_work = st.selectbox("Work type", sorted(df_original["work_type"].unique()))
            p_residence = st.selectbox("Residence type", sorted(df_original["Residence_type"].unique()))
            p_smoking = st.selectbox("Smoking status", sorted(df_original["smoking_status"].unique()))
        with c3:
            p_hyper = st.selectbox("Hypertension", [0, 1])
            p_heart = st.selectbox("Heart disease", [0, 1])
            p_glucose = st.slider("Average glucose level", 50.0, 300.0, 100.0)
            p_bmi = st.slider("BMI", 10.0, 60.0, 26.0)
        submitted = st.form_submit_button("Find matching cluster")

    if submitted:
        patient_raw = pd.DataFrame([{
            "gender": p_gender, "age": p_age, "hypertension": p_hyper, "heart_disease": p_heart,
            "ever_married": p_married, "work_type": p_work, "Residence_type": p_residence,
            "avg_glucose_level": p_glucose, "bmi": p_bmi, "smoking_status": p_smoking,
        }])
        preprocessor = build_preprocessor()
        preprocessor.fit(df_original.drop(columns=["stroke"]))
        patient_processed = preprocessor.transform(patient_raw)
        if hasattr(patient_processed, "toarray"):
            patient_processed = patient_processed.toarray()

        pca_2d_model = PCA(n_components=2, random_state=RANDOM_STATE).fit(X_processed)
        pca_full_model = PCA(n_components=n_components, random_state=RANDOM_STATE).fit(X_processed)
        patient_pca = pca_full_model.transform(patient_processed)
        patient_2d = pca_2d_model.transform(patient_processed)

        # Assign to nearest labeled point's cluster (works for any algorithm, incl. DBSCAN)
        valid_mask = labels != -1
        if valid_mask.any():
            nn = NearestNeighbors(n_neighbors=1).fit(X_pca[valid_mask])
            _, idx = nn.kneighbors(patient_pca)
            assigned_cluster = labels[valid_mask][idx[0][0]]
        else:
            assigned_cluster = -1

        st.success(f"This patient profile matches **Cluster {assigned_cluster}**.")

        cluster_rate = valid_df[valid_df["cluster"] == assigned_cluster]["stroke"].mean() * 100
        risk_flag = cluster_rate > overall_rate * risk_multiplier
        risk_col = st.columns(3)
        risk_col[0].metric("Cluster stroke rate", f"{cluster_rate:.1f}%")
        risk_col[1].metric("Overall stroke rate", f"{overall_rate:.1f}%")
        risk_col[2].metric("Elevated risk?", "Yes ⚠️" if risk_flag else "No")

        plot_df2 = pd.DataFrame({
            "PC1": X_2d[:, 0], "PC2": X_2d[:, 1],
            "Cluster": np.where(labels == -1, "Noise", "Cluster " + labels.astype(str)),
        })
        fig = px.scatter(plot_df2, x="PC1", y="PC2", color="Cluster", opacity=0.4)
        fig.add_trace(go.Scatter(
            x=[patient_2d[0, 0]], y=[patient_2d[0, 1]], mode="markers",
            marker=dict(size=18, color="black", symbol="star"), name="This patient"
        ))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption(
    "Built for the unsupervised-learning assignment: DBSCAN / K-Means / MeanShift clustering "
    "on the Brain Stroke dataset (Jillani SofTech, 2022, Kaggle)."
)

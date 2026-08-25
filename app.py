from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from dbscan_stroke import (
    DBSCANConfig,
    run_dbscan_with_artifacts,
    predict_new_patient as predict_dbscan_patient,
)
from meanshift_stroke import (
    MeanShiftConfig,
    run_meanshift_with_artifacts,
    predict_new_patient as predict_meanshift_patient,
    RAW_INPUT_COLUMNS,
)
try:
    from kmeans_stroke import (
        KMeansConfig,
        run_kmeans_with_artifacts,
        predict_new_patient as predict_kmeans_patient,
    )
    KMEANS_AVAILABLE = True
except ImportError:
    KMEANS_AVAILABLE = False
from data_processing import (
    load_dataset,
    get_clinical_summary,
    get_categorical_analysis,
    get_correlation_matrix,
    get_preprocessing_previews,
    get_pca_scree_data,
    get_pca_loadings,
    get_data_quality_report,
    test_categorical_association,
    test_confounding_with_age
)
from visualizations import (
    plot_cluster_scatter,
    plot_k_distance,
    plot_eps_search,
    plot_stroke_by_cluster,
    plot_continuous_distribution,
    plot_categorical_prevalence,
    plot_correlation_heatmap,
    plot_pca_scree,
    plot_pca_loadings_bar,
    plot_target_distribution,
    plot_numerical_outliers,
    plot_numerical_distributions_grid,
    plot_categorical_distributions_grid,
    plot_scaling_comparison,
    plot_encoding_comparison,
    plot_pca_loadings_heatmap,
    plot_bandwidth_sweep,
    plot_meanshift_cluster_scatter,
    plot_cluster_profile_radar,
    plot_cluster_size_distribution,
)
from evaluation import generate_algorithm_comparison, calculate_feature_contributions

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("stroke_clustering_app")

DATA_PATH = Path(__file__).with_name("brain_stroke.csv")
MODEL_DIR = Path(__file__).with_name("04_Trained_Model")

st.set_page_config(page_title="Stroke Patient Clustering Explorer", page_icon="🧠", layout="wide")


# ────────────────────────────────────────────────────────────────────────────
# Pre-trained model loading (produced by save_models.py)
#
# save_models.py trains each algorithm once, offline, using the exact
# parameter values below, and saves the (result, artifacts) pair to
# 04_Trained_Model/<name>.joblib. If the sidebar is left on these same
# default values, we load that file instead of retraining. Any change to
# a slider/checkbox — or a missing/corrupt file — falls back to normal
# retraining automatically, so this is purely a speed optimisation and
# never blocks the app from working.
# ────────────────────────────────────────────────────────────────────────────

DBSCAN_DEFAULT_PARAMS = (None, 30, 0.90, 1.5)              # eps, min_samples, pca_variance, risk_multiplier
KMEANS_DEFAULT_PARAMS = (None, 0.90, 1.5, 15)              # n_clusters, pca_variance, risk_multiplier, max_k
MEANSHIFT_DEFAULT_PARAMS = (None, 0.25, 0.90, 1.5, 5)      # bandwidth, quantile, pca_variance, risk_multiplier, min_bin_freq


def _load_pretrained(name: str, current_params: tuple, default_params: tuple):
    """Return a cached (result, artifacts) pair from 04_Trained_Model/ if the
    caller is using the same default settings save_models.py was run with,
    otherwise return None so the caller retrains as usual."""
    if current_params != default_params:
        return None
    path = MODEL_DIR / f"{name}.joblib"
    if not path.exists():
        return None
    try:
        result, artifacts = joblib.load(path)
        logger.info("Loaded pre-trained %s model from %s", name, path)
        return result, artifacts
    except Exception as exc:  # noqa: BLE001 - any load failure just means "retrain"
        logger.warning("Could not load saved %s model (%s) — retraining instead.", name, exc)
        return None


# ────────────────────────────────────────────────────────────────────────────
# Cached data / model loading
# ────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Load and validate the source dataset.

    Raises a user-facing error in the Streamlit UI (rather than an
    uncaught traceback) if the file is missing or fails required
    column/type checks performed inside ``load_dataset``/``clean_data``.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{resolved}'. Make sure 'brain_stroke.csv' "
            "sits alongside app.py before launching the dashboard."
        )
    df = load_dataset(path)
    if df.empty:
        raise ValueError("Loaded dataset is empty — check the CSV file contents.")
    logger.info("Loaded dataset with %d rows, %d columns from %s", len(df), df.shape[1], resolved)
    return df


@st.cache_resource(show_spinner="Running DBSCAN analysis...")
def analyse_dbscan(data: pd.DataFrame, eps: float | None, min_samples: int, pca_variance: float, risk_multiplier: float):
    cached = _load_pretrained(
        "dbscan_artifacts", (eps, min_samples, pca_variance, risk_multiplier), DBSCAN_DEFAULT_PARAMS
    )
    if cached is not None:
        return cached
    return run_dbscan_with_artifacts(data, DBSCANConfig(
        eps=eps,
        min_samples=min_samples,
        pca_variance=pca_variance,
        risk_multiplier=risk_multiplier,
    ))


@st.cache_resource(show_spinner="Running MeanShift analysis...")
def analyse_meanshift(
    data: pd.DataFrame,
    bandwidth: float | None,
    quantile: float,
    pca_variance: float,
    risk_multiplier: float,
    min_bin_freq: int,
):
    cached = _load_pretrained(
        "meanshift_artifacts", (bandwidth, quantile, pca_variance, risk_multiplier, min_bin_freq),
        MEANSHIFT_DEFAULT_PARAMS,
    )
    if cached is not None:
        return cached
    return run_meanshift_with_artifacts(data, MeanShiftConfig(
        bandwidth=bandwidth,
        quantile=quantile,
        pca_variance=pca_variance,
        risk_multiplier=risk_multiplier,
        min_bin_freq=min_bin_freq,
    ))

if KMEANS_AVAILABLE:
    @st.cache_resource(show_spinner="Running K-Means analysis...")
    def analyse_kmeans(data: pd.DataFrame, n_clusters: int | None, pca_variance: float, risk_multiplier: float, max_k: int):
        cached = _load_pretrained(
            "kmeans_artifacts", (n_clusters, pca_variance, risk_multiplier, max_k), KMEANS_DEFAULT_PARAMS
        )
        if cached is not None:
            return cached
        return run_kmeans_with_artifacts(data, KMeansConfig(
            n_clusters=n_clusters,
            pca_variance=pca_variance,
            risk_multiplier=risk_multiplier,
            max_k=max_k,
        ))


# ────────────────────────────────────────────────────────────────────────────
# Standalone sidebar control blocks (kept for reuse / power users inside the
# "Technical Details" tab). Not called directly from main() any more — the
# friendlier inline sidebar blocks in main() are used instead — but nothing
# has been removed, only supplemented.
# ────────────────────────────────────────────────────────────────────────────

def dbscan_controls() -> tuple[float | None, int, float, float]:
    st.sidebar.subheader("DBSCAN settings")
    automatic_eps = st.sidebar.checkbox("Choose EPS automatically", value=False)
    eps = None
    if not automatic_eps:
        eps = st.sidebar.number_input("EPS", min_value=0.01, value=0.76, step=0.05, format="%.2f")
    min_samples = st.sidebar.slider("min_samples", min_value=3, max_value=40, value=30)
    pca_variance = st.sidebar.slider("PCA variance retained", min_value=0.60, max_value=1.00, value=0.90, step=0.05)
    risk_multiplier = st.sidebar.slider("Elevated-risk multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
    with st.sidebar.expander("How to tune DBSCAN"):
        st.markdown("""
- **EPS** is the neighbourhood radius. Lower EPS makes tighter, smaller clusters and more noise. Higher EPS merges nearby groups; too high can create one large cluster.
- **min_samples** is the number of neighbours required for a dense core. Higher values make DBSCAN stricter, usually increasing noise and removing tiny clusters.
- **-1 means noise**, not a third medical class. It is okay to have more than two clusters: stroke 0/1 is an outcome used after clustering, while clusters describe different patient profiles.
        """)
    return eps, min_samples, pca_variance, risk_multiplier


def meanshift_controls() -> tuple[float | None, float, float, float, int]:
    st.sidebar.subheader("MeanShift settings")
    automatic_bw = st.sidebar.checkbox("Choose Bandwidth automatically", value=True, key="ms_auto_bw")
    bandwidth = None
    if not automatic_bw:
        bandwidth = st.sidebar.number_input(
            "Bandwidth", min_value=0.01, value=1.0, step=0.05, format="%.3f", key="ms_bw"
        )
    quantile = st.sidebar.slider(
        "Bandwidth quantile (auto)", min_value=0.05, max_value=0.60, value=0.25, step=0.05,
        key="ms_quantile",
        help="Controls sklearn estimate_bandwidth. Lower quantile → narrower kernel → more clusters."
    )
    pca_variance = st.sidebar.slider(
        "PCA variance retained", min_value=0.60, max_value=1.00, value=0.90, step=0.05, key="ms_pca"
    )
    risk_multiplier = st.sidebar.slider(
        "Elevated-risk multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1, key="ms_risk"
    )
    min_bin_freq = st.sidebar.slider(
        "min_bin_freq", min_value=1, max_value=30, value=5, key="ms_mbf",
        help="Clusters with fewer than this many core points are discarded."
    )
    with st.sidebar.expander("How to tune MeanShift"):
        st.markdown("""
- **Bandwidth** is the kernel radius. Smaller values create more, tighter clusters; larger values merge nearby modes.
- **Quantile** (0–1) drives the automatic bandwidth estimate. Use values around 0.2–0.35 for clinical data.
- **min_bin_freq** prunes tiny spurious clusters; increase if the algorithm reports too many micro-clusters.
- MeanShift never produces noise points (every patient is assigned to the nearest mode), so noise ratio is always 0.
        """)
    return bandwidth, quantile, pca_variance, risk_multiplier, min_bin_freq


def kmeans_controls() -> tuple[int | None, float, float, int]:
    st.sidebar.subheader("K-Means settings")
    automatic_k = st.sidebar.checkbox("Choose K automatically", value=True, key="km_auto_k")
    n_clusters = None
    if not automatic_k:
        n_clusters = st.sidebar.slider("Number of clusters (K)", min_value=2, max_value=10, value=3, key="km_k")
    max_k = st.sidebar.slider(
        "Maximum K to evaluate", min_value=3, max_value=20, value=15, key="km_max_k",
        help="Automatic mode evaluates K=2 up to this value and selects the highest Silhouette score."
    )
    pca_variance = st.sidebar.slider(
        "PCA variance retained", min_value=0.60, max_value=1.00, value=0.90, step=0.05, key="km_pca"
    )
    risk_multiplier = st.sidebar.slider(
        "Elevated-risk multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1, key="km_risk"
    )
    with st.sidebar.expander("How to tune K-Means"):
        st.markdown("""
- **K** is the number of patient groups. Automatic mode tests several K values and selects the highest Silhouette score.
- **Silhouette Score**: higher is better (better-separated, more cohesive clusters).
- **Davies-Bouldin Index**: lower is better.
- **Inertia** decreases as K increases, so use it with the elbow pattern rather than minimising it blindly.
- K-Means assigns every patient to a cluster, so its noise ratio is always 0.
        """)
    return n_clusters, pca_variance, risk_multiplier, max_k


# ────────────────────────────────────────────────────────────────────────────
# Styling
# ────────────────────────────────────────────────────────────────────────────

def inject_custom_css() -> None:
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

        .stApp, html, body, [class*="css"] {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        h1, h2, h3 {
            font-weight: 700 !important;
            color: #1e293b !important;
            margin-top: 1rem !important;
            margin-bottom: 0.5rem !important;
        }

        section[data-testid="stSidebar"] {
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            border-bottom: 2px solid #e2e8f0;
        }

        .stTabs [data-baseweb="tab"] {
            font-size: 1.05rem !important;
            font-weight: 500 !important;
            color: #64748b !important;
            background-color: #f1f5f9 !important;
            border-radius: 8px 8px 0 0 !important;
            padding: 10px 20px !important;
            border: 1px solid #e2e8f0 !important;
            border-bottom: none !important;
            transition: all 0.2s ease-in-out !important;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: #1e3c72 !important;
            background-color: #e2e8f0 !important;
        }

        .stTabs [aria-selected="true"] {
            color: #1e3c72 !important;
            background-color: #ffffff !important;
            border-top: 3px solid #1e3c72 !important;
            border-left: 1px solid #e2e8f0 !important;
            border-right: 1px solid #e2e8f0 !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            color: #1e3c72 !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            color: #64748b !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
        }

        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 15px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
            transition: transform 0.2s, box-shadow 0.2s;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }

        code {
            color: #0f172a !important;
            background-color: #f1f5f9 !important;
        }

        .stAlert {
            border-radius: 8px !important;
        }

        /* Buttons — friendlier, higher-contrast primary action */
        .stButton > button, .stFormSubmitButton > button {
            border-radius: 10px !important;
            font-weight: 600 !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease !important;
        }
        .stButton > button:hover, .stFormSubmitButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 10px rgba(30, 60, 114, 0.15);
        }

        .premium-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        }

        .hero-card {
            display: flex;
            align-items: center;
            gap: 18px;
            background: linear-gradient(135deg, #f8fbff 0%, #eef5ff 100%);
            border: 1px solid #dbe7f5;
            border-radius: 18px;
            padding: 28px;
            margin: 8px 0 22px 0;
        }
        .hero-icon { font-size: 3rem; }
        .hero-title { font-size: 2rem; font-weight: 700; color: #163b70; }
        .hero-subtitle { color: #52657d; font-size: 1.05rem; margin-top: 4px; }

        .group-card {
            background: #ffffff;
            border: 1px solid #dfe7f0;
            border-radius: 14px;
            padding: 18px;
            min-height: 120px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }
        .group-number { font-size: 1.2rem; font-weight: 700; color: #1e3c72; }
        .group-patients { font-size: 1.45rem; font-weight: 700; margin-top: 10px; color: #0f172a; }
        .group-rate { color: #64748b; margin-top: 6px; }

        /* ── Result cards ──────────────────────────────────────────── */
        .result-hero {
            border-radius: 18px;
            padding: 32px 28px;
            margin-bottom: 22px;
            text-align: center;
        }
        .result-hero.high { background: linear-gradient(135deg, #fef2f2, #fee2e2); border: 2px solid #fca5a5; }
        .result-hero.low  { background: linear-gradient(135deg, #f0fdf4, #dcfce7); border: 2px solid #86efac; }
        .result-hero.unknown { background: linear-gradient(135deg, #fefce8, #fef9c3); border: 2px solid #fde68a; }

        .result-icon   { font-size: 3.5rem; margin-bottom: 6px; }
        .result-label  { font-size: 1.6rem; font-weight: 800; margin-top: 4px; }
        .result-label.high { color: #b91c1c; }
        .result-label.low  { color: #15803d; }
        .result-label.unknown { color: #a16207; }
        .result-note   { font-size: 0.92rem; color: #475569; margin-top: 10px; line-height: 1.6; max-width: 600px; margin-left: auto; margin-right: auto; }

        /* ── Form styling ───────────────────────────────────────────── */
        .form-section-title {
            font-size: 0.75rem;
            font-weight: 700;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin: 20px 0 8px 0;
            padding-bottom: 6px;
            border-bottom: 1px solid #f1f5f9;
        }

        /* ── Insight card ─────────────────────────────────────────── */
        .insight-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #3b82f6;
            border-radius: 10px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }
        .insight-card.warn { border-left-color: #ef4444; }
        .insight-card.ok   { border-left-color: #22c55e; }

        /* ── Consensus bar ───────────────────────────────────────── */
        .consensus-bar {
            display: flex;
            gap: 16px;
            justify-content: center;
            margin: 16px 0 20px 0;
            flex-wrap: wrap;
        }
        .consensus-dot {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3rem;
            font-weight: 700;
            color: #ffffff;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin: 0 auto;
        }
        .consensus-dot.high { background: #ef4444; }
        .consensus-dot.low  { background: #22c55e; }
        .consensus-dot.noise { background: #cbd5e1; }
        .consensus-name { font-size: 0.75rem; text-align: center; color: #475569; margin-top: 6px; font-weight: 600; }

        /* Disclaimer */
        .disclaimer {
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-radius: 10px;
            padding: 14px 18px;
            font-size: 0.85rem;
            color: #92400e;
            line-height: 1.5;
            margin-top: 24px;
        }

        /* Step pill (how it works) */
        .stat-pill {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 18px;
            text-align: center;
            min-height: 140px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }

        /* Footer */
        .app-footer {
            text-align: center;
            color: #94a3b8;
            font-size: 0.78rem;
            margin-top: 48px;
            padding: 20px 0;
            border-top: 1px solid #e2e8f0;
        }
        </style>
    """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Deep-dive (technical) clustering views — one per algorithm.
# These carry the full parameter-search tables / plots, and now live inside
# "Technical Details → Algorithm Deep-Dive" so casual users aren't confronted
# with them, while power users can still reach every chart.
# ────────────────────────────────────────────────────────────────────────────

def show_dbscan_clustering(result) -> None:
    st.header("🔍 DBSCAN Clustering")

    clean_df = result.data[result.data["cluster"] != -1]
    stroke_rates = clean_df.groupby("cluster")["stroke"].mean()
    max_stroke_rate = stroke_rates.max() if not stroke_rates.empty else 0.0
    baseline_stroke_rate = result.data["stroke"].mean()

    st.markdown(f"""
    <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px;">
        <div style="flex: 1; min-width: 120px; background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); text-align: center;">
            <div style="font-size: 0.68rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; white-space: nowrap;">Clusters Found</div>
            <div style="font-size: 1.15rem; color: #000000; font-weight: 700; margin-top: 3px;">{result.n_clusters}</div>
        </div>
        <div style="flex: 1; min-width: 120px; background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); text-align: center;">
            <div style="font-size: 0.68rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; white-space: nowrap;">Noise Patients</div>
            <div style="font-size: 1.15rem; color: #C0392B; font-weight: 700; margin-top: 3px;">{result.noise_ratio:.2%}</div>
        </div>
        <div style="flex: 1; min-width: 120px; background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); text-align: center;">
            <div style="font-size: 0.68rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; white-space: nowrap;">Selected EPS</div>
            <div style="font-size: 1.15rem; color: #000000; font-weight: 700; margin-top: 3px;">{result.selected_eps:.3f}</div>
        </div>
        <div style="flex: 1; min-width: 120px; background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); text-align: center;">
            <div style="font-size: 0.68rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; white-space: nowrap;">Silhouette Score</div>
            <div style="font-size: 1.15rem; color: #000000; font-weight: 700; margin-top: 3px;">{"N/A" if np.isnan(result.silhouette) else f"{result.silhouette:.4f}"}</div>
        </div>
        <div style="flex: 1; min-width: 120px; background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); text-align: center;">
            <div style="font-size: 0.68rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; white-space: nowrap;">Davies-Bouldin</div>
            <div style="font-size: 1.15rem; color: #000000; font-weight: 700; margin-top: 3px;">{"N/A" if np.isnan(result.davies_bouldin) else f"{result.davies_bouldin:.4f}"}</div>
        </div>
        <div style="flex: 1; min-width: 120px; background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); text-align: center;">
            <div style="font-size: 0.68rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; white-space: nowrap;">Max stroke rate</div>
            <div style="font-size: 1.15rem; color: #000000; font-weight: 700; margin-top: 3px;">{max_stroke_rate:.2%}</div>
        </div>
        <div style="flex: 1; min-width: 120px; background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); text-align: center;">
            <div style="font-size: 0.68rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; white-space: nowrap;">Baseline stroke</div>
            <div style="font-size: 1.15rem; color: #000000; font-weight: 700; margin-top: 3px;">{baseline_stroke_rate:.2%}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if result.n_clusters < 2:
        st.warning("This setting found fewer than two clusters. Try lowering `min_samples` or EPS, or return to automatic EPS.")
    elif result.noise_ratio > .60:
        st.warning("More than 60% of patients are noise. Consider a slightly higher EPS or lower `min_samples`.")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(plot_cluster_scatter(result), use_container_width=True)
    with right:
        st.plotly_chart(plot_k_distance(result), use_container_width=True)

    st.subheader("EPS Parameter Search")
    st.plotly_chart(plot_eps_search(result), use_container_width=True)

    search_table = result.parameter_results.copy()
    search_table["noise_percent"] = search_table["noise_ratio"] * 100
    st.dataframe(search_table, use_container_width=True, hide_index=True)

    st.subheader("Stroke Risk Class Distribution")
    st.plotly_chart(plot_stroke_by_cluster(result), use_container_width=True)

    st.subheader("Cluster Summary Table")
    _render_cluster_summary_table(result)


def _render_cluster_summary_table(result) -> None:
    rates = result.cluster_summary.copy()
    rates["elevated_risk"] = rates["elevated_risk"].map({True: "Yes", False: "No"})
    rates["cluster"] = rates["cluster"].astype(int)
    rates["patients"] = rates["patients"].astype(int)
    rates["stroke_cases"] = rates["stroke_cases"].astype(int)
    rates["age_mean"] = rates["age_mean"].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
    rates["glucose_mean"] = rates["glucose_mean"].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
    rates["bmi_mean"] = rates["bmi_mean"].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
    rates["stroke_rate_pct"] = rates["stroke_rate_pct"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
    rates = rates.rename(columns={
        "cluster": "Cluster", "patients": "Patients", "bmi_mean": "Bmi_mean",
    })
    rates_display = rates[["Cluster", "Patients", "stroke_cases", "age_mean",
                            "glucose_mean", "Bmi_mean", "stroke_rate_pct", "elevated_risk"]]
    cluster_ids = rates_display["Cluster"].tolist()
    transposed = rates_display.astype(str).set_index("Cluster").T.reset_index()
    transposed.columns = ["Attribute"] + [f"Cluster {int(c)}" for c in cluster_ids]

    st.dataframe(transposed, use_container_width=True, hide_index=True)


def show_meanshift_clustering(result) -> None:
    st.header("🌊 MeanShift Clustering")

    clean_df = result.data[result.data["cluster"] != -1]
    stroke_rates = clean_df.groupby("cluster")["stroke"].mean()
    max_stroke_rate = stroke_rates.max() if not stroke_rates.empty else 0.0
    baseline_stroke_rate = result.data["stroke"].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Clusters Found", result.n_clusters)
    c2.metric("Selected Bandwidth", f"{result.selected_eps:.4f}" if result.selected_eps is not None else "N/A")
    c3.metric("PCA Components", result.n_components)
    c4.metric("Silhouette", "N/A" if np.isnan(result.silhouette) else f"{result.silhouette:.4f}")
    c5.metric("Davies-Bouldin", "N/A" if np.isnan(result.davies_bouldin) else f"{result.davies_bouldin:.4f}")
    c6.metric("Max Stroke Rate", f"{max_stroke_rate:.2%}")

    if result.n_clusters < 2:
        st.warning("Fewer than two clusters found. Try a lower quantile value or reduce the bandwidth manually.")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(plot_meanshift_cluster_scatter(result), use_container_width=True)
    with right:
        st.plotly_chart(plot_bandwidth_sweep(result), use_container_width=True)

    st.subheader("Bandwidth Parameter Search")
    sweep_table = result.parameter_results.copy()
    sweep_table["noise_percent"] = sweep_table["noise_ratio"] * 100
    st.dataframe(sweep_table, use_container_width=True, hide_index=True)

    st.subheader("Stroke Risk Class Distribution")
    left2, right2 = st.columns(2)
    with left2:
        st.plotly_chart(plot_stroke_by_cluster(result), use_container_width=True)
    with right2:
        st.plotly_chart(plot_cluster_size_distribution(result), use_container_width=True)

    st.subheader("Clinical Risk Factor Profile per Cluster")
    st.plotly_chart(plot_cluster_profile_radar(result), use_container_width=True)

    pca_importance, cluster_deviations = calculate_feature_contributions(result)
    with st.expander("📊 Attribute Contribution Analysis"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("##### Overall Feature Importance (PCA Weight)")
            st.dataframe(pca_importance, use_container_width=True, hide_index=True)
        with c2:
            st.write("##### Cluster Mean Deviation from Baseline (%)")
            st.dataframe(cluster_deviations, use_container_width=True)

    st.subheader("Cluster Summary Table")
    _render_cluster_summary_table(result)


def show_kmeans_clustering(result) -> None:
    st.header("🎯 K-Means Clustering")
    clean_df = result.data[result.data["cluster"] != -1]
    stroke_rates = clean_df.groupby("cluster")["stroke"].mean()
    max_stroke_rate = stroke_rates.max() if not stroke_rates.empty else 0.0
    baseline_stroke_rate = result.data["stroke"].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Clusters (K)", result.selected_k)
    c2.metric("PCA Components", result.n_components)
    c3.metric("Silhouette", f"{result.silhouette:.4f}")
    c4.metric("Davies-Bouldin", f"{result.davies_bouldin:.4f}")
    c5.metric("Max Stroke Rate", f"{max_stroke_rate:.2%}")
    c6.metric("Baseline Stroke", f"{baseline_stroke_rate:.2%}")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(plot_cluster_scatter(result), use_container_width=True)
    with right:
        search = result.parameter_results.copy()
        fig = px.line(search, x="k", y="silhouette", markers=True,
                      hover_data=["inertia", "davies_bouldin"],
                      title="K Search: Silhouette Score")
        fig.add_vline(x=result.selected_k, line_dash="dash", line_color="#C0392B")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("K Parameter Search")
    st.dataframe(result.parameter_results, use_container_width=True, hide_index=True)

    st.subheader("Stroke Risk Class Distribution")
    left2, right2 = st.columns(2)
    with left2:
        st.plotly_chart(plot_stroke_by_cluster(result), use_container_width=True)
    with right2:
        st.plotly_chart(plot_cluster_size_distribution(result), use_container_width=True)

    st.subheader("Clinical Risk Factor Profile per Cluster")
    st.plotly_chart(plot_cluster_profile_radar(result), use_container_width=True)

    pca_importance, cluster_deviations = calculate_feature_contributions(result)
    with st.expander("📊 Attribute Contribution Analysis"):
        a, b = st.columns(2)
        with a:
            st.write("##### Overall Feature Importance (PCA Weight)")
            st.dataframe(pca_importance, use_container_width=True, hide_index=True)
        with b:
            st.write("##### Cluster Mean Deviation from Baseline (%)")
            st.dataframe(cluster_deviations, use_container_width=True)

    st.subheader("Cluster Summary Table")
    _render_cluster_summary_table(result)


# ────────────────────────────────────────────────────────────────────────────
# Patient Explorer (multi-model consensus) — the app's new landing page
# ────────────────────────────────────────────────────────────────────────────

def _explain_patient_factors(raw_patient: dict, baseline: dict) -> list[str]:
    reasons: list[tuple[float, str]] = []

    def _pct_diff(value: float, base: float) -> float:
        return (value - base) / base if base else 0.0

    age_diff = _pct_diff(raw_patient["age"], baseline["age"])
    if abs(age_diff) >= 0.15:
        direction = "older" if age_diff > 0 else "younger"
        reasons.append((abs(age_diff), (
            f"**Age** — {raw_patient['age']:.0f} years, noticeably {direction} than "
            f"the average patient ({baseline['age']:.0f})."
        )))

    glu_diff = _pct_diff(raw_patient["avg_glucose_level"], baseline["avg_glucose_level"])
    if abs(glu_diff) >= 0.15:
        direction = "higher" if glu_diff > 0 else "lower"
        reasons.append((abs(glu_diff), (
            f"**Average glucose level** — {raw_patient['avg_glucose_level']:.0f} mg/dL, "
            f"{direction} than the average patient ({baseline['avg_glucose_level']:.0f} mg/dL)."
        )))

    bmi_diff = _pct_diff(raw_patient["bmi"], baseline["bmi"])
    if abs(bmi_diff) >= 0.15:
        direction = "higher" if bmi_diff > 0 else "lower"
        reasons.append((abs(bmi_diff), (
            f"**BMI** — {raw_patient['bmi']:.1f}, {direction} than the average patient "
            f"({baseline['bmi']:.1f})."
        )))

    if raw_patient["hypertension"] == 1:
        reasons.append((0.5, (
            f"**Hypertension** — present. Only about {baseline['hypertension_rate']:.0f}% "
            f"of patients in the dataset have hypertension."
        )))

    if raw_patient["heart_disease"] == 1:
        reasons.append((0.5, (
            f"**Heart disease** — present. Only about {baseline['heart_disease_rate']:.0f}% "
            f"of patients in the dataset have heart disease."
        )))

    if raw_patient.get("smoking_status") == "smokes":
        reasons.append((0.3, "**Smoking status** — currently smokes."))

    reasons.sort(key=lambda r: r[0], reverse=True)
    return [sentence for _, sentence in reasons[:4]]


def _risk_gauge(stroke_rate_pct: float, baseline_pct: float) -> go.Figure:
    color = "#dc2626" if stroke_rate_pct > baseline_pct * 1.5 else (
            "#f59e0b" if stroke_rate_pct > baseline_pct else "#059669")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=stroke_rate_pct,
        number={"suffix": "%", "font": {"size": 38, "color": color, "family": "Outfit"}},
        delta={"reference": baseline_pct, "relative": False, "suffix": "%",
               "increasing": {"color": "#dc2626"}, "decreasing": {"color": "#059669"}},
        title={"text": "Stroke Rate in This Group", "font": {"size": 14, "color": "#64748b", "family": "Outfit"}},
        gauge={
            "axis": {"range": [0, max(30, stroke_rate_pct * 1.4)], "tickwidth": 1, "tickcolor": "#cbd5e1",
                     "tickfont": {"size": 10, "color": "#94a3b8"}},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": "#f1f5f9",
            "borderwidth": 0,
            "steps": [
                {"range": [0, baseline_pct], "color": "#ecfdf5"},
                {"range": [baseline_pct, baseline_pct * 1.5], "color": "#fefce8"},
                {"range": [baseline_pct * 1.5, max(30, stroke_rate_pct * 1.4)], "color": "#fef2f2"},
            ],
            "threshold": {
                "line": {"color": "#475569", "width": 2.5},
                "thickness": 0.8,
                "value": baseline_pct,
            },
        },
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=30, r=30, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit"),
    )
    return fig


def _comparison_radar(patient: dict, cluster_profile: dict, baseline: dict) -> go.Figure:
    cats = ["Age", "Glucose", "BMI"]

    def _norm(val, key):
        maxes = {"age": 100, "avg_glucose_level": 300, "bmi": 60}
        return min(val / maxes.get(key, 1), 1.0)

    patient_vals = [
        _norm(patient["age"], "age"),
        _norm(patient["avg_glucose_level"], "avg_glucose_level"),
        _norm(patient["bmi"], "bmi"),
    ]
    cluster_vals = [
        _norm(cluster_profile.get("cluster_age_mean", 0.0), "age"),
        _norm(cluster_profile.get("cluster_glucose_mean", 0.0), "avg_glucose_level"),
        _norm(cluster_profile.get("cluster_bmi_mean", 0.0), "bmi"),
    ]
    baseline_vals = [
        _norm(baseline["age"], "age"),
        _norm(baseline["avg_glucose_level"], "avg_glucose_level"),
        _norm(baseline["bmi"], "bmi"),
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=patient_vals + [patient_vals[0]],
        theta=cats + [cats[0]],
        fill="toself", name="This Patient",
        fillcolor="rgba(59,130,246,0.15)",
        line=dict(color="#3b82f6", width=2.5),
    ))
    fig.add_trace(go.Scatterpolar(
        r=cluster_vals + [cluster_vals[0]],
        theta=cats + [cats[0]],
        fill="toself", name="Cluster Average",
        fillcolor="rgba(249,115,22,0.10)",
        line=dict(color="#f97316", width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatterpolar(
        r=baseline_vals + [baseline_vals[0]],
        theta=cats + [cats[0]],
        fill="toself", name="Dataset Average",
        fillcolor="rgba(100,116,139,0.06)",
        line=dict(color="#94a3b8", width=1.5, dash="dash"),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor="#e2e8f0"),
            angularaxis=dict(tickfont=dict(size=13, color="#334155", family="Outfit"), gridcolor="#e2e8f0"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5,
                    font=dict(size=11, family="Outfit")),
        margin=dict(l=50, r=50, t=30, b=60),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit"),
    )
    return fig


def _render_batch_analysis(data: pd.DataFrame, results_dict: dict, artifacts_map: dict) -> None:
    """Batch-classify many patients at once from an uploaded CSV.

    Validates the uploaded file against ``RAW_INPUT_COLUMNS`` before doing
    any work, reports every problem row instead of failing on the first
    one, and lets the user download the annotated results. This lets the
    same clustering pipeline used for the 4,981-row training set be
    applied to an arbitrarily large batch of new patients.
    """
    st.subheader("📁 Batch Patient Analysis (CSV Upload)")
    st.caption(
        "Upload a CSV of multiple patients to classify all of them at once using the selected model. "
        f"Required columns: {', '.join(RAW_INPUT_COLUMNS)}."
    )

    with st.expander("📄 Expected CSV format", expanded=False):
        st.dataframe(data[RAW_INPUT_COLUMNS].head(3), use_container_width=True, hide_index=True)
        st.caption("hypertension and heart_disease must be 0 or 1. ever_married must be 'Yes' or 'No'.")

    uploaded = st.file_uploader("Upload patient CSV", type=["csv"], key="batch_csv_upload")
    if uploaded is None:
        return

    try:
        batch_df = pd.read_csv(uploaded)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        st.error(f"⚠️ Could not read this file as CSV: {exc}")
        return

    if batch_df.empty:
        st.error("⚠️ The uploaded file has no rows.")
        return

    missing_cols = sorted(set(RAW_INPUT_COLUMNS) - set(batch_df.columns))
    if missing_cols:
        st.error(f"⚠️ Missing required column(s): {', '.join(missing_cols)}. Please match the expected format above.")
        return

    numeric_cols = ["age", "avg_glucose_level", "bmi"]
    binary_cols = ["hypertension", "heart_disease"]
    row_errors: list[str] = []
    for col in numeric_cols:
        bad_rows = batch_df[pd.to_numeric(batch_df[col], errors="coerce").isna()].index.tolist()
        if bad_rows:
            row_errors.append(f"'{col}' has non-numeric values in row(s): {[r + 1 for r in bad_rows]}")
    for col in binary_cols:
        bad_rows = batch_df[~batch_df[col].isin([0, 1])].index.tolist()
        if bad_rows:
            row_errors.append(f"'{col}' must be 0 or 1 — check row(s): {[r + 1 for r in bad_rows]}")

    if row_errors:
        st.error("⚠️ Found problems in the uploaded file:\n\n" + "\n".join(f"- {e}" for e in row_errors))
        return

    algo_choice = st.selectbox(
        "Model to use for batch classification", list(results_dict.keys()), key="batch_algo_choice"
    )
    predict_fn_map = {
        "K-Means": predict_kmeans_patient if KMEANS_AVAILABLE else None,
        "DBSCAN": predict_dbscan_patient,
        "MeanShift": predict_meanshift_patient,
    }
    predict_fn = predict_fn_map.get(algo_choice)
    artifacts = artifacts_map.get(algo_choice)

    if predict_fn is None or artifacts is None:
        st.error(f"⚠️ {algo_choice} is not available for batch prediction.")
        return

    if st.button(f"🚀 Classify {len(batch_df):,} patients with {algo_choice}", type="primary"):
        results_rows = []
        progress = st.progress(0.0, text="Classifying patients...")
        n = len(batch_df)
        for i, row in enumerate(batch_df.itertuples(index=False)):
            raw_patient = {col: getattr(row, col) for col in RAW_INPUT_COLUMNS}
            try:
                pred = predict_fn(raw_patient, artifacts)
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning("Batch row %d failed: %s", i, exc)
                pred = {"predicted_cluster": -1, "matched_profile": "Error", "cluster_stroke_rate_pct": 0.0,
                        "cluster_elevated_risk": False}
            results_rows.append(pred)
            if n:
                progress.progress((i + 1) / n, text=f"Classifying patients... ({i + 1}/{n})")
        progress.empty()

        results_df = pd.concat([batch_df.reset_index(drop=True), pd.DataFrame(results_rows)], axis=1)
        n_errors = int((results_df["predicted_cluster"] == -1).sum()) if "predicted_cluster" in results_df else 0
        n_elevated = int(results_df.get("cluster_elevated_risk", pd.Series(dtype=bool)).sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Patients classified", f"{n - n_errors:,} / {n:,}")
        c2.metric("Flagged elevated-risk", f"{n_elevated:,}")
        c3.metric("Model used", algo_choice)

        st.dataframe(results_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download results as CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name=f"batch_cluster_results_{algo_choice.lower()}.csv",
            mime="text/csv",
        )


def _render_patient_predict_form(data: pd.DataFrame, results_dict: dict, artifacts_map: dict) -> None:
    """
    Patient Explorer for unsupervised clustering.

    1. Takes a patient's characteristics.
    2. Runs predictions across all configured models.
    3. Groups the patient into clusters and presents consensus.

    It does NOT predict an individual's stroke outcome.
    """

    st.subheader("🔍 Explore Patient Profile")
    st.caption(
        "Enter patient characteristics to identify the most similar patient cluster discovered by the unsupervised learning models."
    )

    with st.expander("💡 How does this work?", expanded=False):
        st.markdown(
            """
            **This system uses unsupervised learning to discover groups of similar patients.**

            1. 👤 Enter the patient's characteristics.
            2. 🔍 The clustering model compares the patient with the discovered clusters.
            3. 👥 The most similar cluster is identified for each algorithm.
            4. 📊 The characteristics of the matched clusters are shown side-by-side.
            5. 🩺 The observed stroke rate is used only to describe the cohort.

            **Important:** The observed stroke rate is NOT an individual stroke prediction.
            """
        )

    baseline = {
        "age": float(data["age"].mean()),
        "avg_glucose_level": float(data["avg_glucose_level"].mean()),
        "bmi": float(data["bmi"].mean()),
        "hypertension_rate": float(data["hypertension"].mean() * 100),
        "heart_disease_rate": float(data["heart_disease"].mean() * 100),
        "stroke_rate": float(data["stroke"].mean()),
        "total_patients": len(data),
    }

    with st.form("patient_unsupervised_explorer_form"):
        st.markdown('<div class="form-section-title">Demographics</div>', unsafe_allow_html=True)
        dem1, dem2 = st.columns(2)
        with dem1:
            age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=55.0, step=1.0,
                                  help="Patient's current age in years.", key="explore_patient_age")
        with dem2:
            ever_married = st.selectbox("Marital Status", ["Married", "Not Married"],
                                        help="Has the patient ever been married?", key="explore_patient_married")

        st.markdown('<div class="form-section-title">Clinical Measurements</div>', unsafe_allow_html=True)
        clin1, clin2 = st.columns(2)
        with clin1:
            avg_glucose = st.number_input(
                "Average Glucose Level (mg/dL)",
                min_value=50.0, max_value=400.0, value=106.0, step=1.0,
                help="Fasting or average blood glucose level. Normal range is typically 70–100 mg/dL.",
                key="explore_patient_glucose"
            )
        with clin2:
            bmi = st.number_input(
                "Body Mass Index (BMI)",
                min_value=10.0, max_value=70.0, value=28.0, step=0.1,
                help="BMI = weight (kg) / height² (m). Normal: 18.5–24.9, Overweight: 25–29.9, Obese: ≥30.",
                key="explore_patient_bmi"
            )

        st.markdown('<div class="form-section-title">Medical History</div>', unsafe_allow_html=True)
        med1, med2 = st.columns(2)
        with med1:
            hypertension = st.selectbox("Hypertension", ["No", "Yes"],
                                        help="Has the patient been diagnosed with high blood pressure?",
                                        key="explore_patient_hypertension")
        with med2:
            heart_disease = st.selectbox("Heart Disease", ["No", "Yes"],
                                         help="Has the patient been diagnosed with any form of heart disease?",
                                         key="explore_patient_heart")

        st.markdown('<div class="form-section-title">Lifestyle</div>', unsafe_allow_html=True)
        smoking = st.selectbox(
            "Smoking Status",
            ["Never smoked", "Formerly smoked", "Currently smokes", "Unknown"],
            help="Patient's smoking history.",
            key="explore_patient_smoking"
        )

        st.markdown("")
        submitted = st.form_submit_button("🔍  Analyse Patient Profile", use_container_width=True, type="primary")

    if not submitted:
        st.markdown("---")
        st.markdown("### 💡 How It Works")
        how1, how2, how3 = st.columns(3)
        with how1:
            st.markdown("""
            <div class="stat-pill">
                <div style="font-size: 2rem;">📋</div>
                <div style="font-weight: 700; color: #0f172a; margin-top: 8px;">1. Enter Details</div>
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 4px;">Fill in the patient's clinical data using the form above.</div>
            </div>
            """, unsafe_allow_html=True)
        with how2:
            st.markdown("""
            <div class="stat-pill">
                <div style="font-size: 2rem;">🤖</div>
                <div style="font-weight: 700; color: #0f172a; margin-top: 8px;">2. AI Multi-Model Check</div>
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 4px;">Three distinct clustering algorithms will group-match the patient.</div>
            </div>
            """, unsafe_allow_html=True)
        with how3:
            st.markdown("""
            <div class="stat-pill">
                <div style="font-size: 2rem;">📊</div>
                <div style="font-weight: 700; color: #0f172a; margin-top: 8px;">3. Consensus View</div>
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 4px;">Review and compare similarities across all clustering perspectives.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">
            ⚠️ <b>Medical Disclaimer:</b> This tool is a research-grade decision-support system. It does <b>not</b> provide medical diagnoses. Always consult a qualified healthcare professional.
        </div>
        """, unsafe_allow_html=True)
        return

    smoking_map = {"Never smoked": "never smoked", "Formerly smoked": "formerly smoked",
                   "Currently smokes": "smokes", "Unknown": "Unknown"}
    raw_patient = {
        "age": age,
        "avg_glucose_level": avg_glucose,
        "bmi": bmi,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "ever_married": "Yes" if ever_married == "Married" else "No",
        "smoking_status": smoking_map[smoking],
    }

    # Business-rule sanity checks: numeric bounds on the widgets already
    # prevent hard errors, but implausible *combinations* (e.g. a young
    # child flagged as married) are still worth a soft warning rather
    # than a silent, potentially misleading prediction.
    plausibility_notes = []
    if age < 18 and ever_married == "Married":
        plausibility_notes.append("Age is under 18 but marital status is 'Married' — please double-check this combination.")
    if bmi < 12 or bmi > 60:
        plausibility_notes.append(f"BMI of {bmi:.1f} is outside the typical clinical range (12–60) — results may be unreliable.")
    if avg_glucose < 50:
        plausibility_notes.append(f"Average glucose of {avg_glucose:.0f} mg/dL is unusually low — please verify this value.")
    if plausibility_notes:
        st.warning("⚠️ " + " ".join(plausibility_notes))

    models = {}
    if KMEANS_AVAILABLE and "K-Means" in results_dict:
        models["K-Means"] = (results_dict["K-Means"], artifacts_map["K-Means"], predict_kmeans_patient)
    if "DBSCAN" in results_dict:
        models["DBSCAN"] = (results_dict["DBSCAN"], artifacts_map["DBSCAN"], predict_dbscan_patient)
    if "MeanShift" in results_dict:
        models["MeanShift"] = (results_dict["MeanShift"], artifacts_map["MeanShift"], predict_meanshift_patient)

    predictions = {}
    failed_models: list[str] = []
    for name, (result, artifacts, predict_fn) in models.items():
        try:
            pred = predict_fn(raw_patient, artifacts)
            predictions[name] = pred
        except (ValueError, KeyError, TypeError) as exc:
            # These are the expected failure modes: a missing/invalid raw
            # input column (ValueError from predict_new_patient's own
            # validation), a malformed artifacts bundle (KeyError), or a
            # type mismatch feeding sklearn (TypeError). Anything else is
            # an unexpected bug and is deliberately allowed to propagate
            # so it surfaces during testing rather than being hidden.
            logger.warning("Prediction failed for %s: %s", name, exc)
            failed_models.append(name)
            predictions[name] = {"predicted_cluster": -1, "cluster_elevated_risk": False,
                                  "cluster_stroke_rate_pct": 0.0, "cluster_age_mean": 0.0,
                                  "cluster_glucose_mean": 0.0, "cluster_bmi_mean": 0.0,
                                  "cluster_patients": 0, "error": True}

    if failed_models:
        st.warning(
            f"⚠️ {', '.join(failed_models)} could not classify this patient (unusual input combination) "
            "and were excluded from the consensus below."
        )

    valid_preds = {n: p for n, p in predictions.items() if p.get("predicted_cluster", -1) != -1 and not p.get("error")}
    elevated_count = sum(1 for p in valid_preds.values() if p["cluster_elevated_risk"])
    total_valid = len(valid_preds)

    if total_valid == 0:
        consensus = "unknown"
    elif elevated_count > total_valid / 2:
        consensus = "high"
    else:
        consensus = "low"

    baseline_pct = baseline["stroke_rate"] * 100

    st.markdown("---")
    st.markdown("## 📊 Step 2 — Analysis Results")

    if consensus == "high":
        icon, label, css_class = "⚠️", "Higher Risk Profile", "high"
        description = (
            "This patient's characteristics place them in a <b>higher-risk group</b>. "
            "Patients with similar clinical profiles in our dataset had an above-average stroke incidence."
        )
    elif consensus == "low":
        icon, label, css_class = "✅", "Typical Risk Profile", "low"
        description = (
            "This patient's characteristics place them in a <b>typical-risk group</b>. "
            "Patients with similar clinical profiles in our dataset had an average or below-average stroke incidence."
        )
    else:
        icon, label, css_class = "❓", "Inconclusive", "unknown"
        description = (
            "The AI models could not confidently classify this patient into a clear risk group. "
            "This may happen if the patient's profile is very unusual compared to the training data."
        )

    st.markdown(f"""
    <div class="result-hero {css_class}">
        <div class="result-icon">{icon}</div>
        <div class="result-label {css_class}">{label}</div>
        <div class="result-note">{description}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🤖 AI Model Consensus")
    st.caption("The results of running all three clustering algorithms on the entered patient profile.")

    consensus_html = '<div class="consensus-bar">'
    for name in ["K-Means", "DBSCAN", "MeanShift"]:
        pred = predictions.get(name)
        if pred and pred.get("predicted_cluster", -1) != -1 and not pred.get("error"):
            is_elevated = pred["cluster_elevated_risk"]
            dot_class = "high" if is_elevated else "low"
            dot_icon = "⚠️" if is_elevated else "✅"
        else:
            dot_class = "noise"
            dot_icon = "—"
        consensus_html += f"""
        <div>
            <div class="consensus-dot {dot_class}">{dot_icon}</div>
            <div class="consensus-name">{name}</div>
        </div>"""
    consensus_html += '</div>'

    agree_text = f"**{elevated_count}/{total_valid}** models identified elevated risk" if total_valid > 0 else "No models returned a valid result"
    st.markdown(consensus_html, unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; color:#64748b; font-size:0.88rem;'>{agree_text}</p>", unsafe_allow_html=True)

    st.markdown("### 🧬 Unsupervised Patient Group Placement")
    st.caption("Explore which cluster (patient group) the patient falls into according to each AI algorithm.")

    visible_tab_names = [f"{name} Clustering" for name in models.keys()]
    if visible_tab_names:
        tabs = st.tabs(visible_tab_names)
        for tab, name in zip(tabs, models.keys()):
            with tab:
                pred = predictions.get(name, {})
                result, artifacts, _ = models.get(name, (None, None, None))

                if pred.get("error") or pred.get("predicted_cluster", -1) == -1:
                    st.markdown(f"""
                    <div style="background-color: #fefce8; border-left: 5px solid #eab308; padding: 18px; border-radius: 8px; margin-bottom: 20px;">
                        <h5 style="margin: 0; color: #a16207;">🔮 Outlier / Noise Region (No Group Match)</h5>
                        <div style="margin-top: 6px; font-size: 0.92rem; color: #713f12; line-height: 1.5;">
                            This algorithm classified the patient as a <b>Noise outlier (Cluster -1)</b>.
                            This means their combination of clinical metrics is too rare or unique relative to the main cohorts
                            found in the historical database to place them in a standard group.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    cluster_id = pred["predicted_cluster"]
                    elevated = pred["cluster_elevated_risk"]

                    cluster_df = result.data[result.data["cluster"] == cluster_id]
                    total_pts = len(cluster_df)
                    pct_pts = (total_pts / len(result.data)) * 100
                    avg_age = cluster_df["age"].mean()
                    avg_gl = cluster_df["avg_glucose_level"].mean()
                    avg_bmi = cluster_df["bmi"].mean()
                    stroke_rate = cluster_df["stroke"].mean() * 100

                    hypertension_rate = cluster_df["hypertension"].mean() * 100
                    heart_rate = cluster_df["heart_disease"].mean() * 100

                    smoke_counts = cluster_df["smoking_status"].value_counts(normalize=True) * 100
                    smokes_pct = smoke_counts.get("smokes", 0.0)
                    former_pct = smoke_counts.get("formerly smoked", 0.0)
                    never_pct = smoke_counts.get("never smoked", 0.0)

                    badge_style = "background-color: #fef2f2; border-left: 5px solid #ef4444; padding: 18px;" if elevated else "background-color: #f0fdf4; border-left: 5px solid #22c55e; padding: 18px;"
                    badge_text = "Higher-Risk Clinical Profile" if elevated else "Typical-Risk Clinical Profile"

                    st.markdown(f"""
                    <div style="{badge_style} border-radius: 8px; margin-bottom: 20px;">
                        <h4 style="margin: 0; color: {'#b91c1c' if elevated else '#15803d'}; font-size: 1.15rem;">💬 You belong to Cluster {cluster_id}</h4>
                        <div style="font-size: 0.95rem; font-weight: 700; color: {'#7f1d1d' if elevated else '#064e3b'}; margin-top: 4px;">{badge_text}</div>
                        <div style="font-size: 0.88rem; color: #475569; margin-top: 4px;">Your profile is most similar to patients in this group.</div>
                    </div>
                    """, unsafe_allow_html=True)

                    col_left, col_right = st.columns([1, 1])
                    with col_left:
                        st.markdown(f"""
                        <div style="font-size: 0.95rem; line-height: 1.9; background-color: #ffffff; border: 1px dashed #cbd5e1; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                            <strong>👥 Patients in this group:</strong> {total_pts:,} ({pct_pts:.1f}% of total database)<br>
                            <strong>🎂 Average age:</strong> {avg_age:.1f} years <span style="color: #64748b;">(This Patient: {age:.0f})</span><br>
                            <strong>🩸 Average glucose:</strong> {avg_gl:.1f} mg/dL <span style="color: #64748b;">(This Patient: {avg_glucose:.0f})</span><br>
                            <strong>⚖️ Average BMI:</strong> {avg_bmi:.1f} <span style="color: #64748b;">(This Patient: {bmi:.1f})</span><br>
                            <strong>❤️ Hypertension prevalence:</strong> {hypertension_rate:.1f}% have it <span style="color: #64748b;">(This Patient: {hypertension})</span><br>
                            <strong>💔 Heart Disease prevalence:</strong> {heart_rate:.1f}% have it <span style="color: #64748b;">(This Patient: {heart_disease})</span><br>
                            <strong>🚬 Smoking status in group:</strong> Active Smoker ({smokes_pct:.1f}%), Former Smoker ({former_pct:.1f}%), Never Smoked ({never_pct:.1f}%) <span style="color: #64748b;">(This Patient: {smoking})</span>
                            <hr style="margin: 12px 0; border: 0; border-top: 1px solid #e2e8f0;">
                            <span style="font-size: 1rem; font-weight: 700; color: {'#b91c1c' if elevated else '#15803d'};">🧠 Stroke rate within this cluster: {stroke_rate:.2f}%</span><br>
                            <span style="font-size: 0.82rem; color: #64748b;">(compared to dataset average: {baseline_pct:.2f}%)</span>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_right:
                        subcol1, subcol2 = st.columns(2)
                        with subcol1:
                            st.plotly_chart(_comparison_radar(raw_patient, pred, baseline), use_container_width=True, key=f"radar_{name}")
                        with subcol2:
                            st.plotly_chart(_risk_gauge(stroke_rate, baseline_pct), use_container_width=True, key=f"gauge_{name}")

    st.markdown("#### 🔎 Contributing Factors")
    st.caption("These are the clinical characteristics that most influence which risk group this patient was placed in.")

    reasons = _explain_patient_factors(raw_patient, baseline)
    if not reasons:
        st.markdown('<div class="insight-card ok">✅ No individual factor stands out — this patient\'s profile is close to the dataset average across all measured dimensions.</div>', unsafe_allow_html=True)
    else:
        for r in reasons:
            card_type = "warn"
            icon = "🩺"
            text_lower = r.lower()
            if "age" in text_lower:
                icon = "👴" if "older" in text_lower else "🧒"
                card_type = "warn" if "older" in text_lower else "ok"
            elif "glucose" in text_lower:
                icon = "🍬" if "higher" in text_lower else "✅"
                card_type = "warn" if "higher" in text_lower else "ok"
            elif "bmi" in text_lower:
                icon = "⚖️" if "higher" in text_lower else "✅"
                card_type = "warn" if "higher" in text_lower else "ok"
            elif "hypertension" in text_lower:
                icon = "💉"
                card_type = "warn"
            elif "heart disease" in text_lower:
                icon = "❤️‍🩹"
                card_type = "warn"
            elif "smoking" in text_lower:
                icon = "🚬"
                card_type = "warn"

            clean_r = r.replace("**", "")
            st.markdown(f"""
            <div class="insight-card {card_type}">
                <span style="font-size:1.2rem;">{icon}</span>&ensp;{clean_r}
            </div>
            """, unsafe_allow_html=True)

    with st.expander("📋 Patient Input Summary"):
        summary_df = pd.DataFrame({
            "Field": ["Age", "Average Glucose Level", "BMI", "Hypertension", "Heart Disease",
                       "Marital Status", "Smoking Status"],
            "Value": [f"{age:.0f} years", f"{avg_glucose:.0f} mg/dL", f"{bmi:.1f}",
                       hypertension, heart_disease, ever_married, smoking],
            "Dataset Average": [f"{baseline['age']:.0f} years", f"{baseline['avg_glucose_level']:.0f} mg/dL",
                                f"{baseline['bmi']:.1f}", f"{baseline['hypertension_rate']:.0f}% have it",
                                f"{baseline['heart_disease_rate']:.0f}% have it", "—", "—"],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <b>Important:</b> This assessment is generated by AI clustering models trained on historical patient data.
        It does <b>not</b> constitute a medical diagnosis or clinical recommendation.
        The stroke rate shown is the <b>historical incidence</b> within the patient's matched group in the training dataset —
        it is not a personalised probability.
        Always consult a qualified healthcare professional for clinical decisions.
    </div>
    """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Technical / EDA / Preprocessing views (unchanged content, relocated under
# "Technical Details" so first-time, non-technical users aren't shown them
# by default).
# ────────────────────────────────────────────────────────────────────────────

def show_eda(data: pd.DataFrame) -> None:
    st.header("📊 Description and Analysis of Dataset")

    summary = get_clinical_summary(data)

    st.subheader("Dataset Shape")
    st.text(f"Dataset shape: {data.shape[0]} rows, {data.shape[1]} columns")

    st.subheader("Dataset Structure")
    quality_df = get_data_quality_report(data)
    structure_report = quality_df[["Column Name", "Data Type", "Non-Null Count"]]
    st.dataframe(structure_report, use_container_width=True, hide_index=True)

    st.subheader("Check Missing Values")
    missing_report = quality_df[["Column Name", "Non-Null Count", "Missing Values"]]
    st.dataframe(missing_report, use_container_width=True, hide_index=True)

    st.subheader("Distribution of Target Variable (`stroke`)")
    st.plotly_chart(plot_target_distribution(data), use_container_width=True)

    st.subheader("Distribution of Numerical Attributes")
    st.plotly_chart(plot_numerical_distributions_grid(data), use_container_width=True)

    st.subheader("Outcome-Based Distribution Comparison")
    num_feature = st.selectbox(
        "Select Numerical Feature to View Density",
        ["age", "avg_glucose_level", "bmi"],
        format_func=lambda x: {"age": "Age (Years)", "avg_glucose_level": "Average Glucose Level (mg/dL)", "bmi": "Body Mass Index (BMI)"}[x]
    )
    st.plotly_chart(plot_continuous_distribution(data, num_feature), use_container_width=True)

    st.subheader("Distribution of Categorical Attributes")
    st.plotly_chart(plot_categorical_distributions_grid(data), use_container_width=True)

    st.subheader("Outlier Detection")
    st.plotly_chart(plot_numerical_outliers(data), use_container_width=True)

    outlier_rows = []
    for col in ["age", "avg_glucose_level", "bmi"]:
        col_name_readable = {
            "age": "Age",
            "avg_glucose_level": "Average Glucose Level",
            "bmi": "BMI"
        }[col]
        values = data[col].dropna()
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        num_outliers = int(((values < lower_bound) | (values > upper_bound)).sum())
        pct_outliers = float((num_outliers / len(data)) * 100)

        outlier_rows.append({
            "Attribute": col_name_readable,
            "First Quartile": q1,
            "Third Quartile": q3,
            "Lower Boundary": lower_bound,
            "Upper Boundary": upper_bound,
            "Number of Outliers": num_outliers,
            "Percentage of outliers": pct_outliers
        })
    outlier_df = pd.DataFrame(outlier_rows)

    st.write("#### Outlier Summary Table")
    st.dataframe(
        outlier_df.style.format({
            "First Quartile": "{:.2f}",
            "Third Quartile": "{:.2f}",
            "Lower Boundary": "{:.2f}",
            "Upper Boundary": "{:.2f}",
            "Percentage of outliers": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Correlation among Numeric and Binary Attributes")
    corr_matrix, readable_labels = get_correlation_matrix(data)
    st.plotly_chart(plot_correlation_heatmap(corr_matrix, readable_labels), use_container_width=True)

    st.subheader("Categorical Feature Association (Chi-Square Test)")
    st.markdown("This test proves which lifestyle/demographic features have a statistically significant relationship with stroke risk, justifying which columns we keep or drop for clustering.")
    st.dataframe(test_categorical_association(data), use_container_width=True, hide_index=True)

    st.subheader("Confounding Variable Test (Likelihood-Ratio)")
    st.markdown("This test checks if a feature actually causes strokes, or if it is just a side-effect of getting older. If the feature does not add predictive value beyond age, it can safely be dropped from the clustering model.")
    st.dataframe(test_confounding_with_age(data), use_container_width=True, hide_index=True)


def show_preprocessing_pca(result, data: pd.DataFrame, pca_variance: float) -> None:
    st.header("⚙️ Data Preprocessing")

    st.subheader("Feature Scaling Analysis (Before vs. After)")
    raw_stats = []
    prep_df = pd.DataFrame(result.preprocessed_data, columns=result.preprocessed_feature_names)

    for col in ["age", "avg_glucose_level", "bmi"]:
        raw_vals = data[col].dropna()
        prep_col = f"numeric__{col}"
        if prep_col in prep_df.columns:
            scaled_vals = prep_df[prep_col]
        else:
            matching_cols = [c for c in prep_df.columns if col in c]
            scaled_vals = prep_df[matching_cols[0]] if matching_cols else pd.Series()

        col_readable = {
            "age": "Age (Years)",
            "avg_glucose_level": "Average Glucose Level (mg/dL)",
            "bmi": "Body Mass Index (BMI)"
        }[col]

        raw_stats.append({
            "Column Name": col_readable,
            "Data Type": "Numeric (Continuous)",
            "Scaling Method": "Robust Scaling (Median/IQR)",
            "Before (Mean ± SD)": f"{raw_vals.mean():.2f} ± {raw_vals.std():.2f}",
            "Before (Min / Max)": f"{raw_vals.min():.2f} / {raw_vals.max():.2f}",
            "After (Mean ± SD)": f"{scaled_vals.mean():.2f} ± {scaled_vals.std():.2f}",
            "After (Min / Max)": f"{scaled_vals.min():.2f} / {scaled_vals.max():.2f}"
        })

    for col in ["hypertension", "heart_disease"]:
        raw_vals = data[col].dropna()
        prep_col = f"binary__{col}"
        if prep_col in prep_df.columns:
            scaled_vals = prep_df[prep_col]
        else:
            matching_cols = [c for c in prep_df.columns if col in c]
            scaled_vals = prep_df[matching_cols[0]] if matching_cols else pd.Series()

        col_readable = col.replace('_', ' ').title()

        raw_stats.append({
            "Column Name": col_readable,
            "Data Type": "Binary (Indicator)",
            "Scaling Method": "Standard Scaling (Mean/SD)",
            "Before (Mean ± SD)": f"{raw_vals.mean():.2f} ± {raw_vals.std():.2f}",
            "Before (Min / Max)": f"{raw_vals.min():.2f} / {raw_vals.max():.2f}",
            "After (Mean ± SD)": f"{scaled_vals.mean():.2f} ± {scaled_vals.std():.2f}",
            "After (Min / Max)": f"{scaled_vals.min():.2f} / {scaled_vals.max():.2f}"
        })

    st.dataframe(pd.DataFrame(raw_stats), use_container_width=True, hide_index=True)

    st.write("#### Visualizing Scaling Effect (Distribution Comparison)")
    st.markdown("Select a feature to see how scaling centers the data at 0 and normalizes its range:")
    selected_scale_feature = st.selectbox(
        "Select Continuous Feature to inspect Preprocessing scaling effect:",
        ["age", "avg_glucose_level", "bmi"],
        format_func=lambda x: {"age": "Age (Years)", "avg_glucose_level": "Average Glucose Level (mg/dL)", "bmi": "Body Mass Index (BMI)"}[x]
    )
    st.plotly_chart(
        plot_scaling_comparison(data, result.preprocessed_data, result.preprocessed_feature_names, selected_scale_feature),
        use_container_width=True
    )
    st.write("")

    st.subheader("Categorical Feature Encoding")
    cat_stats = []
    all_cats = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    surviving_cats = [col for col in all_cats if any(c.startswith(col) for c in prep_df.columns)]
    for col in surviving_cats:
        raw_classes = sorted(data[col].dropna().unique().tolist())
        raw_classes_cleaned = [c.replace("_", " ").title() for c in raw_classes]

        generated_cols = [c for c in prep_df.columns if c.startswith(f"{col}_") or c == col]

        col_readable = col.replace('_', ' ').title()

        cat_stats.append({
            "Column Name": col_readable,
            "Data Type": "Categorical",
            "Encoding Method": "One-Hot Encoding",
            "Unique Classes (Before)": ", ".join(raw_classes_cleaned),
            "Generated Encoded Columns (After)": ", ".join(generated_cols)
        })

    st.dataframe(pd.DataFrame(cat_stats), use_container_width=True, hide_index=True)

    st.write("#### Visualizing One-Hot Encoding Effect (Binary Matrix Grid)")
    st.markdown("Select a categorical feature to see how patient categories map to model column bits:")
    selected_encode_feature = st.selectbox(
        "Select Categorical Feature to inspect encoding map:",
        surviving_cats,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    st.plotly_chart(
        plot_encoding_comparison(data, result.preprocessed_data, result.preprocessed_feature_names, selected_encode_feature),
        use_container_width=True
    )
    st.write("")

    st.subheader("PCA Explained Variance & Components Retention")
    st.markdown("PCA project features onto orthogonal components. Let's see how much variance is explained by each component.")

    var_df = get_pca_scree_data(result)
    st.plotly_chart(plot_pca_scree(var_df, pca_variance, result.n_components), use_container_width=True)

    st.subheader("Interpret Principal Dimensions via Feature Weights")
    st.markdown("Analyze how much weight each raw medical feature has on all computed PC components at once:")
    st.plotly_chart(plot_pca_loadings_heatmap(result.pca_selected_loadings), use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ────────────────────────────────────────────────────────────────────────────

def _user_metric(value, decimals=2):
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def _best_algorithm(results_dict: dict):
    valid = {
        name: res for name, res in results_dict.items()
        if getattr(res, "silhouette", np.nan) is not None
        and np.isfinite(getattr(res, "silhouette", np.nan))
    }
    if not valid:
        return next(iter(results_dict))
    return max(valid, key=lambda name: valid[name].silhouette)


def _friendly_algorithm_name(name: str) -> str:
    return {
        "K-Means": "Patient Grouping (K-Means)",
        "DBSCAN": "Density-Based Grouping (DBSCAN)",
        "MeanShift": "Similarity-Based Grouping (MeanShift)",
    }.get(name, name)


def _render_simple_result_header(result, algorithm: str) -> None:
    clean = result.data[result.data["cluster"] != -1]
    stroke_rates = clean.groupby("cluster")["stroke"].mean() if not clean.empty else pd.Series(dtype=float)
    max_rate = float(stroke_rates.max()) if not stroke_rates.empty else 0.0

    st.markdown("### 🎯 AI Analysis Result")
    st.caption(_friendly_algorithm_name(algorithm))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patient Groups", int(result.n_clusters))
    c2.metric("Patients Analysed", f"{len(result.data):,}")
    c3.metric("Group Quality", _user_metric(getattr(result, "silhouette", np.nan), 3))
    c4.metric("Highest Stroke Rate", f"{max_rate:.1%}")

    if result.n_clusters >= 2 and np.isfinite(getattr(result, "silhouette", np.nan)):
        score = result.silhouette
        if score >= 0.50:
            msg = "The groups are reasonably well separated."
        elif score >= 0.25:
            msg = "The groups show some meaningful separation, although they overlap."
        else:
            msg = "The groups overlap considerably, so the clusters should be interpreted carefully."
        st.info(f"📖 **How to read this:** A higher Group Quality (Silhouette Score) generally means clearer separation between groups. {msg}")


def _render_user_friendly_groups(result) -> None:
    """Colour-coded, plain-English group cards (higher-contrast version)."""
    summary = result.cluster_summary.copy()
    summary = summary[summary["cluster"] != -1].copy()

    st.markdown("### 👥 Patient Groups Found")
    st.caption(
        "The AI sorted patients into these groups based on shared characteristics like age, glucose level, and BMI. "
        "Patients in the same group are more similar to each other than to patients in other groups."
    )

    if summary.empty:
        st.warning("No patient groups were found with the current settings. Try choosing a different method from the sidebar.")
        return

    baseline_rate = float(result.data["stroke"].mean())
    cols = st.columns(min(3, len(summary)))
    for i, (_, row) in enumerate(summary.iterrows()):
        with cols[i % len(cols)]:
            cluster_id = int(row["cluster"])
            patients = int(row["patients"])
            stroke_rate = float(row["stroke_rate"])
            elevated = bool(row.get("elevated_risk", False))

            if elevated:
                border_color = "#e53e3e"
                risk_badge = "🔴 Higher Risk"
                risk_color = "#e53e3e"
                tip = f"This group has a stroke rate of {stroke_rate:.1%}, which is above the dataset average ({baseline_rate:.1%})."
            else:
                border_color = "#38a169"
                risk_badge = "🟢 Typical Risk"
                risk_color = "#38a169"
                tip = f"This group has a stroke rate of {stroke_rate:.1%}, which is around or below the dataset average ({baseline_rate:.1%})."

            st.markdown(
                f"""
                <div style="background:#fff; border:2px solid {border_color}; border-radius:14px;
                            padding:18px; margin-bottom:8px; box-shadow:0 2px 8px rgba(0,0,0,0.06);">
                    <div style="font-size:1.1rem; font-weight:700; color:#1e3c72;">Group {cluster_id}</div>
                    <div style="font-size:0.85rem; font-weight:600; color:{risk_color}; margin-top:4px;">{risk_badge}</div>
                    <div style="font-size:1.6rem; font-weight:700; color:#0f172a; margin-top:10px;">{patients:,}</div>
                    <div style="font-size:0.8rem; color:#64748b;">patients in this group</div>
                    <div style="margin-top:10px; font-size:0.9rem; color:#334155;">Stroke rate: <b>{stroke_rate:.1%}</b></div>
                    <div style="font-size:0.78rem; color:#64748b; margin-top:4px;">{tip}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("")
    with st.expander("❓ What does 'stroke rate' mean?"):
        st.markdown(
            "The **stroke rate** is the percentage of patients in that group who had a stroke recorded in our dataset.\n\n"
            f"The **overall dataset average** is **{baseline_rate:.1%}**.\n\n"
            "Groups labelled 🔴 **Higher Risk** have a stroke rate above average — the AI flagged these patients "
            "as sharing characteristics that are more common among stroke patients.\n\n"
            "⚠️ **This is a research tool, not a medical diagnosis.** Always consult a qualified healthcare professional."
        )


def _render_user_clustering(result, algorithm: str) -> None:
    _render_simple_result_header(result, algorithm)
    _render_user_friendly_groups(result)

    st.markdown("### 🗺️ Visual Map of Patient Groups")
    st.caption("Every patient is represented as a single dot on this map. The AI calculates who is similar based on their age, glucose level, and BMI, and places similar patients close to each other. The colors show the groups the AI found.")
    if algorithm == "MeanShift":
        st.plotly_chart(plot_meanshift_cluster_scatter(result), use_container_width=True)
    else:
        st.plotly_chart(plot_cluster_scatter(result), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("### 📊 Stroke Distribution")
        st.plotly_chart(plot_stroke_by_cluster(result), use_container_width=True)
    with right:
        st.markdown("### 📏 Group Size")
        st.plotly_chart(plot_cluster_size_distribution(result), use_container_width=True)

    with st.expander("🔬 Technical details"):
        if np.isfinite(getattr(result, "silhouette", np.nan)):
            st.write(f"**Silhouette Score:** {result.silhouette:.4f} — higher is generally better.")
        if np.isfinite(getattr(result, "davies_bouldin", np.nan)):
            st.write(f"**Davies-Bouldin Index:** {result.davies_bouldin:.4f} — lower is generally better.")
        st.write(f"**Algorithm:** {algorithm}")
        st.write(f"**PCA components:** {getattr(result, 'n_components', 'N/A')}")
        if algorithm == "DBSCAN":
            st.write(f"**Selected EPS:** {getattr(result, 'selected_eps', 'N/A')}")
        elif algorithm == "MeanShift":
            st.write(f"**Selected bandwidth:** {getattr(result, 'selected_eps', 'N/A')}")
        elif algorithm == "K-Means":
            st.write(f"**K:** {getattr(result, 'n_clusters', 'N/A')}")


def _render_compare_page(results_dict: dict) -> None:
    """Colour-coded, plain-English comparison of the three AI methods."""
    st.markdown("# 📊 Which AI Method Works Best?")
    st.markdown(
        "We ran **three different AI methods** on the same patient data. "
        "This page shows you how well each one grouped the patients. "
        "You don't need to understand the maths — just look at the coloured score cards below."
    )

    best_algo = _best_algorithm(results_dict)

    st.markdown("### 🏅 Score Summary")
    st.caption(
        "**Group Quality** → the higher the better (max 1.0). "
        "**Group Overlap** → the lower the better (min 0.0). "
        "The 🏆 badge marks the method with the best Group Quality."
    )

    score_cols = st.columns(len(results_dict))
    for col, (name, res) in zip(score_cols, results_dict.items()):
        sil_val = getattr(res, "silhouette", np.nan)
        db_val = getattr(res, "davies_bouldin", np.nan)
        is_best = (name == best_algo)

        border = "3px solid #1e3c72" if is_best else "1px solid #e2e8f0"
        badge = " 🏆" if is_best else ""

        if np.isfinite(sil_val):
            if sil_val >= 0.5:
                sil_color, sil_tip = "#38a169", "Good separation"
            elif sil_val >= 0.25:
                sil_color, sil_tip = "#d69e2e", "Moderate separation"
            else:
                sil_color, sil_tip = "#e53e3e", "Groups overlap a lot"
            sil_display = f"{sil_val:.3f}"
        else:
            sil_color, sil_tip, sil_display = "#94a3b8", "Not available", "N/A"

        if np.isfinite(db_val):
            if db_val <= 1.0:
                db_color, db_tip = "#38a169", "Groups are well separated"
            elif db_val <= 2.0:
                db_color, db_tip = "#d69e2e", "Some overlap between groups"
            else:
                db_color, db_tip = "#e53e3e", "Groups are hard to tell apart"
            db_display = f"{db_val:.3f}"
        else:
            db_color, db_tip, db_display = "#94a3b8", "Not available", "N/A"

        col.markdown(
            f"""
            <div style="border:{border}; border-radius:14px; padding:18px 14px; background:#fff;
                        box-shadow:0 2px 12px rgba(0,0,0,0.06); text-align:center;">
                <div style="font-size:1.15rem; font-weight:700; color:#163b70;">{name}{badge}</div>
                <div style="margin-top:14px;">
                    <div style="font-size:0.72rem; color:#64748b; text-transform:uppercase; font-weight:600;
                                letter-spacing:0.4px;">Group Quality ↑ Higher is better</div>
                    <div style="font-size:2rem; font-weight:800; color:{sil_color};">{sil_display}</div>
                    <div style="font-size:0.75rem; color:{sil_color};">{sil_tip}</div>
                </div>
                <div style="margin-top:14px;">
                    <div style="font-size:0.72rem; color:#64748b; text-transform:uppercase; font-weight:600;
                                letter-spacing:0.4px;">Group Overlap ↓ Lower is better</div>
                    <div style="font-size:2rem; font-weight:800; color:{db_color};">{db_display}</div>
                    <div style="font-size:0.75rem; color:{db_color};">{db_tip}</div>
                </div>
                <div style="margin-top:14px; font-size:0.8rem; color:#64748b;">
                    Groups found: <b>{res.n_clusters}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    st.markdown("### 📈 Visual Comparison")
    algos, sil, db = [], [], []
    for name, res in results_dict.items():
        if np.isfinite(getattr(res, "silhouette", np.nan)) and np.isfinite(getattr(res, "davies_bouldin", np.nan)):
            algos.append(name)
            sil.append(float(res.silhouette))
            db.append(float(res.davies_bouldin))

    if algos:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Group Quality  (↑ Higher is Better)",
            x=algos, y=sil,
            text=[f"{v:.3f}" for v in sil], textposition="outside",
            marker_color="#3b82f6"
        ))
        fig.add_trace(go.Bar(
            name="Group Overlap  (↓ Lower is Better)",
            x=algos, y=db,
            text=[f"{v:.3f}" for v in db], textposition="outside",
            marker_color="#f97316"
        ))
        fig.update_layout(
            barmode="group",
            yaxis_title="Score",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=20, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("❓ What do these scores mean in plain English?"):
        st.markdown("""
**Think of it like sorting a bag of sweets into piles by colour:**

🔵 **Group Quality (Silhouette Score — higher is better)**
If each pile is perfectly one colour only, the Group Quality is 1.0 (perfect).
If the piles are a jumbled mix of colours, the score drops toward 0.
→ You want this **as high as possible**.

🟠 **Group Overlap (Davies-Bouldin Index — lower is better)**
If the piles are spread out and clearly separate from each other, the overlap score is close to 0.
If the piles are squashed together and hard to tell apart, the score rises.
→ You want this **as low as possible**.

⬜ **Unusual Patients (only in DBSCAN)**
DBSCAN is strict — it refuses to force patients into groups if they don't clearly fit.
These patients are labelled "unusual". This keeps the other groups cleaner and more meaningful.
        """)

    with st.expander("🔢 See full numbers table"):
        comp_df = generate_algorithm_comparison(results_dict)
        display = comp_df.rename(columns={
            "Algorithm": "AI Method",
            "Clusters Found": "Groups Found",
            "Noise Ratio": "Unusual Patients %",
            "Silhouette Score": "Group Quality  (↑ Higher is Better)",
            "Davies-Bouldin Index": "Group Overlap  (↓ Lower is Better)",
            "Max Cluster Stroke Rate": "Highest Stroke Rate in Any Group",
            "Baseline Stroke Rate": "Overall Average Stroke Rate"
        })
        st.dataframe(display, use_container_width=True, hide_index=True)


def _render_technical_page(data: pd.DataFrame, results_dict: dict, result, pca_variance: float) -> None:
    st.markdown("### ⚙️ Technical Details")
    st.caption("This section is intended for students, tutors and users who want to inspect the AI process.")
    t1, t2, t3, t4 = st.tabs(["📊 EDA", "🧹 Preprocessing", "🧠 PCA & Features", "🔬 Algorithm Deep-Dive"])
    with t1:
        show_eda(data)
    with t2:
        show_preprocessing_pca(result, result.data, pca_variance)
    with t3:
        pca_importance, cluster_deviations = calculate_feature_contributions(result)
        st.subheader("Feature Contribution")
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(pca_importance, use_container_width=True, hide_index=True)
        with c2:
            st.dataframe(cluster_deviations, use_container_width=True, hide_index=True)
    with t4:
        st.caption("Full parameter-search tables and detailed plots for each clustering algorithm.")
        deep_dive_algo = st.selectbox("Choose an algorithm to inspect:", list(results_dict.keys()), key="deep_dive_algo")
        deep_result = results_dict[deep_dive_algo]
        if deep_dive_algo == "K-Means":
            show_kmeans_clustering(deep_result)
        elif deep_dive_algo == "DBSCAN":
            show_dbscan_clustering(deep_result)
        elif deep_dive_algo == "MeanShift":
            show_meanshift_clustering(deep_result)


# ────────────────────────────────────────────────────────────────────────────
# Main app
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if "cache_cleaned" not in st.session_state:
        st.cache_data.clear()
        st.session_state["cache_cleaned"] = True

    inject_custom_css()

    try:
        data = load_data(str(DATA_PATH))
    except (FileNotFoundError, ValueError) as exc:
        logger.exception("Failed to load dataset")
        st.error(
            "⚠️ Could not load the patient dataset. "
            f"Details: {exc}\n\nPlease confirm 'brain_stroke.csv' is present next to app.py and reload the page."
        )
        st.stop()

    # ── Sidebar: branding ────────────────────────────────────────────────
    st.sidebar.markdown("# 🧠 Patient Explorer")
    st.sidebar.caption("Simple AI-assisted patient clustering")

    # ── Sidebar: navigation (Explore Patient is the landing page) ──────────
    st.sidebar.markdown("### 🗺️ Navigation")
    PAGE_OPTIONS = ["🔎 Explore Patient", "📁 Batch Analysis", "👥 Patient Groups", "📊 Compare AI Methods", "⚙️ Technical Details"]
    page = st.sidebar.radio("Navigation", PAGE_OPTIONS, index=0, label_visibility="collapsed")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 AI Grouping Method")
    METHOD_OPTIONS = [
        "🌟 AI Recommended (Auto)",
        "🎯 K-Means  (Fixed number of groups)",
        "🔍 DBSCAN  (Natural density groups)",
        "🌊 MeanShift  (Similarity-based groups)"
    ]
    selected_method = st.sidebar.selectbox(
        "Choose grouping method:",
        options=METHOD_OPTIONS,
        index=0,
        help="Not sure? Leave on AI Recommended — it automatically picks the best method. "
             "This choice affects the 'Patient Groups' and 'Technical Details' pages; "
             "'Explore Patient' always checks all three methods for you."
    )

    show_advanced = st.sidebar.checkbox(
        "🔧 Show technical settings", value=False,
        help="Reveal extra sliders (PCA variance retained, elevated-risk multiplier). "
             "Most users can safely leave this off."
    )

    # Default settings (used unless the user picks a method + advanced options)
    km_k, km_max_k, km_pca, km_risk = None, 15, 0.90, 1.5
    db_eps, db_min_samples, db_pca, db_risk = None, 30, 0.90, 1.5
    ms_bw, ms_quantile, ms_mbf, ms_pca, ms_risk = None, 0.25, 5, 0.90, 1.5

    if "K-Means" in selected_method:
        with st.sidebar.expander("⚙️ K-Means Settings", expanded=True):
            automatic_k = st.checkbox("Let AI decide number of groups", value=True, key="km_auto_k")
            if not automatic_k:
                km_k = st.slider("Number of groups", min_value=2, max_value=10, value=3, key="km_k_val",
                                 help="How many distinct patient groups to create.")
            km_max_k = st.slider("Max groups to test", min_value=3, max_value=20, value=15, key="km_max_k_val",
                                 help="The AI will test up to this many groups and pick the best one.")
            if show_advanced:
                km_pca = st.slider("PCA variance retained", 0.60, 1.00, 0.90, 0.05, key="km_pca_adv")
                km_risk = st.slider("Elevated-risk multiplier", 1.0, 3.0, 1.5, 0.1, key="km_risk_adv")

    elif "DBSCAN" in selected_method:
        with st.sidebar.expander("⚙️ DBSCAN Settings", expanded=True):
            automatic_eps = st.checkbox("Let AI choose radius automatically", value=True, key="db_auto_eps")
            if not automatic_eps:
                db_eps = st.number_input("Neighborhood radius", min_value=0.01, value=0.76, step=0.05,
                                         format="%.2f", key="db_eps_val",
                                         help="How close patients need to be to belong to the same group. Lower = smaller, tighter groups.")
            db_min_samples = st.slider("Minimum patients per group", min_value=3, max_value=40, value=30,
                                       key="db_min_samples_val",
                                       help="A group needs at least this many patients to be considered real.")
            if show_advanced:
                db_pca = st.slider("PCA variance retained", 0.60, 1.00, 0.90, 0.05, key="db_pca_adv")
                db_risk = st.slider("Elevated-risk multiplier", 1.0, 3.0, 1.5, 0.1, key="db_risk_adv")

    elif "MeanShift" in selected_method:
        with st.sidebar.expander("⚙️ MeanShift Settings", expanded=True):
            automatic_bw = st.checkbox("Let AI choose bandwidth automatically", value=True, key="ms_auto_bw")
            if not automatic_bw:
                ms_bw = st.number_input("Bandwidth", min_value=0.01, value=1.0, step=0.05,
                                        format="%.3f", key="ms_bw_val",
                                        help="Controls how broadly patients are compared. Smaller = more groups.")
            ms_quantile = st.slider("Grouping sensitivity", min_value=0.05, max_value=0.60, value=0.25, step=0.05,
                                    key="ms_quantile_val",
                                    help="Lower value = more sensitive = more groups found.")
            if show_advanced:
                ms_pca = st.slider("PCA variance retained", 0.60, 1.00, 0.90, 0.05, key="ms_pca_adv")
                ms_risk = st.slider("Elevated-risk multiplier", 1.0, 3.0, 1.5, 0.1, key="ms_risk_adv")

    elif show_advanced:
        with st.sidebar.expander("⚙️ Advanced (applies to all methods)", expanded=True):
            shared_pca = st.slider("PCA variance retained", 0.60, 1.00, 0.90, 0.05, key="shared_pca_adv")
            shared_risk = st.slider("Elevated-risk multiplier", 1.0, 3.0, 1.5, 0.1, key="shared_risk_adv")
            km_pca = db_pca = ms_pca = shared_pca
            km_risk = db_risk = ms_risk = shared_risk

    # ── Run all three algorithms (cached — no repeated work) ──────────────
    results_dict = {}
    if KMEANS_AVAILABLE:
        km_result, km_artifacts = analyse_kmeans(data, km_k, km_pca, km_risk, km_max_k)
        results_dict["K-Means"] = km_result
    else:
        km_artifacts = None

    db_result, db_artifacts = analyse_dbscan(data, db_eps, db_min_samples, db_pca, db_risk)
    results_dict["DBSCAN"] = db_result

    ms_result, ms_artifacts = analyse_meanshift(data, ms_bw, ms_quantile, ms_pca, ms_risk, ms_mbf)
    results_dict["MeanShift"] = ms_result

    best_algorithm = _best_algorithm(results_dict)

    if "AI Recommended" in selected_method:
        chosen_algo = best_algorithm
    elif "K-Means" in selected_method:
        chosen_algo = "K-Means"
    elif "DBSCAN" in selected_method:
        chosen_algo = "DBSCAN"
    else:
        chosen_algo = "MeanShift"

    result = results_dict[chosen_algo]
    artifacts_map = {"K-Means": km_artifacts, "DBSCAN": db_artifacts, "MeanShift": ms_artifacts}
    chosen_pca_variance = {"K-Means": km_pca, "DBSCAN": db_pca, "MeanShift": ms_pca}[chosen_algo]

    # ── Page routing ────────────────────────────────────────────────────
    if page == "🔎 Explore Patient":
        st.markdown("# 🔎 Find a Patient's Group")
        st.info(
            "👤 Enter a patient's details below. The AI checks **all three** clustering methods and shows you "
            "where they agree. \n\n⚠️ **This is not a medical diagnosis.** This tool is for research and exploration only."
        )
        _render_patient_predict_form(data, results_dict, artifacts_map)

    elif page == "📁 Batch Analysis":
        st.markdown("# 📁 Batch Patient Analysis")
        st.info(
            "⚠️ **This is not a medical diagnosis.** Upload a CSV of multiple patients to classify all of them at once."
        )
        _render_batch_analysis(data, results_dict, artifacts_map)

    elif page == "👥 Patient Groups":
        st.markdown("# 🧠 Patient Group Explorer")
        if "AI Recommended" in selected_method:
            sil_score = getattr(result, 'silhouette', np.nan)
            sil_text = f" — Group Quality score: **{sil_score:.3f}**" if np.isfinite(sil_score) else ""
            st.success(
                f"✅ The AI automatically selected **{chosen_algo}** as the best method for this dataset{sil_text}. "
                f"You can change the method in the sidebar if you want to explore manually."
            )
        else:
            st.info(f"You are viewing results from: **{chosen_algo}**. Switch to *AI Recommended* in the sidebar to let the AI decide.")
        _render_user_clustering(result, chosen_algo)

    elif page == "📊 Compare AI Methods":
        _render_compare_page(results_dict)

    else:
        _render_technical_page(data, results_dict, result, chosen_pca_variance)

    st.markdown(
        '<div class="app-footer">🧠 Stroke Patient Clustering Explorer — a research and education tool, not a medical device.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
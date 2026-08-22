
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

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

DATA_PATH = Path(__file__).with_name("brain_stroke.csv")

st.set_page_config(page_title="Stroke Patient Clustering Explorer", page_icon="🧠", layout="wide")


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return load_dataset(path)


@st.cache_resource(show_spinner="Running DBSCAN analysis...")
def analyse_dbscan(data: pd.DataFrame, eps: float | None, min_samples: int, pca_variance: float, risk_multiplier: float):
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
        return run_kmeans_with_artifacts(data, KMeansConfig(
            n_clusters=n_clusters,
            pca_variance=pca_variance,
            risk_multiplier=risk_multiplier,
            max_k=max_k,
        ))

def dbscan_controls() -> tuple[float | None, int, float, float]:
    st.sidebar.subheader("DBSCAN settings")
    automatic_eps = st.sidebar.checkbox("Choose EPS automatically", value=False)
    eps = None
    if not automatic_eps:
        eps = st.sidebar.number_input("EPS", min_value=0.01, value=0.76, step=0.05, format="%.2f")
    
    # Ensure min_samples defaults to 20
    min_samples = st.sidebar.slider("min_samples", min_value=3, max_value=40, value=30)
    
    # Ensure PCA defaults to 0.90
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

def inject_custom_css() -> None:
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        /* Apply font family globally */
        .stApp, html, body, [class*="css"] {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        /* Customize headings */
        h1, h2, h3 {
            font-weight: 700 !important;
            color: #1e293b !important;
            margin-top: 1rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }
        
        /* Tab formatting */
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

        /* Metric styling */
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
        
        /* Metric container border/shadow */
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
        
        /* Make code blocks clean */
        code {
            color: #0f172a !important;
            background-color: #f1f5f9 !important;
        }
        
        /* Info boxes */
        .stAlert {
            border-radius: 8px !important;
        }
        
        .premium-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        }
        </style>
    """, unsafe_allow_html=True)


def show_dbscan_clustering(result) -> None:
    st.header("🔍 DBSCAN Clustering")

    # Calculate supplemental metrics for comparison and render in a custom HTML flexbox stats bar
    clean_df = result.data[result.data["cluster"] != -1]
    stroke_rates = clean_df.groupby("cluster")["stroke"].mean()
    max_stroke_rate = stroke_rates.max() if not stroke_rates.empty else 0.0
    baseline_stroke_rate = result.data["stroke"].mean()

    st.markdown(f"""
    <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px;">
        <!-- Card 1 -->
        <div style="flex: 1; min-width: 120px; background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); text-align: center;">
            <div style="font-size: 0.68rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; white-space: nowrap;">Clusters Found</div>
            <div style="font-size: 1.15rem; color: #000000; font-weight: 700; margin-top: 3px;">{result.n_clusters}</div>
        </div>
        <!-- Card 2 -->
        <div style="flex: 1; min-width: 120px; background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); text-align: center;">
            <div style="font-size: 0.68rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; white-space: nowrap;">Noise Patients</div>
            <div style="font-size: 1.15rem; color: #C0392B; font-weight: 700; margin-top: 3px;">{result.noise_ratio:.2%}</div>
        </div>
        <!-- Card 3 -->
        <div style="flex: 1; min-width: 120px; background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); text-align: center;">
            <div style="font-size: 0.68rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; white-space: nowrap;">Selected EPS</div>
            <div style="font-size: 1.15rem; color: #000000; font-weight: 700; margin-top: 3px;">{result.selected_eps:.3f}</div>
        </div>
        <!-- Card 4 -->
        <div style="flex: 1; min-width: 120px; background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); text-align: center;">
            <div style="font-size: 0.68rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; white-space: nowrap;">Silhouette Score</div>
            <div style="font-size: 1.15rem; color: #000000; font-weight: 700; margin-top: 3px;">{"N/A" if np.isnan(result.silhouette) else f"{result.silhouette:.4f}"}</div>
        </div>
        <!-- Card 5 -->
        <div style="flex: 1; min-width: 120px; background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); text-align: center;">
            <div style="font-size: 0.68rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; white-space: nowrap;">Davies-Bouldin</div>
            <div style="font-size: 1.15rem; color: #000000; font-weight: 700; margin-top: 3px;">{"N/A" if np.isnan(result.davies_bouldin) else f"{result.davies_bouldin:.4f}"}</div>
        </div>
        <!-- Card 6 -->
        <div style="flex: 1; min-width: 120px; background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); text-align: center;">
            <div style="font-size: 0.68rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; white-space: nowrap;">Max stroke rate</div>
            <div style="font-size: 1.15rem; color: #000000; font-weight: 700; margin-top: 3px;">{max_stroke_rate:.2%}</div>
        </div>
        <!-- Card 7 -->
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
        fig_scatter = plot_cluster_scatter(result)
        st.plotly_chart(fig_scatter, use_container_width=True)
    with right:
        figure = plot_k_distance(result)
        st.plotly_chart(figure, use_container_width=True)

    st.subheader("EPS Parameter Search")
    fig_search = plot_eps_search(result)
    st.plotly_chart(fig_search, use_container_width=True)
    
    search_table = result.parameter_results.copy()
    search_table["noise_percent"] = search_table["noise_ratio"] * 100
    st.dataframe(search_table, use_container_width=True, hide_index=True)

    st.subheader("Stroke Risk Class Distribution")
    chart = plot_stroke_by_cluster(result)
    st.plotly_chart(chart, use_container_width=True)
    
    rates = result.cluster_summary.copy()
    
    # Map elevated_risk boolean to Yes/No
    rates["elevated_risk"] = rates["elevated_risk"].map({True: "Yes", False: "No"})
    
    # Assign the raw stroke_rate fraction ratio to age_mean
    rates["age_mean"] = rates["stroke_rate"]
    
    # String format all numbers before transposing for clean layout alignment
    rates["cluster"] = rates["cluster"].astype(int)
    rates["patients"] = rates["patients"].astype(int)
    rates["stroke_cases"] = rates["stroke_cases"].astype(int)
    rates["age_mean"] = rates["age_mean"].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
    rates["glucose_mean"] = rates["glucose_mean"].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
    rates["bmi_mean"] = rates["bmi_mean"].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
    rates["stroke_rate_pct"] = rates["stroke_rate_pct"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
    
    # Rename columns to match user structure
    rates = rates.rename(columns={
        "cluster": "Cluster",
        "patients": "Patients",
        "stroke_cases": "stroke_cases",
        "age_mean": "age_mean",
        "glucose_mean": "glucose_mean",
        "bmi_mean": "Bmi_mean",
        "stroke_rate_pct": "stroke_rate_pct",
        "elevated_risk": "elevated_risk"
    })
    
    # Select and order columns
    rates_display = rates[[
        "Cluster", "Patients", "stroke_cases", "age_mean", 
        "glucose_mean", "Bmi_mean", "stroke_rate_pct", "elevated_risk"
    ]]
    
    # Transpose the dataframe by setting Cluster as the index first, avoiding any duplicate "Cluster" row in the table body
    cluster_ids = rates_display["Cluster"].tolist()
    transposed = rates_display.astype(str).set_index("Cluster").T.reset_index()
    transposed.columns = ["Attribute"] + [f"Cluster {int(c)}" for c in cluster_ids]
    
    st.dataframe(transposed, use_container_width=True, hide_index=True)


def _render_cluster_summary_table(result, csv_filename: str) -> None:
    rates = result.cluster_summary.copy()
    rates["elevated_risk"] = rates["elevated_risk"].map({True: "Yes", False: "No"})
    rates["age_mean"] = rates["stroke_rate"]
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

    # ── Key metrics row ──────────────────────────────────────────────────────
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
            <div style="font-size: 0.68rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; white-space: nowrap;">Selected Bandwidth</div>
            <div style="font-size: 1.15rem; color: #000000; font-weight: 700; margin-top: 3px;">{f"{result.selected_eps:.4f}" if result.selected_eps is not None else "N/A"}</div>
        </div>
        <div style="flex: 1; min-width: 120px; background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); text-align: center;">
            <div style="font-size: 0.68rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; white-space: nowrap;">PCA Components</div>
            <div style="font-size: 1.15rem; color: #000000; font-weight: 700; margin-top: 3px;">{result.n_components}</div>
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
            <div style="font-size: 0.68rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; white-space: nowrap;">Max Stroke Rate</div>
            <div style="font-size: 1.15rem; color: #000000; font-weight: 700; margin-top: 3px;">{max_stroke_rate:.2%}</div>
        </div>
        <div style="flex: 1; min-width: 120px; background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); text-align: center;">
            <div style="font-size: 0.68rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; white-space: nowrap;">Baseline Stroke</div>
            <div style="font-size: 1.15rem; color: #000000; font-weight: 700; margin-top: 3px;">{baseline_stroke_rate:.2%}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if result.n_clusters < 2:
        st.warning("Fewer than two clusters found. Try a lower quantile value or reduce the bandwidth manually.")

    # ── Scatter + Bandwidth sweep ────────────────────────────────────────────
    left, right = st.columns(2)
    with left:
        st.plotly_chart(plot_meanshift_cluster_scatter(result), use_container_width=True)
    with right:
        st.plotly_chart(plot_bandwidth_sweep(result), use_container_width=True)

    # ── Bandwidth sweep table ────────────────────────────────────────────────
    st.subheader("Bandwidth Parameter Search")
    st.info(
        "📐 Each row shows quality metrics for a candidate bandwidth value. "
        "The selected bandwidth (red dashed line above) maximises Silhouette score "
        "while minimising Davies-Bouldin index."
    )
    sweep_table = result.parameter_results.copy()
    sweep_table["noise_percent"] = sweep_table["noise_ratio"] * 100
    st.dataframe(sweep_table, use_container_width=True, hide_index=True)

    # ── Stroke risk + size distribution ─────────────────────────────────────
    st.subheader("Stroke Risk Class Distribution")
    left2, right2 = st.columns(2)
    with left2:
        st.plotly_chart(plot_stroke_by_cluster(result), use_container_width=True)
    with right2:
        st.plotly_chart(plot_cluster_size_distribution(result), use_container_width=True)

    # ── Clinical profile radar ───────────────────────────────────────────────
    st.subheader("Clinical Risk Factor Profile per Cluster")
    st.info(
        "🕸️ Each axis of the radar is min-max normalised across clusters so "
        "profiles can be compared on a common [0, 1] scale. "
        "A cluster that is large on all axes represents a high-risk phenotype."
    )
    st.plotly_chart(plot_cluster_profile_radar(result), use_container_width=True)

    # ── Feature contribution analysis ────────────────────────────────────────
    pca_importance, cluster_deviations = calculate_feature_contributions(result)
    with st.expander("📊 Attribute Contribution Analysis"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("##### Overall Feature Importance (PCA Weight)")
            st.dataframe(pca_importance, use_container_width=True, hide_index=True)
        with c2:
            st.write("##### Cluster Mean Deviation from Baseline (%)")
            st.dataframe(cluster_deviations, use_container_width=True)

    # ── Summary table + download ─────────────────────────────────────────────
    st.subheader("Cluster Summary Table")
    _render_cluster_summary_table(result, "meanshift_patient_clusters.csv")


def _explain_patient_factors(raw_patient: dict, baseline: dict) -> list[str]:
    reasons: list[tuple[float, str]] = []  # (magnitude, sentence) for sorting

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


def _render_patient_predict_form(form_key: str, predict_fn, subheader: str) -> None:
    st.subheader(f"🩺 {subheader}")
    st.caption("Enter patient details to see which risk group they match.")

    with st.form(form_key):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0.0, max_value=120.0, value=50.0, step=1.0, key=f"{form_key}_age")
            avg_glucose = st.number_input(
                "Avg Glucose Level (mg/dL)", min_value=0.0, max_value=500.0, value=100.0, step=1.0, key=f"{form_key}_glucose"
            )
            bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, step=0.1, key=f"{form_key}_bmi")
        with col2:
            hypertension = st.selectbox("Hypertension", ["No", "Yes"], key=f"{form_key}_hypertension")
            heart_disease = st.selectbox("Heart Disease", ["No", "Yes"], key=f"{form_key}_heart_disease")
            ever_married = st.selectbox("Ever Married", ["Yes", "No"], key=f"{form_key}_ever_married")
            smoking = st.selectbox(
                "Smoking Status",
                ["never smoked", "formerly smoked", "smokes", "Unknown"],
                key=f"{form_key}_smoking",
            )

        submitted = st.form_submit_button("Predict Cluster", use_container_width=True)

    if submitted:
        raw_patient = {
            "age": age,
            "avg_glucose_level": avg_glucose,
            "bmi": bmi,
            "hypertension": 1 if hypertension == "Yes" else 0,
            "heart_disease": 1 if heart_disease == "Yes" else 0,
            "ever_married": ever_married,
            "smoking_status": smoking,
        }

        pred = predict_fn(raw_patient)
        baseline_rate = st.session_state.get("_baseline_stroke_rate_pct")
        baseline_stats = st.session_state.get("_baseline_stats")

        if pred["predicted_cluster"] == -1:
            st.warning("⚠️ No clear match found for this combination of inputs.")
            return

        if pred["cluster_elevated_risk"]:
            icon, label, color = "🔴", "Higher Risk", "#C0392B"
        else:
            icon, label, color = "🟢", "Typical Risk", "#1E8449"

        times_avg = pred['cluster_stroke_rate_pct'] / baseline_rate if baseline_rate else None
        reasons = _explain_patient_factors(raw_patient, baseline_stats) if baseline_stats else []

        reasons_html = "".join(f"<li>{r}</li>" for r in reasons) or "<li>No single factor stands out — it's the overall combination.</li>"

        st.markdown(f"""
        <div style="background:#fff; border:1px solid #e2e8f0; border-left:6px solid {color};
                    border-radius:10px; padding:18px 22px; margin:10px 0 16px 0;">
            <div style="font-size:1.3rem; font-weight:700; color:{color};">{icon} {label}</div>
            <div style="font-size:1.0rem; color:#334155; margin-top:4px;">
                Stroke rate in this group: <b>{pred['cluster_stroke_rate_pct']:.1f}%</b>
                {f"(dataset average: {baseline_rate:.1f}%, ~{times_avg:.1f}x)" if times_avg else ""}
            </div>
            <div style="font-size:0.95rem; font-weight:600; color:#334155; margin-top:14px;">Why:</div>
            <ul style="margin:6px 0 0 0; padding-left:20px; color:#334155;">
                {reasons_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)


def show_meanshift_predictor(artifacts) -> None:
    _render_patient_predict_form(
        form_key="meanshift_predict_form",
        predict_fn=lambda raw: predict_meanshift_patient(raw, artifacts),
        subheader="Predict New Patient's Risk Group (MeanShift)",
    )


def show_dbscan_predictor(artifacts) -> None:
    _render_patient_predict_form(
        form_key="dbscan_predict_form",
        predict_fn=lambda raw: predict_dbscan_patient(raw, artifacts),
        subheader="Predict New Patient's Risk Group (DBSCAN)",
    )


def show_kmeans_predictor(artifacts) -> None:
    _render_patient_predict_form(
        form_key="kmeans_predict_form",
        predict_fn=lambda raw: predict_kmeans_patient(raw, artifacts),
        subheader="Predict New Patient's Risk Group (K-Means)",
    )

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
        import plotly.express as px
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
    _render_cluster_summary_table(result, "kmeans_patient_clusters.csv")

def show_eda(data: pd.DataFrame) -> None:
    st.header("📊 Description and Analysis of Dataset")
    
    summary = get_clinical_summary(data)
    
    # Dataset Shape
    st.subheader("Dataset Shape")
    st.text(f"Dataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Dataset Structure
    st.subheader("Dataset Structure")
    quality_df = get_data_quality_report(data)
    structure_report = quality_df[["Column Name", "Data Type", "Non-Null Count"]]
    st.dataframe(structure_report, use_container_width=True, hide_index=True)
    
    # Check Missing Values
    st.subheader("Check Missing Values")
    missing_report = quality_df[["Column Name", "Non-Null Count", "Missing Values"]]
    st.dataframe(missing_report, use_container_width=True, hide_index=True)
    
    # Distribution of Target Variable
    st.subheader("Distribution of Target Variable (`stroke`)")
    fig_target = plot_target_distribution(data)
    st.plotly_chart(fig_target, use_container_width=True)
        
    # Distribution of Numerical Attributes
    st.subheader("Distribution of Numerical Attributes")
    fig_grid = plot_numerical_distributions_grid(data)
    st.plotly_chart(fig_grid, use_container_width=True)
    
    st.subheader("Outcome-Based Distribution Comparison")
    num_feature = st.selectbox(
        "Select Numerical Feature to View Density", 
        ["age", "avg_glucose_level", "bmi"], 
        format_func=lambda x: {"age": "Age (Years)", "avg_glucose_level": "Average Glucose Level (mg/dL)", "bmi": "Body Mass Index (BMI)"}[x]
    )
    fig_dist = plot_continuous_distribution(data, num_feature)
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Distribution of Categorical Attributes
    st.subheader("Distribution of Categorical Attributes")
    fig_cat_grid = plot_categorical_distributions_grid(data)
    st.plotly_chart(fig_cat_grid, use_container_width=True)
        
    # Outlier Detection
    st.subheader("Outlier Detection")
    fig_outliers = plot_numerical_outliers(data)
    st.plotly_chart(fig_outliers, use_container_width=True)
    
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
    
    # Correlation among numeric and binary attributes
    st.subheader("Correlation among Numeric and Binary Attributes")
    corr_matrix, readable_labels = get_correlation_matrix(data)
    fig_corr = plot_correlation_heatmap(corr_matrix, readable_labels)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Categorical Feature Association (Chi-Square Test)")
    st.markdown("This test proves which lifestyle/demographic features have a statistically significant relationship with stroke risk, justifying which columns we keep or drop for clustering.")
    chi2_df = test_categorical_association(data)
    st.dataframe(chi2_df, use_container_width=True, hide_index=True)

    st.subheader("Confounding Variable Test (Likelihood-Ratio)")
    st.markdown("This test checks if a feature actually causes strokes, or if it is just a side-effect of getting older. If the feature does not add predictive value beyond age, it can safely be dropped from the clustering model.")
    lr_df = test_confounding_with_age(data)
    st.dataframe(lr_df, use_container_width=True, hide_index=True)


def show_preprocessing_pca(result, data: pd.DataFrame, pca_variance: float) -> None:
    st.header("⚙️ Data Preprocessing")
    
    # Table 1: Feature Scaling Analysis
    st.subheader("Feature Scaling Analysis (Before vs. After)")
    raw_stats = []
    prep_df = pd.DataFrame(result.preprocessed_data, columns=result.preprocessed_feature_names)
    
    # Continuous Features
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
        
    # Binary Features
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
        
    scaling_comparison_df = pd.DataFrame(raw_stats)
    st.dataframe(scaling_comparison_df, use_container_width=True, hide_index=True)
    
    st.write("#### Visualizing Scaling Effect (Distribution Comparison)")
    st.markdown("Select a feature to see how scaling centers the data at 0 and normalizes its range:")
    selected_scale_feature = st.selectbox(
        "Select Continuous Feature to inspect Preprocessing scaling effect:",
        ["age", "avg_glucose_level", "bmi"],
        format_func=lambda x: {"age": "Age (Years)", "avg_glucose_level": "Average Glucose Level (mg/dL)", "bmi": "Body Mass Index (BMI)"}[x]
    )
    fig_scale_comp = plot_scaling_comparison(data, result.preprocessed_data, result.preprocessed_feature_names, selected_scale_feature)
    st.plotly_chart(fig_scale_comp, use_container_width=True)
    st.write("")
    
    # Table 2: Categorical Feature Encoding
    st.subheader("Categorical Feature Encoding")
    cat_stats = []
    all_cats = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    surviving_cats = [col for col in all_cats if any(c.startswith(col) for c in prep_df.columns)]
    for col in surviving_cats:
        raw_classes = sorted(data[col].dropna().unique().tolist())
        raw_classes_cleaned = [c.replace("_", " ").title() for c in raw_classes]
        
        # Match generated columns in clean feature list
        generated_cols = [c for c in prep_df.columns if c.startswith(f"{col}_") or c == col]
        generated_cols_cleaned = generated_cols
        
        col_readable = col.replace('_', ' ').title()
        
        cat_stats.append({
            "Column Name": col_readable,
            "Data Type": "Categorical",
            "Encoding Method": "One-Hot Encoding",
            "Unique Classes (Before)": ", ".join(raw_classes_cleaned),
            "Generated Encoded Columns (After)": ", ".join(generated_cols_cleaned)
        })
        
    encoding_df = pd.DataFrame(cat_stats)
    st.dataframe(encoding_df, use_container_width=True, hide_index=True)
    
    st.write("#### Visualizing One-Hot Encoding Effect (Binary Matrix Grid)")
    st.markdown("Select a categorical feature to see how patient categories map to model column bits:")
    selected_encode_feature = st.selectbox(
        "Select Categorical Feature to inspect encoding map:",
        surviving_cats,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    fig_encode_comp = plot_encoding_comparison(data, result.preprocessed_data, result.preprocessed_feature_names, selected_encode_feature)
    st.plotly_chart(fig_encode_comp, use_container_width=True)
    st.write("")
    

    # PCA Scree Plot
    st.subheader("PCA Explained Variance & Components Retention")
    st.markdown("PCA project features onto orthogonal components. Let's see how much variance is explained by each component.")
    
    var_df = get_pca_scree_data(result)
    fig_scree = plot_pca_scree(var_df, pca_variance, result.n_components)
    st.plotly_chart(fig_scree, use_container_width=True)
    
    # PCA feature loadings/interpretations
    st.subheader("Interpret Principal Dimensions via Feature Weights")
    st.markdown("Analyze how much weight each raw medical feature has on all computed PC components at once:")
    
    fig_loadings_heatmap = plot_pca_loadings_heatmap(result.pca_selected_loadings)
    st.plotly_chart(fig_loadings_heatmap, use_container_width=True)

def main() -> None:
    if "cache_cleaned" not in st.session_state:
        st.cache_data.clear()
        st.session_state["cache_cleaned"] = True

    st.sidebar.title("Controls")
    _algo_options = ["DBSCAN", "MeanShift"] + (["K-Means"] if KMEANS_AVAILABLE else [])
    selected = st.sidebar.selectbox("Algorithm", _algo_options)
    data = load_data(str(DATA_PATH))
    st.session_state["_baseline_stroke_rate_pct"] = float(data["stroke"].mean() * 100)
    st.session_state["_baseline_stats"] = {
        "age": float(data["age"].mean()),
        "avg_glucose_level": float(data["avg_glucose_level"].mean()),
        "bmi": float(data["bmi"].mean()),
        "hypertension_rate": float(data["hypertension"].mean() * 100),
        "heart_disease_rate": float(data["heart_disease"].mean() * 100),
    }

    inject_custom_css()

    # ── Sidebar controls & run the selected algorithm ─────────────────────
    meanshift_artifacts = None
    dbscan_artifacts = None
    kmeans_artifacts = None
    if selected == "DBSCAN":
        eps, min_samples, pca_variance, risk_multiplier = dbscan_controls()
        result, dbscan_artifacts = analyse_dbscan(data, eps, min_samples, pca_variance, risk_multiplier)
    elif selected == "MeanShift":
        bandwidth, quantile, pca_variance, risk_multiplier, min_bin_freq = meanshift_controls()
        result, meanshift_artifacts = analyse_meanshift(data, bandwidth, quantile, pca_variance, risk_multiplier, min_bin_freq)
    elif selected == "K-Means" and KMEANS_AVAILABLE:
        n_clusters, pca_variance, risk_multiplier, max_k = kmeans_controls()
        result, kmeans_artifacts = analyse_kmeans(data, n_clusters, pca_variance, risk_multiplier, max_k)
    else:
        # Defensive fallback — should not be reached with the guarded dropdown
        pca_variance = 0.90
        result, dbscan_artifacts = analyse_dbscan(data, None, 12, pca_variance, 1.5)

    tab_clustering, tab_comparison, tab_eda, tab_preprocessing = st.tabs([
        "🔍 Clustering Dashboard", 
        "⚔️ Algorithm Comparison",
        "📊 Exploratory Data Analysis (EDA)", 
        "⚙️ Data Preprocessing & PCA"
    ])
    
    with tab_clustering:
        if selected == "DBSCAN":
            show_dbscan_clustering(result)

            # Display Feature Contribution Insights
            pca_importance, cluster_deviations = calculate_feature_contributions(result)
            with st.expander("📊 Attribute Contribution Analysis"):
                c1, c2 = st.columns(2)
                with c1:
                    st.write("##### Overall Feature Importance (PCA Weight)")
                    st.dataframe(pca_importance, use_container_width=True, hide_index=True)
                with c2:
                    st.write("##### Cluster Mean Deviation from Baseline (%)")
                    st.dataframe(cluster_deviations, use_container_width=True)

            st.divider()
            show_dbscan_predictor(dbscan_artifacts)

        elif selected == "MeanShift":
            show_meanshift_clustering(result)
            st.divider()
            show_meanshift_predictor(meanshift_artifacts)

        else:
            show_kmeans_clustering(result)
            if KMEANS_AVAILABLE and kmeans_artifacts is not None:
                st.divider()
                show_kmeans_predictor(kmeans_artifacts)


    with tab_comparison:
        st.header("⚔️ Clustering Algorithm Comparison")

        results_dict: dict = {}
        _dbscan_result, _ = analyse_dbscan(data, 0.76, 30, 0.90, 1.5)
        results_dict["DBSCAN"] = _dbscan_result
        _ms_result, _ = analyse_meanshift(data, None, 0.25, 0.90, 1.5, 5)
        results_dict["MeanShift"] = _ms_result
        if KMEANS_AVAILABLE:
            _km_result, _ = analyse_kmeans(data, None, 0.90, 1.5, 15)
            results_dict["K-Means"] = _km_result

        comp_df = generate_algorithm_comparison(results_dict)
        algorithm_names = comp_df["Algorithm"].tolist()

        comp_transposed = comp_df.astype(str).set_index("Algorithm").T.reset_index()
        comp_transposed.columns = ["Comparison Metric"] + algorithm_names

        st.dataframe(comp_transposed, use_container_width=True, hide_index=True)

        # ── Grouped bar chart: Silhouette vs Davies-Bouldin ───────────────────
        import plotly.graph_objects as go

        _chart_algos, _chart_sil, _chart_db = [], [], []
        for _algo, _res in results_dict.items():
            if np.isfinite(_res.silhouette) and np.isfinite(_res.davies_bouldin):
                _chart_algos.append(_algo)
                _chart_sil.append(round(_res.silhouette, 4))
                _chart_db.append(round(_res.davies_bouldin, 4))

        if _chart_algos:
            st.subheader("Silhouette Score vs Davies-Bouldin Index")
            _fig_cmp = go.Figure()
            _fig_cmp.add_trace(go.Bar(
                name="Silhouette Score (↑ better)",
                x=_chart_algos,
                y=_chart_sil,
                marker_color="#1e3c72",
                text=[f"{v:.4f}" for v in _chart_sil],
                textposition="outside",
            ))
            _fig_cmp.add_trace(go.Bar(
                name="Davies-Bouldin Index (↓ better)",
                x=_chart_algos,
                y=_chart_db,
                marker_color="#e74c3c",
                text=[f"{v:.4f}" for v in _chart_db],
                textposition="outside",
            ))
            _fig_cmp.update_layout(
                barmode="group",
                xaxis_title="Algorithm",
                yaxis_title="Score",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=60, b=40),
                font=dict(family="Outfit, sans-serif"),
            )
            _fig_cmp.update_xaxes(showgrid=False)
            _fig_cmp.update_yaxes(gridcolor="#e2e8f0")
            st.plotly_chart(_fig_cmp, use_container_width=True)

        if not KMEANS_AVAILABLE:
            st.info(
                "K-Means results will appear here once your teammate's "
                "kmeans_stroke.py is available."
            )
            
    with tab_eda:
        show_eda(data)
        
    with tab_preprocessing:
        show_preprocessing_pca(result, result.data, pca_variance)


if __name__ == "__main__":
    main()
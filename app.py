"""Interactive Stroke Patient Clustering Explorer.

Focuses solely on page configuration, user state selections, custom layout cards,
and rendering visual modules. Business logic is placed in data_processing.py,
evaluation/metrics structures are in evaluation.py, and plotting calculations 
are in visualizations.py.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from dbscan_stroke import DBSCANConfig, run_dbscan
from meanshift_stroke import MeanShiftConfig, run_meanshift
from kmeans_stroke import KMeansConfig, run_kmeans
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


@st.cache_data(show_spinner="Running DBSCAN analysis...")
def analyse_dbscan(data: pd.DataFrame, eps: float | None, min_samples: int, pca_variance: float, risk_multiplier: float):
    return run_dbscan(data, DBSCANConfig(
        eps=eps,
        min_samples=min_samples,
        pca_variance=pca_variance,
        risk_multiplier=risk_multiplier,
    ))


@st.cache_data(show_spinner="Running MeanShift analysis...")
def analyse_meanshift(
    data: pd.DataFrame,
    bandwidth: float | None,
    quantile: float,
    pca_variance: float,
    risk_multiplier: float,
    min_bin_freq: int,
):
    return run_meanshift(data, MeanShiftConfig(
        bandwidth=bandwidth,
        quantile=quantile,
        pca_variance=pca_variance,
        risk_multiplier=risk_multiplier,
        min_bin_freq=min_bin_freq,
    ))

@st.cache_data(show_spinner="Running K-Means analysis...")
def analyse_kmeans(data: pd.DataFrame, n_clusters: int | None, pca_variance: float, risk_multiplier: float, max_k: int):
    return run_kmeans(data, KMeansConfig(
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
        # Change the default value to 0.66 (or 0.658 if you change the format to "%.3f")
        eps = st.sidebar.number_input("EPS", min_value=0.01, value=0.66, step=0.05, format="%.2f")
    
    # Ensure min_samples defaults to 20
    min_samples = st.sidebar.slider("min_samples", min_value=3, max_value=40, value=20)
    
    # Ensure PCA defaults to 0.90
    pca_variance = st.sidebar.slider("PCA variance retained", min_value=0.60, max_value=1.00, value=0.90, step=0.05)
    risk_multiplier = st.sidebar.slider("Elevated-risk multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
    with st.sidebar.expander("How to tune DBSCAN"):
        st.markdown("""
- **EPS** is the neighbourhood radius. Lower EPS makes tighter, smaller clusters and more noise. Higher EPS merges nearby groups; too high can create one large cluster.
- **min_samples** is the number of neighbours required for a dense core. Higher values make DBSCAN stricter, usually increasing noise and removing tiny clusters.
- Start with automatic EPS, inspect the k-distance graph, then test nearby manual values. Prefer a useful number of clusters, modest noise, and a higher silhouette score—not a particular cluster count.
- **-1 means noise**, not a third medical class. It is okay to have more than two clusters: stroke 0/1 is an outcome used after clustering, while clusters describe different patient profiles.
        """)
    return eps, min_samples, pca_variance, risk_multiplier


def meanshift_controls() -> tuple[float | None, float, float, float, int]:
    """Render MeanShift sidebar controls and return selected parameter values."""
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
        "Maximum K to evaluate", min_value=3, max_value=15, value=10, key="km_max_k",
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
    st.download_button("Download clustered patient data", result.data.to_csv(index=False).encode("utf-8"), "dbscan_patient_clusters.csv", "text/csv")


def _render_cluster_summary_table(result, csv_filename: str) -> None:
    """Shared helper: render a transposed cluster summary table + download button."""
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
    st.download_button(
        "Download clustered patient data",
        result.data.to_csv(index=False).encode("utf-8"),
        csv_filename, "text/csv"
    )


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
    
    st.info("""
        💡 **Data Pipeline Execution Note**:
        * **Robustness**: Any extreme outliers identified in **3.2.7** are treated during preprocessing using robust scaling techniques (`RobustScaler` median/IQR) rather than being dropped, which keeps all clinical cohorts intact.
        * **Missing Values**: Imputed automatically to prevent clustering algorithm disruption.
    """)


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
    
    # Interpretations guideline text
    st.info("""
        💡 **Clinical PCA Dimension Interpretation Guideline**:
        * **Diverging Colormap (Red-to-Blue)**: Blue cells indicate strong positive feature associations; Red cells indicate strong negative associations.
        * **Eigenvector Loading Scores**: Values range from -1.0 to +1.0. A magnitude close to 1.0 (or cell values |x| > 0.4) indicates that the feature is a core driver of that specific Principal Component's direction (clinical subtype definition).
    """)


def main() -> None:
    if "cache_cleaned" not in st.session_state:
        st.cache_data.clear()
        st.session_state["cache_cleaned"] = True

    st.sidebar.title("Controls")
    selected = st.sidebar.selectbox("Algorithm", ["DBSCAN", "MeanShift", "K-Means"])
    data = load_data(str(DATA_PATH))

    inject_custom_css()

    # ── Sidebar controls & run the selected algorithm ─────────────────────
    if selected == "DBSCAN":
        eps, min_samples, pca_variance, risk_multiplier = dbscan_controls()
        result = analyse_dbscan(data, eps, min_samples, pca_variance, risk_multiplier)
    elif selected == "MeanShift":
        bandwidth, quantile, pca_variance, risk_multiplier, min_bin_freq = meanshift_controls()
        result = analyse_meanshift(data, bandwidth, quantile, pca_variance, risk_multiplier, min_bin_freq)
    else:
        n_clusters, pca_variance, risk_multiplier, max_k = kmeans_controls()
        result = analyse_kmeans(data, n_clusters, pca_variance, risk_multiplier, max_k)

        # K-Means placeholder – keeps shared tabs working with DBSCAN result
        #st.sidebar.subheader("PCA settings")
        #pca_variance = st.sidebar.slider("PCA variance retained", min_value=0.60, max_value=1.00, value=0.90, step=0.05)
        #result = analyse_dbscan(data, None, 12, pca_variance, 1.5)

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

        elif selected == "MeanShift":
            show_meanshift_clustering(result)

        else:
            show_kmeans_clustering(result)


    with tab_comparison:
        st.header("⚔️ Clustering Algorithm Comparison")
        st.markdown("Compare performance indices and clinical outcomes across medical patient clustering algorithms:")

        # Build results dict with every algorithm that has been run
        results_dict = {selected: result}

        comp_df = generate_algorithm_comparison(results_dict)
        algorithm_names = comp_df["Algorithm"].tolist()

        comp_transposed = comp_df.astype(str).set_index("Algorithm").T.reset_index()
        comp_transposed.columns = ["Comparison Metric"] + algorithm_names
        
        st.dataframe(comp_transposed, use_container_width=True, hide_index=True)

        st.info("""
            📝 **Teammate Integration Guide**:
            * To compare K-Means or MeanShift runs, instantiate their results via your teammate's adapters (using the same `ClusteringResult` interface) and append them to the `results_dict` inside `app.py`.
            * The comparison table will automatically calculate and display the new models' clusters count, noise rates, silhouette ratios, and maximum group stroke rates side-by-side.
        """)
            
    with tab_eda:
        show_eda(data)
        
    with tab_preprocessing:
        show_preprocessing_pca(result, result.data, pca_variance)


if __name__ == "__main__":
    main()

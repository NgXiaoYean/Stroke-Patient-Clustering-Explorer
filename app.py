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
from data_processing import (
    load_dataset,
    get_clinical_summary,
    get_categorical_analysis,
    get_correlation_matrix,
    get_preprocessing_previews,
    get_pca_scree_data,
    get_pca_loadings,
    get_data_quality_report
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
    plot_scaling_comparison
)

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


def dbscan_controls() -> tuple[float | None, int, float, float]:
    st.sidebar.subheader("DBSCAN settings")
    automatic_eps = st.sidebar.checkbox("Choose EPS automatically", value=True)
    eps = None
    if not automatic_eps:
        eps = st.sidebar.number_input("EPS", min_value=0.01, value=1.0, step=0.05, format="%.2f")
    min_samples = st.sidebar.slider("min_samples", min_value=3, max_value=40, value=12)
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
    st.header("🔍 DBSCAN Clustering Analysis")
    st.caption("Clusters are formed without using the stroke label; stroke rates are shown afterwards for interpretation.")

    first, second, third, fourth = st.columns(4)
    first.metric("Clusters found", result.n_clusters)
    second.metric("Noise patients", f"{result.noise_ratio:.1%}")
    third.metric("Selected EPS", f"{result.selected_eps:.3f}", help=f"Robust suggestion: {result.suggested_eps:.3f}")
    fourth.metric("Silhouette", "N/A" if np.isnan(result.silhouette) else f"{result.silhouette:.3f}", help="Higher is better; compare settings only on the same data.")

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
    st.dataframe(rates, use_container_width=True, hide_index=True)
    st.download_button("Download clustered patient data", result.data.to_csv(index=False).encode("utf-8"), "dbscan_patient_clusters.csv", "text/csv")


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
    st.write("")
    
    # Table 2: Categorical Feature Encoding
    st.subheader("Categorical Feature Encoding (One-Hot Encoding)")
    cat_stats = []
    for col in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]:
        raw_classes = sorted(data[col].dropna().unique().tolist())
        raw_classes_cleaned = [c.replace("_", " ").title() for c in raw_classes]
        
        # Match generated columns in clean feature list
        generated_cols = [c for c in prep_df.columns if c.startswith(f"{col}_") or c == col]
        generated_cols_cleaned = [c.replace(f"{col}_", "") for c in generated_cols]
        
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
    
    raw_preview, prep_preview = get_preprocessing_previews(result, data)
    left_c, right_c = st.columns(2)
    with left_c:
        st.write("#### Original Features (Preview Raw Input)")
        st.dataframe(raw_preview, use_container_width=True)
        
    with right_c:
        st.write("#### Preprocessed Numerical representation (Preview Pipeline Output)")
        st.dataframe(prep_preview, use_container_width=True)
        st.caption("Showing first 7 column indexes of the scale-standardized feature matrix.")
        
    # PCA Scree Plot
    st.subheader("PCA Explained Variance & Components Retention")
    st.markdown("PCA project features onto orthogonal components. Let's see how much variance is explained by each component.")
    
    var_df = get_pca_scree_data(result)
    fig_scree = plot_pca_scree(var_df, pca_variance, result.n_components)
    st.plotly_chart(fig_scree, use_container_width=True)
    
    # PCA feature loadings/interpretations
    st.subheader("Interpret Principal Dimensions via Feature Weights")
    st.markdown("Analyze how much weight each raw medical feature has on the primary computed PC components.")
    
    pc_list = [f"PC{i+1}" for i in range(result.n_components)]
    selected_pc = st.selectbox("Select Principal Component to Inspect", pc_list)
    
    loadings, top_pos_features, top_neg_features = get_pca_loadings(result, selected_pc)
    fig_loadings = plot_pca_loadings_bar(loadings, selected_pc)
    st.plotly_chart(fig_loadings, use_container_width=True)
    
    # Interpretations text
    st.info(f"""
        💡 **Clinical Interpretation for {selected_pc}**:
        * **Strong Positive Associations**: Features like `{', '.join(top_pos_features)}` increase index scores on `{selected_pc}`.
        * **Strong Negative Associations**: Features like `{', '.join(top_neg_features)}` decrease index scores on `{selected_pc}`.
        * High loadings help explain what clinical subtype (e.g. older cardiac patient vs younger smoking patient) the dimension PC maps to.
    """)


def main() -> None:
    if "cache_cleaned" not in st.session_state:
        st.cache_data.clear()
        st.session_state["cache_cleaned"] = True
        
    st.sidebar.title("Controls")
    selected = st.sidebar.selectbox("Algorithm", ["DBSCAN", "K-Means", "MeanShift"])
    upload = st.sidebar.file_uploader("Optional CSV upload", type="csv")
    data = pd.read_csv(upload) if upload is not None else load_data(str(DATA_PATH))
    
    inject_custom_css()
    
    # Render controls and get values
    if selected == "DBSCAN":
        eps, min_samples, pca_variance, risk_multiplier = dbscan_controls()
    else:
        st.sidebar.subheader("PCA settings")
        pca_variance = st.sidebar.slider("PCA variance retained", min_value=0.60, max_value=1.00, value=0.90, step=0.05)
        eps, min_samples, risk_multiplier = None, 12, 1.5
        
    result = analyse_dbscan(data, eps, min_samples, pca_variance, risk_multiplier)
    
    tab_clustering, tab_eda, tab_preprocessing = st.tabs([
        "🔍 Clustering Dashboard", 
        "📊 Exploratory Data Analysis (EDA)", 
        "⚙️ Data Preprocessing & PCA"
    ])
    
    with tab_clustering:
        if selected == "DBSCAN":
            show_dbscan_clustering(result)
        else:
            st.title(f"{selected} Clustering Workspace")
            st.info(f"The shared UI is ready. Add your teammate's `{selected.lower().replace('-', '')}_stroke.py` adapter here using the same result interface as `dbscan_stroke.run_dbscan`.")
            st.dataframe(result.data.head(), use_container_width=True)
            
    with tab_eda:
        show_eda(data)
        
    with tab_preprocessing:
        show_preprocessing_pca(result, result.data, pca_variance)


if __name__ == "__main__":
    main()

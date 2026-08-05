from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from evaluation import ClusteringResult


def plot_cluster_scatter(result: ClusteringResult) -> go.Figure:
    """Project patient clusters onto the first 2 principal components."""
    frame = pd.DataFrame({
        "PC1": result.projection_2d[:, 0], 
        "PC2": result.projection_2d[:, 1], 
        "cluster": result.labels.astype(str)
    })
    frame["cluster"] = frame["cluster"].replace("-1", "Noise")
    fig = px.scatter(
        frame, 
        x="PC1", 
        y="PC2", 
        color="cluster", 
        opacity=.68,
        title="Patient clusters (2D PCA projection)", 
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0")
    )
    return fig


def plot_k_distance(result: ClusteringResult) -> go.Figure:
    # Line graph - distance to the kth neighbor 
    if result.k_distances is None:
        raise ValueError("DBSCAN k_distances are required for this plot.")
        
    kd = pd.DataFrame({
        "Patient rank": np.arange(1, len(result.k_distances) + 1), 
        "k-distance": result.k_distances
    })
    figure = px.line(
        kd, 
        x="Patient rank", 
        y="k-distance", 
        title=f"k-distance graph (k = {result.min_samples})"
    )
    if result.selected_eps is not None:
        figure.add_hline(
            y=result.selected_eps, 
            line_dash="dash", 
            line_color="#C0392B", 
            annotation_text=f"EPS {result.selected_eps:.3f}"
        )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0")
    )
    return figure


# DBSCAN only 
def plot_eps_search(result: ClusteringResult) -> go.Figure:
    # Analyze silhouette and noise metrics with eps param 
    if result.parameter_results is None:
        raise ValueError("DBSCAN parameter_results search table is required for this plot.")
        
    search = result.parameter_results.copy()
    search["noise_percent"] = search["noise_ratio"] * 100
    figure = px.line(
        search, 
        x="eps", 
        y="silhouette", 
        markers=True, 
        hover_data=["clusters", "noise_percent", "davies_bouldin", "valid"], 
        title="Compare automatic EPS candidates"
    )
    if result.selected_eps is not None:
        figure.add_vline(x=result.selected_eps, line_dash="dash", line_color="#C0392B")
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0")
    )
    return figure


def plot_stroke_by_cluster(result: ClusteringResult) -> go.Figure:
    # Bar chart -> compare stroke rate % vs cluster with overall dataset 
    overall_rate = result.data["stroke"].mean() * 100
    rates = result.cluster_summary.copy()
    rates["cluster_label"] = rates["cluster"].apply(lambda c: f"Cluster {c}")
    
    chart = px.bar(
        rates, 
        x="cluster_label", 
        y="stroke_rate_pct", 
        color="elevated_risk", 
        hover_data=["patients", "stroke_cases", "age_mean", "glucose_mean", "bmi_mean"], 
        title="Stroke rate by patient profile",
        color_discrete_map={False: "#356C9B", True: "#C0392B"},
        labels={"cluster_label": "Cluster Group", "stroke_rate_pct": "Stroke Rate (%)", "elevated_risk": "Elevated Risk (>1.5x Baseline)"}
    )
    chart.add_hline(
        y=overall_rate, 
        line_dash="dash", 
        line_color="#333333", 
        annotation_text=f"Baseline Rate: {overall_rate:.2f}%"
    )
    chart.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0")
    )
    return chart


def plot_continuous_distribution(data: pd.DataFrame, num_feature: str) -> go.Figure:
    # Histogram -> compare data between stroke and non-stroke pateints 
    plot_df = data.copy()
    plot_df["Stroke Status"] = plot_df["stroke"].map({0: "Healthy / No Stroke", 1: "Stroke Patient"})
    
    fig_dist = px.histogram(
        plot_df, 
        x=num_feature, 
        color="Stroke Status", 
        marginal="box", 
        barmode="overlay", 
        opacity=0.7,
        color_discrete_map={"Healthy / No Stroke": "#356C9B", "Stroke Patient": "#C0392B"},
        title=f"Distribution of {num_feature.replace('_', ' ').title()} by Stroke Outcome"
    )
    fig_dist.update_layout(
        xaxis_title=num_feature.replace('_', ' ').title(), 
        yaxis_title="Patient Frequency Count",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0")
    )
    return fig_dist


def plot_categorical_prevalence(grouped: pd.DataFrame, cat_feature: str, overall_stroke_rate: float) -> go.Figure:
    # Grouped bar chart -> categorical 
    fig_cat = px.bar(
        grouped,
        x=cat_feature,
        y="Stroke_Prevalence_Pct",
        labels={cat_feature: cat_feature.replace('_', ' ').title(), "Stroke_Prevalence_Pct": "Stroke Prevalence (%)"},
        title=f"Stroke Rates across {cat_feature.replace('_', ' ').title()}",
        color="Stroke_Prevalence_Pct",
        color_continuous_scale=px.colors.sequential.Reds
    )
    fig_cat.add_hline(
        y=overall_stroke_rate, 
        line_dash="dash", 
        line_color="#333333", 
        annotation_text=f"Baseline Rate ({overall_stroke_rate:.2f}%)"
    )
    fig_cat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0")
    )
    return fig_cat


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, readable_labels: list[str]) -> go.Figure:
    # Pearson correlation coefficient matrix heatmap 
    fig_corr = px.imshow(
        corr_matrix,
        x=readable_labels,
        y=readable_labels,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        zmin=-1,
        zmax=1,
        text_auto=".2f",
        title="Pearson Correlation Strengths"
    )
    fig_corr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig_corr


def plot_pca_scree(var_df: pd.DataFrame, pca_variance: float, n_components: int) -> go.Figure:
    # PCA Scree plot -> individual and cumulative explained variance ratios 
    fig_scree = go.Figure()
    fig_scree.add_trace(go.Bar(
        x=var_df["Component"],
        y=var_df["Individual Variance (%)"],
        name="Individual Explained Variance",
        marker_color="#356C9B",
        opacity=0.85
    ))
    fig_scree.add_trace(go.Scatter(
        x=var_df["Component"],
        y=var_df["Cumulative Variance (%)"],
        name="Cumulative Explained Variance",
        mode="lines+markers",
        line=dict(color="#C0392B", width=3),
        marker=dict(size=8, symbol="circle")
    ))
    
    cutoff_val = pca_variance * 100
    fig_scree.add_hline(
        y=cutoff_val,
        line_dash="dash",
        line_color="#10B981",
        line_width=2,
        annotation_text=f"Target variance: {cutoff_val:.0f}% (Retained {n_components} PCs)",
        annotation_position="bottom right"
    )
    
    fig_scree.update_layout(
        title="PCA Cumulative & Individual Variance Ratios",
        xaxis_title="Principal Components",
        yaxis_title="Explained Variance Percentage (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0")
    )
    return fig_scree


def plot_pca_loadings_bar(loadings: pd.DataFrame, selected_pc: str) -> go.Figure:
    # Horizontal bar chart -> original variable weights on selected PC component 
    fig_loadings = px.bar(
        loadings,
        x="Weight",
        y="Feature",
        orientation="h",
        color="Weight",
        color_continuous_scale="RdBu",
        range_color=[-1.0, 1.0],
        title=f"Feature Contribution Weights (Loadings) for {selected_pc}"
    )
    fig_loadings.update_layout(
        xaxis_title="Loading Coefficient (Weight)",
        yaxis_title="Feature Dimension",
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0")
    )
    return fig_loadings

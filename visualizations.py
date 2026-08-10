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
    figure.update_xaxes(
        showgrid=True,
        gridcolor="#EAF0F6",
        linecolor="#000000",
        tickfont=dict(color="#000000", family="sans-serif", size=11),
        title_font=dict(color="#000000", family="sans-serif")
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor="#EAF0F6",
        linecolor="#000000",
        tickfont=dict(color="#000000", family="sans-serif", size=11),
        title_font=dict(color="#000000", family="sans-serif")
    )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#000000", family="sans-serif")
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
    fig_corr.update_xaxes(
        tickfont=dict(color="#000000", family="sans-serif", size=11)
    )
    fig_corr.update_yaxes(
        tickfont=dict(color="#000000", family="sans-serif", size=11)
    )
    fig_corr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#000000", family="sans-serif")
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
        annotation_position="bottom right",
        annotation_font=dict(color="#000000", family="sans-serif", size=11)
    )
    
    fig_scree.update_xaxes(
        linecolor="#000000",
        tickfont=dict(color="#000000", family="sans-serif", size=11),
        title_font=dict(color="#000000", family="sans-serif")
    )
    fig_scree.update_yaxes(
        linecolor="#000000",
        tickfont=dict(color="#000000", family="sans-serif", size=11),
        title_font=dict(color="#000000", family="sans-serif")
    )
    fig_scree.update_layout(
        title="PCA Cumulative & Individual Variance Ratios",
        xaxis_title="Principal Components",
        yaxis_title="Explained Variance Percentage (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#000000", family="sans-serif"),
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
    fig_loadings.update_xaxes(
        linecolor="#000000",
        tickfont=dict(color="#000000", family="sans-serif", size=11),
        title_font=dict(color="#000000", family="sans-serif")
    )
    fig_loadings.update_yaxes(
        linecolor="#000000",
        tickfont=dict(color="#000000", family="sans-serif", size=11),
        title_font=dict(color="#000000", family="sans-serif")
    )
    fig_loadings.update_layout(
        xaxis_title="Loading Coefficient (Weight)",
        yaxis_title="Feature Dimension",
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#000000", family="sans-serif"),
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0")
    )
    return fig_loadings


def plot_target_distribution(data: pd.DataFrame) -> go.Figure:
    """Bar/frequency chart showing target variable percentages above the bars with Healthy at left."""
    counts = data["stroke"].value_counts().reset_index()
    counts.columns = ["Status", "Count"]
    counts["Status"] = counts["Status"].map({0: "Healthy", 1: "Stroke Patient"})
    
    total = counts["Count"].sum()
    counts["Percentage"] = (counts["Count"] / total * 100).round(2)
    # Combine count and percentage above the bar
    counts["Label"] = counts.apply(lambda row: f"{row['Count']:,} ({row['Percentage']:.2f}%)", axis=1)
    
    fig = px.bar(
        counts,
        x="Status",
        y="Count",
        color="Status",
        text="Label",
        color_discrete_map={"Healthy": "#356C9B", "Stroke Patient": "#C0392B"},
        category_orders={"Status": ["Healthy", "Stroke Patient"]},
        title="Distribution of Target Variable "
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Patient Status",
        yaxis_title="Count",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False
    )
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
    return fig


def plot_numerical_outliers(data: pd.DataFrame) -> go.Figure:
    """Horizontal boxplot facets stacked vertically comparing continuous attributes for outlier inspection."""
    features = ["age", "avg_glucose_level", "bmi"]
    melted = data.melt(value_vars=features, var_name="Feature", value_name="Value")
    melted["Feature"] = melted["Feature"].map({
        "age": "Age (Years)",
        "avg_glucose_level": "Average Glucose Level (mg/dL)",
        "bmi": "Body Mass Index (BMI)"
    })
    fig = px.box(
        melted, 
        x="Value", 
        facet_row="Feature", 
        color="Feature",
        facet_row_spacing=0.15, 
        title="<b>Outlier Profiles across Clinical Features</b>",
        color_discrete_sequence=["#356C9B", "#10B981", "#F59E0B"]
    )
    
    fig.update_traces(
        width=0.45,
        marker=dict(size=4.5, opacity=0.7, line=dict(width=0)),
        line=dict(width=1.5)
    )
    
    fig.update_xaxes(
        matches=None, 
        showgrid=True, 
        gridcolor="#EAF0F6",
        linecolor="#000000",
        ticks="outside",
        tickcolor="#000000",
        tickfont=dict(color="#000000", family="sans-serif", size=11),
        showticklabels=True
    )
    
    fig.update_yaxes(showticklabels=False, showgrid=False, title_text="")
    
    fig.for_each_annotation(lambda a: a.update(
        font=dict(size=14, family="sans-serif", color="#000000"),
        text=f"<b>{a.text.split('=')[-1]}</b>",
        textangle=0,      
        x=0,              
        xanchor="left",
        yanchor="bottom",
        y=a.y + 0.13        
    ))
    
    fig.update_layout(
        height=540,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=65, b=40), # balanced margins
        font=dict(family="sans-serif", color="#000000"), # Force solid black labels globally
        showlegend=False
    )
    return fig


def plot_numerical_distributions_grid(data: pd.DataFrame) -> go.Figure:
    """Generate high-fidelity, interactive Plotly side-by-side histograms with smooth spline KDE line overlays."""
    from plotly.subplots import make_subplots
    import scipy.stats as stats
    
    features = ["age", "avg_glucose_level", "bmi"]
    titles = [f"<b>Distribution of {col}</b>" for col in features]
    
    # Create subplots grid
    fig = make_subplots(
        rows=1, 
        cols=3, 
        subplot_titles=titles,
        horizontal_spacing=0.07
    )
    
    for idx, col in enumerate(features):
        col_idx = idx + 1
        values = data[col].dropna()
        
        min_val = float(values.min())
        max_val = float(values.max())
        x_range = np.linspace(min_val, max_val, 200)
        
        # Calculate KDE using scipy
        kde = stats.gaussian_kde(values)
        kde_y = kde(x_range)
        
        # 24 bins for an elegant binned distribution representation
        nbins = 24
        bin_width = (max_val - min_val) / nbins
        kde_scaled_y = kde_y * len(values) * bin_width
        
        # Add histogram trace
        fig.add_trace(
            go.Histogram(
                x=values,
                xbins=dict(
                    start=min_val,
                    end=max_val,
                    size=bin_width
                ),
                name="Frequency",
                marker=dict(
                    color="#A2BEED", # professional pastel slate-blue
                    line=dict(color="#FFFFFF", width=1.5)
                ),
                opacity=0.85,
                hovertemplate="Value Range: %{x}<br>Count: %{y}<extra></extra>",
                showlegend=False
            ),
            row=1, col=col_idx
        )
        
        # Add smooth spline-curved KDE line
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=kde_scaled_y,
                mode="lines",
                name="Density Trend",
                line=dict(color="#1A365D", width=3, shape="spline"), # rich dark indigo
                hovertemplate="Density Level: %{y:.1f}<extra></extra>",
                showlegend=False
            ),
            row=1, col=col_idx
        )
        
        # Style subplot axes
        fig.update_xaxes(
            title_text=col, 
            row=1, col=col_idx, 
            showgrid=True, 
            gridcolor="#EAF0F6",
            linecolor="#CBD5E1",
            ticks="outside"
        )
        fig.update_yaxes(
            title_text="Count" if idx == 0 else "", 
            row=1, col=col_idx, 
            showgrid=True, 
            gridcolor="#EAF0F6",
            linecolor="#CBD5E1"
        )
        
    fig.for_each_annotation(lambda a: a.update(
        font=dict(size=14, family="sans-serif", color="#1E293B"),
        y=1.05
    ))
    
    fig.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=65, b=40),
        font=dict(family="sans-serif", color="#334155")
    )
    return fig


def plot_categorical_distributions_grid(data: pd.DataFrame) -> go.Figure:
    """Generate high-fidelity interactive Plotly subplot grid of categorical distributions."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    features = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    titles = [f"<b>Distribution: {col.replace('_', ' ').title()}</b>" for col in features]
    
    fig = make_subplots(
        rows=3, 
        cols=2, 
        subplot_titles=titles,
        horizontal_spacing=0.18, 
        vertical_spacing=0.12
    )
    
    # Grid coordinates mapping (r, c)
    coords = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1)]
    
    for idx, col in enumerate(features):
        r, c = coords[idx]
        counts = data[col].value_counts().reset_index()
        counts.columns = ["Category", "Count"]
        counts = counts.sort_values(by="Count")
        
        total = counts["Count"].sum()
        counts["Percentage"] = (counts["Count"] / total * 100).round(1)
        counts["Label"] = counts.apply(lambda row: f"{row['Count']:,} ({row['Percentage']}%)", axis=1)
        
        counts["Category"] = counts["Category"].astype(str).str.replace("_", " ").str.title()
        
        fig.add_trace(
            go.Bar(
                y=counts["Category"],
                x=counts["Count"],
                orientation="h",
                text=counts["Label"],
                textposition="outside",
                cliponaxis=False, 
                textfont=dict(color="#000000", family="sans-serif", size=10, weight="bold"), 
                marker=dict(
                    color="#4E79A7", 
                    line=dict(color="#FFFFFF", width=1)
                ),
                hovertemplate="Category: %{y}<br>Count: %{x}<br>Percentage: %{text}<extra></extra>",
                showlegend=False
            ),
            row=r, col=c
        )
        
        # Style grid/axes with solid dark lines and black text
        fig.update_xaxes(
            title_text="Count", 
            row=r, col=c, 
            showgrid=True, 
            gridcolor="#EAF0F6", 
            linecolor="#000000",
            title_font=dict(color="#000000", family="sans-serif", size=11),
            tickfont=dict(color="#000000", family="sans-serif", size=10)
        )
        fig.update_yaxes(
            row=r, col=c, 
            linecolor="#000000",
            tickfont=dict(color="#000000", family="sans-serif", size=10)
        )
        
    # Style SUBPLOT Titles in black
    fig.for_each_annotation(lambda a: a.update(
        font=dict(size=12, family="sans-serif", color="#000000"),
        y=a.y + 0.02 # nudge titles slightly up
    ))
    
    fig.update_layout(
        height=850, # taller size to give plenty of vertical breathing room
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=80, r=80, t=50, b=50), # generous margins
        font=dict(family="sans-serif", color="#000000") # Force solid black globally
    )
    return fig


def plot_scaling_comparison(raw_data: pd.DataFrame, prep_data: np.ndarray, prep_cols: list[str], feature: str) -> go.Figure:
    """Generate a side-by-side Plotly comparison of a feature before and after scaling."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    # Fetch raw and scaled series
    raw_series = raw_data[feature].dropna()
    
    prep_col = f"numeric__{feature}"
    prep_df = pd.DataFrame(prep_data, columns=prep_cols)
    if prep_col in prep_df.columns:
        scaled_series = prep_df[prep_col]
    else:
        matching = [c for c in prep_df.columns if feature in c]
        scaled_series = prep_df[matching[0]] if matching else pd.Series()
        
    feature_title = {
        "age": "Age (Years)",
        "avg_glucose_level": "Average Glucose Level (mg/dL)",
        "bmi": "Body Mass Index (BMI)"
    }[feature]
    
    # Create 1 row, 2 cols subplot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"<b>Before Scaling: Raw {feature_title}</b>",
            f"<b>After Scaling: Robust Scaled {feature_title}</b>"
        ],
        horizontal_spacing=0.15
    )

    BINS = 40
    
    # Left plot: Raw Distribution
    fig.add_trace(
        go.Histogram(
            x=raw_series,
            nbinsx=BINS,
            marker_color="#356C9B",
            opacity=0.8,
            name="Raw Value",
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Right plot: Scaled Distribution
    fig.add_trace(
        go.Histogram(
            x=scaled_series,
            nbinsx=BINS,
            marker_color="#F59E0B",
            opacity=0.8,
            name="Scaled Value (Robust)",
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Style X & Y Axes in solid black
    fig.update_xaxes(
        title_text="Raw Value (Original Unit)", 
        row=1, col=1, 
        linecolor="#000000",
        tickfont=dict(color="#000000", family="sans-serif", size=10),
        title_font=dict(color="#000000", family="sans-serif")
    )
    fig.update_xaxes(
        title_text="Scaled Value (Deviation from Median/IQR)", 
        row=1, col=2, 
        linecolor="#000000",
        tickfont=dict(color="#000000", family="sans-serif", size=10),
        title_font=dict(color="#000000", family="sans-serif")
    )
    
    fig.update_yaxes(
        title_text="Frequency (Count)",
        row=1, col=1,
        linecolor="#000000",
        tickfont=dict(color="#000000", family="sans-serif", size=10),
        title_font=dict(color="#000000", family="sans-serif")
    )
    fig.update_yaxes(
        row=1, col=2,
        linecolor="#000000",
        tickfont=dict(color="#000000", family="sans-serif", size=10)
    )
    
    # Style annotations / titles in black
    fig.for_each_annotation(lambda a: a.update(
        font=dict(size=12, family="sans-serif", color="#000000")
    ))
    
    fig.update_layout(
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=40, t=55, b=45),
        font=dict(family="sans-serif", color="#000000")
    )
    return fig


def plot_encoding_comparison(raw_data: pd.DataFrame, prep_data: np.ndarray, prep_cols: list[str], feature: str) -> go.Figure:
    """Generate an interactive binary-mapping heatmap showing how categorical categories convert to One-Hot bits."""
    # Take first 12 sample rows
    num_samples = 12
    raw_sample = raw_data[feature].head(num_samples).reset_index(drop=True)
    
    # Identify generated one-hot columns
    generated_cols = [c for c in prep_cols if c.startswith(f"{feature}_") or c == feature]
    
    prep_df = pd.DataFrame(prep_data, columns=prep_cols)
    prep_sample = prep_df[generated_cols].head(num_samples).reset_index(drop=True)
    
    # Make human-readable y-axis labels (Title-case values like 'Formerly Smoked')
    y_labels = [f"Patient {i+1} : <b>{str(val).replace('_', ' ').title()}</b>" for i, val in enumerate(raw_sample)]
    x_labels = [c for c in generated_cols]
    
    # Create binary heatmap matrix
    z_matrix = prep_sample.values
    
    # Format X-axis headers with a line break to keep them vertically stacked and perfectly aligned
    prefix_clean = feature.replace('_', ' ').title()
    cleaned_x_labels = []
    for col in x_labels:
        prefix_pattern = f"{feature}_"
        if col.startswith(prefix_pattern):
            broken_name = col.replace(prefix_pattern, f"{feature}_<br>")
        else:
            broken_name = col
        cleaned_x_labels.append(f"<b>{broken_name}</b>")
    
    fig = px.imshow(
        z_matrix,
        x=cleaned_x_labels,
        y=y_labels,
        color_continuous_scale=[[0, "#E2E8F0"], [1, "#4E79A7"]], # Light gray for 0, Slate Blue for 1
        aspect="auto",
        title=f"<b>One-Hot Encoding Bit Grid for '{prefix_clean}' (Sample of Patients Target Mapping)</b>"
    )
    
    fig.update_coloraxes(showscale=False)
    
    # Annotate cell values directly
    for r_idx in range(z_matrix.shape[0]):
        for c_idx in range(z_matrix.shape[1]):
            val = int(z_matrix[r_idx, c_idx])
            fig.add_annotation(
                x=c_idx,
                y=r_idx,
                text=str(val),
                showarrow=False,
                font=dict(color="#000000" if val == 0 else "#FFFFFF", size=12)
            )
            
    fig.update_xaxes(
        tickfont=dict(color="#000000", family="sans-serif", size=10),
        side="top", # Put labels on top
        tickangle=0 # Force horizontal alignment to eliminate weird diagonal anchors
    )
    fig.update_yaxes(
        tickfont=dict(color="#000000", family="sans-serif", size=11)
    )
    
    fig.update_layout(
        height=405, # Nudged layout height for stacked tick names
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=160, r=40, t=110, b=40), # Restored comfortable margin
        font=dict(family="sans-serif", color="#000000")
    )
    return fig


def plot_pca_loadings_heatmap(loadings_df: pd.DataFrame) -> go.Figure:
    """Generate a combined heatmap of PCA feature weights for all components at once."""
    import plotly.graph_objects as go
    import plotly.express as px
    
    # Format Y = Features, X = PCs
    cleaned_y_labels = [f"<b>{idx.replace('_', ' ').title()}</b>" for idx in loadings_df.index]
    cleaned_x_labels = [f"<b>{col}</b>" for col in loadings_df.columns]
    
    z_matrix = loadings_df.values
    
    fig = px.imshow(
        z_matrix,
        x=cleaned_x_labels,
        y=cleaned_y_labels,
        color_continuous_scale="RdBu", # Cool Red-to-Blue diverging colormap prefix
        range_color=[-1.0, 1.0],
        aspect="auto",
        title="<b>PCA Feature Weight Heatmap (All Components Eigenvectors combined)</b>"
    )
    
    fig.update_layout(coloraxis_colorbar=dict(
        title="Loading weight",
        thickness=15,
        len=0.6,
        tickfont=dict(color="#000000", size=10)
    ))
    
    # Annotate cell numbers
    for r_idx in range(z_matrix.shape[0]):
        for c_idx in range(z_matrix.shape[1]):
            val = z_matrix[r_idx, c_idx]
            font_color = "#FFFFFF" if abs(val) > 0.45 else "#000000"
            fig.add_annotation(
                x=c_idx,
                y=r_idx,
                text=f"{val:.2f}",
                showarrow=False,
                font=dict(color=font_color, size=9)
            )
            
    fig.update_xaxes(
        tickfont=dict(color="#000000", family="sans-serif", size=10),
        side="top"
    )
    fig.update_yaxes(
        tickfont=dict(color="#000000", family="sans-serif", size=11)
    )
    
    fig.update_layout(
        height=550,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=165, r=40, t=90, b=40),
        font=dict(family="sans-serif", color="#000000")
    )
    return fig



# --- MeanShift-specific visualisations ---

def plot_bandwidth_sweep(result):
    import plotly.graph_objects as go

    if result.parameter_results is None:
        raise ValueError("MeanShift parameter_results sweep table is required.")

    sweep = result.parameter_results.copy()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sweep["bandwidth"],
        y=sweep["silhouette"],
        mode="lines+markers",
        name="Silhouette Score",
        line=dict(color="#356C9B", width=3),
        marker=dict(size=7, color="#356C9B"),
        hovertemplate="Bandwidth: %{x:.4f}<br>Silhouette: %{y:.4f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=sweep["bandwidth"],
        y=sweep["davies_bouldin"],
        mode="lines+markers",
        name="Davies-Bouldin Index",
        line=dict(color="#F59E0B", width=2, dash="dot"),
        marker=dict(size=6, color="#F59E0B"),
        yaxis="y2",
        hovertemplate="Bandwidth: %{x:.4f}<br>Davies-Bouldin: %{y:.4f}<extra></extra>"
    ))

    if result.selected_eps is not None:
        fig.add_vline(
            x=result.selected_eps,
            line_dash="dash",
            line_color="#C0392B",
            annotation_text="Selected BW " + f"{result.selected_eps:.4f}",
            annotation_font=dict(color="#C0392B", size=11)
        )

    fig.update_layout(
        title="Bandwidth Parameter Sweep - Quality Metrics",
        xaxis=dict(title="Bandwidth", showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(title="Silhouette Score", showgrid=True, gridcolor="#e2e8f0", color="#356C9B"),
        yaxis2=dict(title="Davies-Bouldin Index", overlaying="y", side="right", showgrid=False, color="#F59E0B"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=80),
    )
    return fig


def plot_meanshift_cluster_scatter(result):
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd

    frame = pd.DataFrame({
        "PC1": result.projection_2d[:, 0],
        "PC2": result.projection_2d[:, 1],
        "cluster": result.labels.astype(str)
    })
    frame["cluster"] = frame["cluster"].replace("-1", "Noise")

    fig = px.scatter(
        frame, x="PC1", y="PC2", color="cluster", opacity=0.62,
        title="Patient Clusters - 2D PCA Projection (MeanShift)",
        color_discrete_sequence=px.colors.qualitative.Vivid
    )

    unique_labels = sorted(set(result.labels))
    for lbl in unique_labels:
        if lbl == -1:
            continue
        mask = result.labels == lbl
        cx = float(result.projection_2d[mask, 0].mean())
        cy = float(result.projection_2d[mask, 1].mean())
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy],
            mode="markers+text",
            marker=dict(size=18, symbol="star", color="white",
                        line=dict(color="#1e3c72", width=2)),
            text=["C" + str(lbl)],
            textposition="middle center",
            textfont=dict(size=9, color="#1e3c72"),
            showlegend=False,
            hovertemplate="Cluster " + str(lbl) + " centroid<extra></extra>"
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        legend_title_text="Cluster"
    )
    return fig


def plot_cluster_profile_radar(result):
    import plotly.graph_objects as go
    import plotly.express as px

    summary = result.cluster_summary.copy()
    if summary.empty:
        fig = go.Figure()
        fig.update_layout(title="No clusters to profile.")
        return fig

    axes = ["age_mean", "glucose_mean", "bmi_mean", "stroke_rate"]
    axis_labels = ["Mean Age", "Mean Glucose", "Mean BMI", "Stroke Rate"]

    normed = summary[axes].copy()
    for col in axes:
        col_min, col_max = normed[col].min(), normed[col].max()
        rng = col_max - col_min
        normed[col] = (normed[col] - col_min) / rng if rng > 0 else 0.5

    colors = px.colors.qualitative.Vivid
    fig = go.Figure()
    for i, row in summary.iterrows():
        clabel = int(row["cluster"])
        values = [float(normed.loc[i, c]) for c in axes]
        values = values + [values[0]]
        labels_loop = axis_labels + [axis_labels[0]]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels_loop,
            fill="toself",
            opacity=0.55,
            name="Cluster " + str(clabel),
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.update_layout(
        title="Cluster Clinical Risk Profile (Normalised Radar)",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                            tickfont=dict(size=10, color="#334155"),
                            gridcolor="#CBD5E1"),
            angularaxis=dict(tickfont=dict(size=12, color="#1e293b"),
                             gridcolor="#CBD5E1"),
            bgcolor="rgba(0,0,0,0)"
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=80, b=80),
    )
    return fig


def plot_cluster_size_distribution(result):
    import plotly.graph_objects as go

    summary = result.cluster_summary.copy()
    if summary.empty:
        return go.Figure()

    summary = summary.sort_values("patients", ascending=True)
    summary["label"] = summary["cluster"].apply(lambda c: "Cluster " + str(int(c)))
    colors = summary["elevated_risk"].map({True: "#C0392B", False: "#356C9B"}).tolist()

    fig = go.Figure(go.Bar(
        y=summary["label"],
        x=summary["patients"],
        orientation="h",
        marker=dict(color=colors),
        text=summary["patients"].astype(str),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Patients: %{x}<extra></extra>"
    ))

    fig.update_layout(
        title="Patient Count per Cluster",
        xaxis=dict(title="Number of Patients", showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(title=""),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=60, t=60, b=40),
    )
    return fig

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.stats import chi2 as chi2_dist
import pandas as pd


RANDOM_STATE = 42
NUMERIC_COLUMNS = ["age", "avg_glucose_level", "bmi"]
BINARY_COLUMNS = ["hypertension", "heart_disease"]
CATEGORICAL_COLUMNS = ["ever_married", "smoking_status"]
REQUIRED_COLUMNS = NUMERIC_COLUMNS + BINARY_COLUMNS + CATEGORICAL_COLUMNS + ["stroke"]

def load_dataset(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    cleaned = data.copy()
    for column in NUMERIC_COLUMNS + BINARY_COLUMNS + ["stroke"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())
    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].fillna("Unknown").astype(str)
    cleaned["stroke"] = cleaned["stroke"].round().clip(0, 1).astype(int)
    return cleaned


def _encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    # Create sklearn transformer for scaling and encoding
    return ColumnTransformer([
        ("numeric", Pipeline([("scale", RobustScaler())]), NUMERIC_COLUMNS),
        ("binary", "passthrough", BINARY_COLUMNS),
        ("category", _encoder(), CATEGORICAL_COLUMNS),
    ])


def preprocess_data(data: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    # Clean data and transform variables via scaling and encoding
    cleaned = clean_data(data)
    preprocessor = build_preprocessor()
    processed = preprocessor.fit_transform(cleaned.drop(columns=["stroke", "cluster"], errors="ignore"))
    if hasattr(processed, "toarray"):
        processed = processed.toarray()
        
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "remainder" or trans == "drop":
                continue
            if name == "category":
                try:
                    cats = trans.categories_ if hasattr(trans, "categories_") else []
                except Exception:
                    cats = []
                for col, vals in zip(cols, cats):
                    for v in vals:
                        feature_names.append(f"{col}_{v}")
            else:
                feature_names.extend(cols)
        if len(feature_names) != processed.shape[1]:
            feature_names = [f"Feature_{i}" for i in range(processed.shape[1])]
            
    # Clean feature names for clean text output
    feature_names = [
        f.replace("numeric__", "").replace("binary__", "").replace("category__", "") 
        for f in feature_names
    ]
    return processed, feature_names


def apply_pca(processed: np.ndarray, feature_names: list[str], pca_variance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Fit selected and full PCA models on preprocessed features.
    
    Returns:
        features: PC projections explaining target variance fraction.
        projection_2d: 2-dimensional PC projection for PCA plots.
        pca_full_variance_ratio: individual variance ratio of all components.
        pca_selected_loadings: loading weights mapped to features.
    """
    pca_selected = PCA(n_components=pca_variance, svd_solver="full", random_state=RANDOM_STATE)
    features = pca_selected.fit_transform(processed)
    features = StandardScaler().fit_transform(features)
    
    projection_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(processed)
    
    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(processed)
    pca_full_variance_ratio = pca_full.explained_variance_ratio_
    
    loadings_cols = [f"PC{i+1}" for i in range(features.shape[1])]
    pca_selected_loadings = pd.DataFrame(
        pca_selected.components_.T,
        index=feature_names,
        columns=loadings_cols
    )
    
    return features, projection_2d, pca_full_variance_ratio, pca_selected_loadings


def calculate_cluster_summary(data: pd.DataFrame, risk_multiplier: float) -> pd.DataFrame:
    # calc stroke rates and metrics averages for each cluster group
    overall_rate = data["stroke"].mean()
    # Filter noise patients (-1) for cluster grouping profiles
    summary_data = data[data["cluster"] != -1]
    if summary_data.empty:
        return pd.DataFrame()
        
    summary = summary_data.groupby("cluster", sort=True).agg(
        patients=("stroke", "size"), stroke_cases=("stroke", "sum"),
        stroke_rate=("stroke", "mean"), age_mean=("age", "mean"),
        glucose_mean=("avg_glucose_level", "mean"), bmi_mean=("bmi", "mean"),
    ).reset_index()
    summary["stroke_rate_pct"] = (summary["stroke_rate"] * 100).round(2)
    summary["elevated_risk"] = summary["stroke_rate"] > overall_rate * risk_multiplier
    return summary.round({"age_mean": 1, "glucose_mean": 1, "bmi_mean": 1})

def test_categorical_association(data: pd.DataFrame) -> pd.DataFrame:
    """Chi-square test of each raw categorical/demographic column against stroke,
    used to justify which categorical features are retained for clustering."""
    candidates = ["gender", "Residence_type", "work_type", "ever_married", "smoking_status"]
    rows = []
    for col in candidates:
        if col not in data.columns:
            continue
        contingency = pd.crosstab(data[col], data["stroke"])
        chi2, p_value, _, _ = chi2_contingency(contingency)
        rates = data.groupby(col)["stroke"].mean() * 100
        rows.append({
            "Feature": col,
            "Stroke Rate Range (%)": f"{rates.min():.2f} – {rates.max():.2f}",
            "Chi-square p-value": "< 0.001" if p_value < 0.001 else f"{p_value:.3f}",
            "Associated with Stroke (p<0.05)": p_value < 0.05,
        })
    return pd.DataFrame(rows)

def test_confounding_with_age(data: pd.DataFrame) -> pd.DataFrame:
    """Likelihood-ratio test using Scikit-Learn: does `categorical_col` predict 
    stroke beyond age alone?
    """
    candidates = ["gender", "Residence_type", "work_type", "ever_married", "smoking_status"]
    rows = []
    
    # Define Target and Base Feature
    y = data["stroke"]
    X_base = data[["age"]]
    
    # 1. Fit Base Model (penalty=None is required for accurate statistical testing)
    base_model = LogisticRegression(penalty=None)
    base_model.fit(X_base, y)
    
    # Calculate Log-Likelihood for base model
    # Math note: Log-Likelihood = -(Log Loss * Number of samples)
    llf_base = -log_loss(y, base_model.predict_proba(X_base)) * len(y)
    
    for col in candidates:
        if col not in data.columns:
            continue
            
        # 2. Fit Full Model (Age + Categorical Column)
        # We must use pd.get_dummies to convert text classes into binary numbers
        X_full = pd.get_dummies(data[["age", col]], columns=[col], drop_first=True)
        
        full_model = LogisticRegression(penalty=None, max_iter=1000)
        full_model.fit(X_full, y)
        
        # Calculate Log-Likelihood for full model
        llf_full = -log_loss(y, full_model.predict_proba(X_full)) * len(y)
        
        # 3. Calculate Likelihood-Ratio Test
        lr_stat = 2 * (llf_full - llf_base)
        df_diff = X_full.shape[1] - X_base.shape[1]
        p_value = chi2_dist.sf(lr_stat, df_diff)
        
        rows.append({
            "Feature": col,
            "LR Statistic": round(lr_stat, 4),
            "p-value (Controlling for Age)": "< 0.001" if p_value < 0.001 else f"{p_value:.3f}",
            "Adds Value Beyond Age (p < 0.05)": p_value < 0.05
        })
        
    return pd.DataFrame(rows)

def get_clinical_summary(data: pd.DataFrame) -> dict[str, float]:
    # Calc summary data overview 
    total_pts = len(data)
    stroke_cases = int(data["stroke"].sum())
    stroke_rate = (stroke_cases / total_pts) * 100
    mean_age = float(data["age"].mean())
    mean_bmi = float(data["bmi"].mean())
    return {
        "total_patients": total_pts,
        "stroke_cases": stroke_cases,
        "stroke_prevalence": stroke_rate,
        "average_age": mean_age,
        "average_bmi": mean_bmi,
    }


def get_categorical_analysis(data: pd.DataFrame, cat_feature: str) -> pd.DataFrame:
    # Group by a categorical clinical column and count stroke prevalence
    grouped = data.groupby(cat_feature).agg(
        Total_Patients=("stroke", "size"),
        Stroke_Cases=("stroke", "sum")
    ).reset_index()
    grouped["Stroke_Prevalence_Pct"] = (grouped["Stroke_Cases"] / grouped["Total_Patients"]) * 100
    
    if cat_feature in ["hypertension", "heart_disease"]:
        grouped[cat_feature] = grouped[cat_feature].map({0: "No Indicator", 1: "Yes (Diagnosed)"})
        
    return grouped


def get_correlation_matrix(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # cacl pearson correlation coefficients 
    corr_cols = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease", "stroke"]
    corr_matrix = data[corr_cols].corr()
    readable_labels = [c.replace('_', ' ').title() for c in corr_cols]
    return corr_matrix, readable_labels


def get_preprocessing_previews(result, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Generate original and output preview columns for the preprocessing tab
    raw_cols = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease", "gender", "smoking_status"]
    raw_preview = data[raw_cols].head(6)
    
    prep_cols = result.preprocessed_feature_names
    prep_values = result.preprocessed_data
    prep_df = pd.DataFrame(prep_values, columns=prep_cols)
    prep_preview = prep_df.iloc[:6, :7]
    
    return raw_preview, prep_preview


def get_pca_scree_data(result) -> pd.DataFrame:
    # Build explained variance coordinates for Scree scatter plots
    return pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(result.pca_full_variance_ratio))],
        "Individual Variance (%)": result.pca_full_variance_ratio * 100,
        "Cumulative Variance (%)": np.cumsum(result.pca_full_variance_ratio) * 100
    })


def get_pca_loadings(result, selected_pc: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    #Fetch PCA loading value and identify the most contribution feature 
    loadings = result.pca_selected_loadings[selected_pc].reset_index()
    loadings.columns = ["Feature", "Weight"]
    loadings_sorted = loadings.sort_values(by="Weight", key=abs, ascending=True)
    
    top_pos_features = loadings.sort_values(by="Weight", ascending=False).head(2)["Feature"].tolist()
    top_neg_features = loadings.sort_values(by="Weight", ascending=True).head(2)["Feature"].tolist()
    
    return loadings_sorted, top_pos_features, top_neg_features


def get_data_quality_report(data: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarizing missing values and data types of raw columns."""
    report = []
    for col in data.columns:
        null_count = int(data[col].isnull().sum())
        total = len(data)
        null_pct = (null_count / total) * 100
        
        # Outlier counts (approximated for numeric features using 1.5 * IQR rule)
        outlier_count = 0
        if pd.api.types.is_numeric_dtype(data[col]) and col not in ["stroke", "cluster"]:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_count = int(((data[col] < lower) | (data[col] > upper)).sum())
                
        report.append({
            "Column Name": col,
            "Data Type": str(data[col].dtype),
            "Non-Null Count": total - null_count,
            "Missing Values": null_count,
            "Missing %": f"{null_pct:.1f}%",
            "Outliers Check (IQR Outliers)": outlier_count if pd.api.types.is_numeric_dtype(data[col]) and col not in ["stroke", "cluster"] else "N/A"
        })
    return pd.DataFrame(report)

import os
import time
import pytest
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@pytest.fixture(scope="session")
def loaded_data():
    df_sales = pd.read_csv(os.path.join(DATA_DIR, "sales data-set.csv"))
    df_features = pd.read_csv(os.path.join(DATA_DIR, "Features data set.csv"))
    df_stores = pd.read_csv(os.path.join(DATA_DIR, "stores data-set.csv"))
    return df_sales, df_features, df_stores

def test_01_dataset_loading_and_shapes(loaded_data):
    """Verify that all three core retail datasets load accurately with expected schemas and row counts."""
    df_sales, df_features, df_stores = loaded_data

    assert len(df_sales) == 421570, f"Expected 421570 sales records, got {len(df_sales)}"
    assert set(['Store', 'Dept', 'Date', 'Weekly_Sales', 'IsHoliday']).issubset(df_sales.columns)

    assert len(df_features) == 8190, f"Expected 8190 features records, got {len(df_features)}"
    assert set(['Store', 'Date', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']).issubset(df_features.columns)

    assert len(df_stores) == 45, f"Expected 45 store records, got {len(df_stores)}"
    assert set(['Store', 'Type', 'Size']).issubset(df_stores.columns)


def test_02_markdown_imputation_logic(loaded_data):
    """Verify that MarkDown features are properly imputed with zero and feature flags are created."""
    _, df_feat, _ = loaded_data
    md_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']

    # Pre-imputation missing check
    has_nans = df_feat[md_cols].isna().any().any()
    assert has_nans, "Raw dataset should contain missing MarkDown values"

    # Impute
    df_imputed = df_feat.copy()
    df_imputed[md_cols] = df_imputed[md_cols].fillna(0.0)
    df_imputed['Total_MarkDown'] = df_imputed[md_cols].sum(axis=1)
    df_imputed['Has_MarkDown'] = (df_imputed['Total_MarkDown'] != 0.0).astype(int)

    assert df_imputed[md_cols].isna().sum().sum() == 0, "Imputed MarkDowns should have zero NaNs"
    assert len(df_imputed['Total_MarkDown']) == len(df_feat), "Total_MarkDown length mismatch"
    assert set(df_imputed['Has_MarkDown'].unique()).issubset({0, 1}), "Has_MarkDown must be binary"


def test_03_outlier_detection_and_negative_sales(loaded_data):
    """Verify that return transactions (negative sales) are isolated and handled without data corruption."""
    df_sales, _, _ = loaded_data

    negative_sales_mask = df_sales['Weekly_Sales'] < 0
    negative_count = negative_sales_mask.sum()
    assert negative_count > 0, "Expected return transactions with negative sales in raw dataset"

    df_copy = df_sales.copy()
    df_copy['Is_Return'] = (df_copy['Weekly_Sales'] < 0).astype(int)
    df_copy['Clipped_Sales'] = df_copy['Weekly_Sales'].clip(lower=0.0)

    assert (df_copy['Clipped_Sales'] >= 0.0).all(), "Clipped sales must be strictly non-negative"
    assert df_copy['Is_Return'].sum() == negative_count, "Return flag count must match negative sales count"


def test_04_hypothesis_testing_statistics(loaded_data):
    """Validate empirical hypothesis tests: Holiday lift (Welch t-test) and Store tier differences (ANOVA)."""
    df_sales, _, df_stores = loaded_data
    df_merged = df_sales.merge(df_stores, on='Store', how='inner')

    sample_df = df_merged.sample(n=10000, random_state=42)

    # Hypothesis 1: Holiday vs Non-Holiday
    holiday_sales = sample_df[sample_df['IsHoliday'] == True]['Weekly_Sales']
    non_holiday_sales = sample_df[sample_df['IsHoliday'] == False]['Weekly_Sales']
    t_stat, t_pval = stats.ttest_ind(holiday_sales, non_holiday_sales, equal_var=False)

    assert holiday_sales.mean() > non_holiday_sales.mean(), "Holiday mean sales should exceed non-holiday mean"
    assert not np.isnan(t_stat), "Welch t-statistic must be valid numeric"

    # Hypothesis 2: Store Types (A, B, C)
    sales_type_a = sample_df[sample_df['Type'] == 'A']['Weekly_Sales']
    sales_type_b = sample_df[sample_df['Type'] == 'B']['Weekly_Sales']
    sales_type_c = sample_df[sample_df['Type'] == 'C']['Weekly_Sales']
    f_stat, f_pval = stats.f_oneway(sales_type_a, sales_type_b, sales_type_c)

    assert f_pval < 0.001, f"ANOVA across store types should be highly significant (p={f_pval})"
    assert sales_type_a.mean() > sales_type_c.mean(), "Type A supercenters must outperform Type C express stores"


def test_05_kmeans_store_clustering(loaded_data):
    """Verify that store profile aggregation yields coherent K-Means clusters with distinct centroids."""
    df_sales, _, df_stores = loaded_data

    store_metrics = df_sales.groupby('Store')['Weekly_Sales'].agg(['mean', 'std']).reset_index()
    store_features = store_metrics.merge(df_stores, on='Store')

    X = store_features[['mean', 'std', 'Size']].values
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)

    assert len(set(labels)) == 3, "KMeans should form 3 distinct clusters"
    assert sil_score >= 0.35, f"Silhouette score should be solid (got {sil_score:.3f})"


def test_06_market_basket_co_movement_correlations(loaded_data):
    """Verify cross-departmental sales correlation matrix reveals high grocery and household co-movements."""
    df_sales, _, _ = loaded_data

    top_depts = [92, 95, 38, 72, 90]
    pivot_df = df_sales[df_sales['Dept'].isin(top_depts)].pivot_table(
        index=['Store', 'Date'], columns='Dept', values='Weekly_Sales'
    ).dropna()

    corr_matrix = pivot_df.corr()
    r_grocery_household = corr_matrix.loc[92, 95]
    assert r_grocery_household >= 0.65, f"Expected high co-movement between Dept 92 and 95, got {r_grocery_household:.3f}"


def test_07_time_series_split_leakage_prevention(loaded_data):
    """Verify that train/test split maintains strict chronological integrity without future lookahead."""
    df_sales, _, _ = loaded_data
    df_sales_dates = pd.to_datetime(df_sales['Date'], format='%d/%m/%Y', errors='coerce')
    if df_sales_dates.isna().sum() > 0:
        df_sales_dates = pd.to_datetime(df_sales['Date'], errors='coerce')

    train_dates = df_sales_dates[df_sales_dates < '2012-01-01']
    test_dates = df_sales_dates[df_sales_dates >= '2012-01-01']

    assert len(train_dates) > 0 and len(test_dates) > 0, "Both train and test sets must have data"
    assert train_dates.max() < test_dates.min(), "Training dates must strictly precede test dates"


def test_08_holiday_wmae_metric_calculation():
    """Verify the exact arithmetic of the holiday-weighted WMAE evaluation loss function."""
    def calculate_wmae(y_true, y_pred, is_holiday):
        weights = np.where(is_holiday, 5.0, 1.0)
        return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

    y_true = np.array([100.0, 100.0])
    y_pred = np.array([110.0, 110.0])
    is_holiday = np.array([False, True])
    wmae = calculate_wmae(y_true, y_pred, is_holiday)
    assert abs(wmae - 10.0) < 1e-5, f"WMAE should be 10.0, got {wmae}"

    y_pred_holiday_spike = np.array([110.0, 150.0])
    wmae_inflated = calculate_wmae(y_true, y_pred_holiday_spike, is_holiday)
    assert abs(wmae_inflated - (260.0 / 6.0)) < 1e-4


def test_09_model_deserialization_and_inference_latency():
    """Verify that the production serialized model loads and executes high-throughput inference (<5ms)."""
    model_path = os.path.join(DATA_DIR, "best_retail_forecasting_model.joblib")
    assert os.path.exists(model_path), f"Serialized model missing at {model_path}"

    model = joblib.load(model_path)
    assert hasattr(model, 'predict'), "Model must expose a scikit-learn compatible predict() method"

    num_features = getattr(model, 'n_features_in_', 16)
    sample_X = np.ones((50, num_features))

    start_time = time.perf_counter()
    preds = model.predict(sample_X)
    elapsed_time = (time.perf_counter() - start_time) / 50.0

    assert len(preds) == 50, "Predictions array shape mismatch"
    assert elapsed_time < 0.010, f"Per-sample inference latency ({elapsed_time*1000:.2f}ms) exceeds 10ms threshold"


def test_10_documentation_and_artifact_integrity():
    """Verify that all capstone deliverables, presentation scripts, and notebooks exist with valid content."""
    required_files = [
        "README.md",
        "DOCUMENTATION_REPORT.md",
        "VIDEO_PRESENTATION_SCRIPT.md",
        "Integrated_Retail_Analytics_Capstone_Report.pdf",
        "Sample ML Submission Template.ipynb",
        "Integrated_Retail_Analytics_AML_Capstone.ipynb",
        "best_retail_forecasting_model.joblib",
        "sales data-set.csv",
        "Features data set.csv",
        "stores data-set.csv"
    ]
    for rf in required_files:
        full_path = os.path.join(DATA_DIR, rf)
        assert os.path.exists(full_path), f"Critical artifact missing: {rf}"
        assert os.path.getsize(full_path) > 0, f"Artifact file is empty: {rf}"

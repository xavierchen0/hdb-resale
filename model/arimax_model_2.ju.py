# %% [md]
# # ARIMAX Modelling Pipeline for HDB Resale Prices
#
# Builds on the probability, econometrics, and time-series tooling covered in Sessions 1–8
# to engineer a monthly panel, run diagnostic tests, carry out correlation filtering, and fit
# an ARIMAX model leveraging the full exogenous feature set for Singapore HDB resale prices.

# %% [md]
# ## Workflow Outline
# 1. Load the merged feature dataset and collapse it to a monthly panel.
# 2. Run preliminary diagnostics (missingness, stationarity tests, ACF/PACF).
# 3. Select ARIMA orders via a light grid-search after differencing checks.
# 4. Prepare exogenous features (impute, scale), apply correlation filtering, and retain the pool.
# 5. Fit, evaluate, and diagnose the final model; optionally refit on the full sample.

# %% [md]
# ## 1. Imports & Configuration

# %%
from __future__ import annotations

from itertools import product
from pathlib import Path
import sys
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import jarque_bera
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX

project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DATA_DIR  # noqa: E402

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 220)

DATA_PATH = DATA_DIR / "final_dataset.parquet"
DATE_COL = "month"
TARGET_COL = "price_per_sqm"
DROP_COLUMNS = {
    "txn_id",
    "geometry",
    "town",
    "region",
    "flat_type",
    "year",
}
INDEX_FREQ = "MS"  # Month start frequency
TARGET_AGG = "median"
FEATURE_AGG = "mean"
MIN_OBS_REQUIRED = 36
TEST_HORIZON = 12  # last N months kept for evaluation
MIN_NON_NULL_RATIO = 0.75
MIN_STD_THRESHOLD = 1e-4
USE_LOG_TARGET = True
MAX_DIFF = 2
USE_SEASONAL = False
SEASONAL_PERIOD = 12
SEASONAL_DIFF = 1
MAX_P = 3
MAX_Q = 3
MAX_P_SEASONAL = 1
MAX_Q_SEASONAL = 1
MODEL_DIR = DATA_DIR / "model"
MODEL_EXPORT_PATH = MODEL_DIR / "arimax2_model.pkl"

# %% [md]
# ## 2. Data Loading & Basic Checks


# %%
def load_dataset(
    path: Path, date_col: str, drop_columns: Iterable[str]
) -> pd.DataFrame:
    """Read the feature-enriched dataset and return a pandas DataFrame."""
    try:
        import geopandas as gpd

        df = gpd.read_parquet(path)
        if hasattr(df, "geometry"):
            df = df.drop(columns=["geometry"], errors="ignore")
    except Exception:
        df = pd.read_parquet(path)

    df = pd.DataFrame(df)  # ensure plain DataFrame
    cols_to_drop = set(drop_columns) & set(df.columns)
    if cols_to_drop:
        df = df.drop(columns=list(cols_to_drop), errors="ignore")
    df = df.drop_duplicates()
    if date_col not in df.columns:
        raise KeyError(f"'{date_col}' not found in columns: {df.columns.tolist()}")
    return df


df_raw = load_dataset(DATA_PATH, DATE_COL, DROP_COLUMNS)
print(f"Loaded shape: {df_raw.shape}")
df_raw.head()

# %%
if {"resale_price", "floor_area_sqm"} - set(df_raw.columns):
    missing = {"resale_price", "floor_area_sqm"} - set(df_raw.columns)
    raise KeyError(f"Missing required columns for price-per-sqm target: {missing}")

df_raw["price_per_sqm"] = np.log(df_raw["resale_price"] / df_raw["floor_area_sqm"])
df_raw.drop(columns=["resale_price", "floor_area_sqm"], inplace=True)
df_raw

# %%
missing_summary = (
    df_raw.isna().mean().sort_values(ascending=False).head(20).to_frame("missing_ratio")
)
missing_summary

# %% [md]
# ## 3. Aggregate to a Monthly Panel


# %%
def build_monthly_panel(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    index_freq: str,
    target_agg: str,
    feature_agg: str,
) -> pd.DataFrame:
    """Collapse raw transactions to a monthly panel."""
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    tmp = tmp.sort_values(date_col)
    tmp = tmp.set_index(date_col)

    numeric_cols = tmp.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in numeric_cols:
        raise ValueError(f"Target '{target_col}' must be numeric.")

    feature_cols = [c for c in numeric_cols if c != target_col]

    target_series = (
        tmp[target_col]
        .groupby(pd.Grouper(freq=index_freq))
        .agg(target_agg)
        .rename(target_col)
    )
    feature_frame = (
        tmp[feature_cols]
        .groupby(pd.Grouper(freq=index_freq))
        .agg(feature_agg)
        .astype(float)
    )

    ts_df = pd.concat([target_series, feature_frame], axis=1)
    ts_df.index.name = "period"

    if len(ts_df) < MIN_OBS_REQUIRED:
        raise RuntimeError(
            f"Insufficient observations ({len(ts_df)}). Need at least {MIN_OBS_REQUIRED} months."
        )

    full_idx = pd.date_range(ts_df.index.min(), ts_df.index.max(), freq=index_freq)
    ts_df = ts_df.reindex(full_idx)
    return ts_df


ts_df = build_monthly_panel(
    df_raw,
    date_col=DATE_COL,
    target_col=TARGET_COL,
    index_freq=INDEX_FREQ,
    target_agg=TARGET_AGG,
    feature_agg=FEATURE_AGG,
)
ts_df.head()

# %%
ts_df[[TARGET_COL]].plot(figsize=(12, 4), title="Monthly HDB Resale Price Target")
plt.tight_layout()

# %% [md]
# ## 4. Stationarity Diagnostics & Order Selection Helpers


# %%
def safe_adf(series: pd.Series) -> dict:
    result = {"adf_stat": np.nan, "p_value": np.nan, "lags": np.nan, "nobs": np.nan}
    try:
        stat, pval, usedlag, nobs, *_ = adfuller(
            series.dropna(), autolag="AIC", maxlag=12
        )
        result.update(
            {"adf_stat": stat, "p_value": pval, "lags": usedlag, "nobs": nobs}
        )
    except Exception as exc:
        result["error"] = str(exc)
    return result


def safe_kpss(series: pd.Series, regression: str = "c") -> dict:
    result = {"kpss_stat": np.nan, "p_value": np.nan, "lags": np.nan}
    try:
        stat, pval, nlags, *_ = kpss(
            series.dropna(), regression=regression, nlags="auto"
        )
        result.update({"kpss_stat": stat, "p_value": pval, "lags": nlags})
    except Exception as exc:
        result["error"] = str(exc)
    return result


def estimate_d(series: pd.Series, max_d: int) -> int:
    """Estimate differencing order via successive stationarity checks."""
    for d in range(max_d + 1):
        if d == 0:
            candidate = series
        else:
            candidate = series.diff(d).dropna()
        adf_res = safe_adf(candidate)
        kpss_res = safe_kpss(candidate)
        adf_ok = adf_res["p_value"] is not np.nan and adf_res["p_value"] < 0.05
        kpss_ok = kpss_res["p_value"] is not np.nan and kpss_res["p_value"] > 0.05
        if adf_ok and kpss_ok:
            return d
    return max_d


def sarimax_order_grid_search(
    series: pd.Series,
    d: int,
    seasonal: bool,
    seasonal_period: int,
    seasonal_d: int,
    max_p: int,
    max_q: int,
    max_P: int,
    max_Q: int,
) -> pd.DataFrame:
    """Lightweight grid-search for SARIMAX orders using AIC."""
    results = []
    y = series.dropna()
    if y.empty:
        raise ValueError("Series has no observations for order search.")

    p_range = range(max_p + 1)
    q_range = range(max_q + 1)
    P_range = range(max_P + 1) if seasonal else [0]
    Q_range = range(max_Q + 1) if seasonal else [0]

    for p, q, P, Q in product(p_range, q_range, P_range, Q_range):
        order = (p, d, q)
        seasonal_order = (
            (P, seasonal_d, Q, seasonal_period) if seasonal else (0, 0, 0, 0)
        )
        try:
            model = SARIMAX(
                y,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit_res = model.fit(disp=False)
            results.append(
                {
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "aic": fit_res.aic,
                    "bic": fit_res.bic,
                    "converged": fit_res.mle_retvals.get("converged", True),
                }
            )
        except Exception:
            continue

    if not results:
        raise RuntimeError("No SARIMAX models converged during grid search.")

    return pd.DataFrame(results).sort_values("aic").reset_index(drop=True)


stationarity_report = {
    "adf": safe_adf(ts_df[TARGET_COL]),
    "kpss": safe_kpss(ts_df[TARGET_COL]),
}
stationarity_report

# %%
d_estimate = estimate_d(ts_df[TARGET_COL], MAX_DIFF)
print(f"Suggested differencing order d = {d_estimate}")

# %%
order_candidates = sarimax_order_grid_search(
    ts_df[TARGET_COL],
    d=1,
    seasonal=USE_SEASONAL,
    seasonal_period=SEASONAL_PERIOD,
    seasonal_d=SEASONAL_DIFF if USE_SEASONAL else 0,
    max_p=MAX_P,
    max_q=MAX_Q,
    max_P=MAX_P_SEASONAL,
    max_Q=MAX_Q_SEASONAL,
)
order_candidates.head()

# %%
BEST_ORDER = tuple(order_candidates.loc[0, "order"])
BEST_SEASONAL_ORDER = tuple(order_candidates.loc[0, "seasonal_order"])
print(f"Selected order={BEST_ORDER}, seasonal_order={BEST_SEASONAL_ORDER}")

# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
plot_acf(ts_df[TARGET_COL].dropna(), lags=24, ax=axes[0])
plot_pacf(ts_df[TARGET_COL].dropna(), lags=24, ax=axes[1])
axes[0].set_title("ACF of Target")
axes[1].set_title("PACF of Target")
plt.tight_layout()

# %% [md]
# ## 5. Feature Preparation for Stepwise AIC


# %%
def prepare_feature_matrices(
    panel: pd.DataFrame,
    target_col: str,
    min_non_null_ratio: float,
    min_std: float,
    test_horizon: int,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Split into train/test and return cleaned feature matrices."""
    if test_horizon >= len(panel):
        raise ValueError("Test horizon is too large relative to sample size.")

    y_all = panel[target_col]
    X_all = panel.drop(columns=[target_col])

    train_slice = slice(0, len(panel) - test_horizon)
    y_train = y_all.iloc[train_slice]
    y_test = y_all.iloc[len(panel) - test_horizon :]
    X_train = X_all.iloc[train_slice]
    X_test = X_all.iloc[len(panel) - test_horizon :]

    valid_mask = X_train.notna().mean(axis=0) >= min_non_null_ratio
    X_train = X_train.loc[:, valid_mask]
    X_test = X_test.loc[:, valid_mask]

    std_mask = X_train.std(ddof=0) > min_std
    X_train = X_train.loc[:, std_mask]
    X_test = X_test.loc[:, std_mask]

    return y_train, y_test, X_train, X_test


y_train, y_test, X_train_raw, X_test_raw = prepare_feature_matrices(
    ts_df,
    target_col=TARGET_COL,
    min_non_null_ratio=MIN_NON_NULL_RATIO,
    min_std=MIN_STD_THRESHOLD,
    test_horizon=TEST_HORIZON,
)
print(f"Train months: {len(y_train)}, Test months: {len(y_test)}")
print(f"Initial feature count (train): {X_train_raw.shape[1]}")

# %%
imputer = SimpleImputer(strategy="median")
X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train_raw),
    index=X_train_raw.index,
    columns=X_train_raw.columns,
)
X_test_imputed = pd.DataFrame(
    imputer.transform(X_test_raw),
    index=X_test_raw.index,
    columns=X_test_raw.columns,
)

X_train_prepared = X_train_imputed
X_test_prepared = X_test_imputed[X_train_prepared.columns]

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_prepared),
    index=X_train_prepared.index,
    columns=X_train_prepared.columns,
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_prepared),
    index=X_test_prepared.index,
    columns=X_test_prepared.columns,
)

print(f"Exogenous features available for ARIMAX: {X_train_scaled.shape[1]}")

# %% [md]
# ## 6. Fit ARIMAX without Feature Selection


# %%
def fit_sarimax_model(
    endog: pd.Series,
    exog: Optional[pd.DataFrame],
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> Optional[object]:
    try:
        model = SARIMAX(
            endog,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False)
        return result
    except Exception:
        return None


# %% [md]
# ## 7. Fit Final Model & Evaluate

# %%
selected_cols = X_train_scaled.columns.tolist()
exog_train = X_train_scaled if selected_cols else None
exog_test = X_test_scaled[selected_cols] if selected_cols else None
print(f"Using all {len(selected_cols)} exogenous features for ARIMAX fit.")

# %%
results = []
d = 1
for q in range(4):
    for p in range(1, 4):
        test_model = fit_sarimax_model(
            y_train,
            exog_train,
            order=(p, d, q),
            seasonal_order=BEST_SEASONAL_ORDER,
        )
        results.append({"order": (p, d, q), "aic": test_model.aic})

order_candidates = pd.DataFrame(results)
order_candidates["aic"] = order_candidates["aic"].astype("float64")
order_candidates = order_candidates.sort_values(by="aic")

BEST_ORDER = tuple(order_candidates.loc[0, "order"])

order_candidates.head()

# %%
final_model = fit_sarimax_model(
    y_train,
    exog_train,
    order=BEST_ORDER,
    seasonal_order=BEST_SEASONAL_ORDER,
)

if final_model is None:
    raise RuntimeError("Final SARIMAX fit failed.")

print(final_model.summary())

# %%
forecast_res = final_model.get_forecast(
    steps=len(y_test),
    exog=exog_test,
)
pred_mean = forecast_res.predicted_mean
conf_int = forecast_res.conf_int()

if USE_LOG_TARGET:
    y_test_eval = np.exp(y_test)
    pred_eval = np.exp(pred_mean)
else:
    y_test_eval = y_test
    pred_eval = pred_mean

rmse = mean_squared_error(y_test_eval, pred_eval)
mae = mean_absolute_error(y_test_eval, pred_eval)
mape = mean_absolute_percentage_error(y_test_eval, pred_eval)

print(f"RMSE: {rmse:,.2f}")
print(f"MAE: {mae:,.2f}")
print(f"MAPE: {100 * mape:.2f}%")

# %%
plt.figure(figsize=(12, 4))
plt.plot(
    y_train.index, y_train if not USE_LOG_TARGET else np.exp(y_train), label="Train"
)
plt.plot(y_test.index, y_test_eval, label="Test Actual", color="black")
plt.plot(y_test.index, pred_eval, label="Forecast", color="tab:orange")
plt.fill_between(
    y_test.index,
    conf_int.iloc[:, 0] if not USE_LOG_TARGET else np.exp(conf_int.iloc[:, 0]),
    conf_int.iloc[:, 1] if not USE_LOG_TARGET else np.exp(conf_int.iloc[:, 1]),
    color="tab:orange",
    alpha=0.2,
    label="Confidence Interval",
)
plt.title("ARIMAX Forecast vs Actuals")
plt.legend()
plt.tight_layout()

# %%
residuals = final_model.resid
fig, axes = plt.subplots(3, 1, figsize=(10, 9))
residuals.plot(ax=axes[0], title="Residuals")
plot_acf(residuals.dropna(), lags=24, ax=axes[1])
plot_pacf(residuals.dropna(), lags=24, ax=axes[2])
axes[1].set_title("Residual ACF")
axes[2].set_title("Residual PACF")
plt.tight_layout()

# %%
ljung_box = acorr_ljungbox(residuals.dropna(), lags=[12, 24], return_df=True)
jb_stat, jb_p = jarque_bera(residuals.dropna())
diagnostics = {
    "ljung_box": ljung_box,
    "jarque_bera_stat": jb_stat,
    "jarque_bera_pvalue": jb_p,
}
diagnostics

# %%
MODEL_DIR.mkdir(parents=True, exist_ok=True)
final_model.save(str(MODEL_EXPORT_PATH))
print(f"Exported SARIMAX model to {MODEL_EXPORT_PATH}")

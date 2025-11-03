# %% [md]
# # ARIMAX Modelling Pipeline for HDB Resale Prices
#
# Builds on the probability, econometrics, and time-series tooling covered in Sessions 1–8
# to engineer a monthly panel, run diagnostic tests, carry out correlation filtering, and fit
# an ARIMAX model with stepwise AIC feature selection for Singapore HDB resale prices.

# %% [md]
# ## Workflow Outline
# 1. Load the merged feature dataset and collapse it to a monthly panel.
# 2. Run preliminary diagnostics (missingness, stationarity tests, ACF/PACF).
# 3. Select ARIMA orders via a light grid-search after differencing checks.
# 4. Prepare exogenous features (impute, scale), apply correlation filtering, and restrict the pool.
# 5. Run stepwise AIC feature selection for ARIMAX.
# 6. Fit, evaluate, and diagnose the final model; optionally refit on the full sample.

# %% [md]
# ## 1. Imports & Configuration

# %%
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import sys
from typing import Iterable, Optional, Sequence

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
from tqdm.auto import tqdm

project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DATA_DIR  # noqa: E402

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 220)

DATA_PATH = DATA_DIR / "final_dataset.parquet"
DATE_COL = "month"
TARGET_COL = "resale_price"
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
MAX_FEATURES_STEPWISE = 40
MUST_KEEP_FEATURES: Sequence[str] = (
    "floor_area_sqm",
    "remaining_lease",
    "distance_to_nearest_mrt_lrt",
    "distance_to_nearest_prischool",
    "distance_to_nearest_foodcentre",
    "distance_to_nearest_hawker",
    "distance_to_nearest_mall",
    "distance_to_nearest_preschool",
    "distance_to_nearest_cc",
    "distance_to_nearest_park",
    "storey_median",
    "flat_type_2 ROOM",
    "flat_type_3 ROOM",
    "flat_type_4 ROOM",
    "flat_type_5 ROOM",
    "flat_type_EXECUTIVE",
    "flat_type_MULTI-GENERATION",
)
USE_LOG_TARGET = False
MAX_DIFF = 2
USE_SEASONAL = True
SEASONAL_PERIOD = 12
SEASONAL_DIFF = 1
MAX_P = 3
MAX_Q = 3
MAX_P_SEASONAL = 1
MAX_Q_SEASONAL = 1
STEPWISE_TOL = 1e-3

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

# %%
if USE_LOG_TARGET:
    if (ts_df[TARGET_COL] <= 0).any():
        raise ValueError("Log transform requested but target has non-positive values.")
    ts_df[TARGET_COL] = np.log(ts_df[TARGET_COL])
    print("Applied natural log transform to target.")

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
    d=d_estimate,
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

protected = [feat for feat in MUST_KEEP_FEATURES if feat in X_train_imputed.columns]
missing_protected = [
    feat for feat in MUST_KEEP_FEATURES if feat not in X_train_imputed.columns
]
if missing_protected:
    print("Warning: required features not found and excluded:", missing_protected)

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

print(f"Features available for stepwise selection: {X_train_scaled.shape[1]}")

# %% [md]
# ## 6. Stepwise AIC Feature Selection for ARIMAX


# %%
@dataclass
class StepwiseResult:
    selected_features: tuple[str, ...]
    model: Optional[SARIMAX]
    fit_result: Optional[object]
    history: list[dict]


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


def stepwise_arimax_selection(
    y: pd.Series,
    X: pd.DataFrame,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    must_have: Iterable[str] | None = None,
    max_features: int = 10,
    tol: float = 1e-3,
    verbose: bool = True,
) -> StepwiseResult:
    must_have = tuple(feat for feat in (must_have or tuple()) if feat in X.columns)
    cache: dict[tuple[str, ...], Optional[object]] = {}
    history: list[dict] = []

    def evaluate(features: Sequence[str]) -> Optional[object]:
        key = tuple(features)
        if key in cache:
            return cache[key]
        exog = X[list(key)] if key else None
        result = fit_sarimax_model(y, exog, order, seasonal_order)
        cache[key] = result
        if result is not None:
            history.append({"features": key, "aic": result.aic})
        return result

    current_features = must_have if must_have else tuple()
    best_model = evaluate(current_features)
    if best_model is None:
        raise RuntimeError("Failed to fit baseline ARIMAX model.")
    best_aic = best_model.aic

    remaining = [feat for feat in X.columns if feat not in current_features]
    improved = True

    while improved and len(current_features) < max_features:
        improved = False
        candidate_features = current_features
        candidate_model = best_model
        candidate_aic = best_aic

        forward_bar = tqdm(
            remaining,
            desc=f"Forward search (k={len(current_features)})",
            leave=False,
            disable=not verbose,
        )
        for feat in forward_bar:
            trial = current_features + (feat,)
            result = evaluate(trial)
            if result is None:
                continue
            if result.aic + tol < candidate_aic:
                candidate_aic = result.aic
                candidate_features = trial
                candidate_model = result
                if verbose:
                    forward_bar.set_postfix(best_aic=f"{candidate_aic:.2f}", add=feat)
        forward_bar.close()

        if candidate_aic + tol < best_aic:
            current_features = candidate_features
            best_model = candidate_model
            best_aic = candidate_aic
            remaining = [feat for feat in X.columns if feat not in current_features]
            improved = True
            if verbose:
                tqdm.write(f"Forward step -> {current_features} | AIC={best_aic:.2f}")
            continue

        # Backward step
        if len(current_features) > len(must_have):
            backward_bar = tqdm(
                [feat for feat in current_features if feat not in must_have],
                desc=f"Backward search (k={len(current_features)})",
                leave=False,
                disable=not verbose,
            )
            for feat in backward_bar:
                if feat in must_have:
                    continue
                trial = tuple(f for f in current_features if f != feat)
                result = evaluate(trial)
                if result is None:
                    continue
                if result.aic + tol < candidate_aic:
                    candidate_aic = result.aic
                    candidate_features = trial
                    candidate_model = result
                    if verbose:
                        backward_bar.set_postfix(
                            best_aic=f"{candidate_aic:.2f}", drop=feat
                        )
            backward_bar.close()
            if candidate_aic + tol < best_aic:
                current_features = candidate_features
                best_model = candidate_model
                best_aic = candidate_aic
                remaining = [feat for feat in X.columns if feat not in current_features]
                improved = True
                if verbose:
                    tqdm.write(
                        f"Backward step -> {current_features} | AIC={best_aic:.2f}"
                    )

    return StepwiseResult(
        selected_features=current_features,
        model=None,
        fit_result=best_model,
        history=history,
    )


stepwise_result = stepwise_arimax_selection(
    y_train,
    X_train_scaled,
    order=BEST_ORDER,
    seasonal_order=BEST_SEASONAL_ORDER,
    must_have=MUST_KEEP_FEATURES,
    max_features=MAX_FEATURES_STEPWISE,
    tol=STEPWISE_TOL,
    verbose=True,
)

print(f"Selected features: {stepwise_result.selected_features}")

# %% [md]
# ## 7. Fit Final Model & Evaluate

# %%
selected_cols = list(stepwise_result.selected_features)
exog_train = X_train_scaled[selected_cols] if selected_cols else None
exog_test = X_test_scaled[selected_cols] if selected_cols else None

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

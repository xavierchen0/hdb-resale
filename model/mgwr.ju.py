# %% [md]
# # MGWR Modelling Across Years (2017–2025)
#
# This notebook-style script prepares yearly slices of the feature-enriched HDB resale dataset,
# prunes collinear predictors via VIF, scales the design matrix, and fits Multiscale Geographically
# Weighted Regression (MGWR) models for each calendar year from 2017 through 2025. Local parameter
# estimates and diagnostics are persisted for downstream spatial analysis.

# %% [md]
# ## 1. Imports and configuration
# - `YEARS`: modelling horizon.
# - `VIF_THRESHOLD`: maximum acceptable VIF before dropping a feature.
# - `MAX_SAMPLE`: optional cap on rows per year (set to `None` to keep all observations).
# - `OUTPUT_DIR`: where parquet/CSV outputs are written.

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DATA_DIR  # noqa: E402

YEARS: Sequence[int] = tuple(range(2017, 2026))
VIF_THRESHOLD: float = 10.0
MAX_SAMPLE: int | None = None
MIN_FEATURES: int = 2

OUTPUT_DIR = DATA_DIR / "model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [md]
# ## 2. Helper utilities
#
# - `clean_feature_frame`: keeps numeric predictors, drops constants, handles infs/NaNs.
# - `select_features_with_vif`: iteratively removes features above the VIF threshold.
# - `prepare_design_matrices`: coordinates the scaling, VIF filtering, and returns matrices.


# %%
@dataclass
class VIFResult:
    selected_idx: List[int]
    kept_features: List[str]
    vif_series: pd.Series
    dropped: List[Dict[str, object]]
    initial_count: int


def clean_feature_frame(df: pd.DataFrame, drop_cols: Iterable[str]) -> pd.DataFrame:
    feat_df = df.drop(columns=list(drop_cols), errors="ignore")
    feat_df = feat_df.select_dtypes(include=[np.number]).copy()
    feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat_df.dropna(axis=1, inplace=True)

    nunique = feat_df.nunique(dropna=True)
    non_constant_cols = nunique[nunique > 1].index
    feat_df = feat_df[non_constant_cols]

    return feat_df.astype(float)


def select_features_with_vif(
    X_scaled: np.ndarray,
    feature_names: List[str],
    threshold: float,
) -> VIFResult:
    remaining = list(range(X_scaled.shape[1]))
    dropped: List[Dict[str, object]] = []
    initial_count = len(feature_names)

    while remaining:
        X_curr = X_scaled[:, remaining]
        if X_curr.shape[1] <= 1:
            vif_vals = pd.Series(
                [np.nan] if X_curr.shape[1] == 1 else [],
                index=[feature_names[remaining[0]]] if X_curr.shape[1] == 1 else [],
                dtype=float,
            )
            return VIFResult(
                remaining.copy(),
                [feature_names[i] for i in remaining],
                vif_vals,
                dropped,
                initial_count,
            )

        try:
            vif_array = np.array(
                [variance_inflation_factor(X_curr, i) for i in range(X_curr.shape[1])],
                dtype=float,
            )
        except np.linalg.LinAlgError:
            corr = np.corrcoef(X_curr, rowvar=False)
            upper = np.triu(np.abs(corr), k=1)
            if not np.any(upper):
                drop_idx_local = 0
            else:
                flat_idx = int(np.argmax(upper))
                i, j = divmod(flat_idx, upper.shape[1])
                drop_idx_local = (
                    j if np.sum(np.abs(corr[j])) >= np.sum(np.abs(corr[i])) else i
                )
            drop_global_idx = remaining.pop(drop_idx_local)
            dropped.append(
                {
                    "feature": feature_names[drop_global_idx],
                    "vif": float("inf"),
                    "reason": "singular",
                }
            )
            continue

        max_vif = float(np.max(vif_array))
        if max_vif <= threshold:
            vif_series = pd.Series(
                vif_array, index=[feature_names[i] for i in remaining]
            )
            return VIFResult(
                remaining.copy(),
                [feature_names[i] for i in remaining],
                vif_series,
                dropped,
                initial_count,
            )

        drop_idx_local = int(np.argmax(vif_array))
        drop_global_idx = remaining.pop(drop_idx_local)
        dropped.append(
            {
                "feature": feature_names[drop_global_idx],
                "vif": max_vif,
                "reason": "above_threshold",
            }
        )

    raise ValueError("All features were removed during VIF selection.")


def prepare_design_matrices(
    gdf_year: gpd.GeoDataFrame,
    drop_cols: Iterable[str],
    vif_threshold: float,
) -> Tuple[np.ndarray, List[str], StandardScaler, VIFResult]:
    feature_df = clean_feature_frame(gdf_year, drop_cols)
    if feature_df.empty:
        raise ValueError("No numeric features available after cleaning.")

    scaler = StandardScaler()
    X_scaled_all = scaler.fit_transform(feature_df.to_numpy())
    feature_names = list(feature_df.columns)

    vif_result = select_features_with_vif(X_scaled_all, feature_names, vif_threshold)

    selected_matrix = X_scaled_all[:, vif_result.selected_idx]
    selected_features = [feature_names[i] for i in vif_result.selected_idx]

    if selected_matrix.shape[1] < MIN_FEATURES:
        raise ValueError(
            f"Insufficient features after VIF filtering (found {selected_matrix.shape[1]})."
        )

    scaler.mean_ = scaler.mean_[vif_result.selected_idx]
    scaler.scale_ = scaler.scale_[vif_result.selected_idx]

    return selected_matrix, selected_features, scaler, vif_result


# %% [md]
# ## 3. Load dataset and ensure CRS

# %%
full_gdf = gpd.read_parquet(DATA_DIR / "final_dataset.parquet")
if full_gdf.crs is None:
    full_gdf = full_gdf.set_crs("EPSG:3414")
else:
    full_gdf = full_gdf.to_crs("EPSG:3414")

full_gdf["month"] = pd.to_datetime(full_gdf["month"])
full_gdf["year"] = full_gdf["month"].dt.year

print(f"Loaded {len(full_gdf):,} records with {full_gdf.shape[1]} columns")

# %% [md]
# ## 4. Iterate through years, fit MGWR, and persist outputs

# %%
DROP_COLUMNS = {"txn_id", "resale_price", "month", "geometry", "year"}

run_summaries: List[Dict[str, object]] = []

for year in YEARS:
    gdf_year = full_gdf[full_gdf["year"] == year].copy()
    if gdf_year.empty:
        print(f"[skip] No data found for {year}")
        continue

    if MAX_SAMPLE is not None and len(gdf_year) > MAX_SAMPLE:
        gdf_year = gdf_year.sample(n=MAX_SAMPLE, random_state=42).sort_values("month")

    try:
        X_scaled, feature_names, scaler, vif_info = prepare_design_matrices(
            gdf_year, DROP_COLUMNS, VIF_THRESHOLD
        )
    except ValueError as err:
        print(f"[skip] {year}: {err}")
        continue

    coords = np.column_stack([gdf_year.geometry.x, gdf_year.geometry.y])
    y = np.log(gdf_year["resale_price"].to_numpy()).reshape(-1, 1)

    selector = Sel_BW(coords, y, X_scaled, multi=True, constant=True)
    bandwidths = selector.search()

    mgwr_model = MGWR(
        coords,
        y,
        X_scaled,
        selector,
        constant=True,
        kernel="bisquare",
        spherical=False,
    )
    mgwr_results = mgwr_model.fit()

    param_names = ["mgwr_intercept"] + [f"mgwr_{col}" for col in feature_names]
    se_names = ["mgwr_se_intercept"] + [f"mgwr_se_{col}" for col in feature_names]

    params = pd.DataFrame(
        mgwr_results.params, columns=param_names, index=gdf_year.index
    )
    std_errors = pd.DataFrame(
        mgwr_results.se_betas, columns=se_names, index=gdf_year.index
    )
    local_r2 = pd.Series(
        mgwr_results.localR2, name="mgwr_local_r2", index=gdf_year.index
    )

    export_cols = ["txn_id", "month", "resale_price", "geometry"] + feature_names
    export_gdf = gdf_year[export_cols].join(params).join(std_errors).join(local_r2)

    year_dir = OUTPUT_DIR / f"mgwr_{year}"
    year_dir.mkdir(exist_ok=True)

    export_gdf.to_parquet(year_dir / f"local_params_{year}.parquet")
    if not vif_info.vif_series.empty:
        vif_info.vif_series.to_csv(year_dir / f"vif_{year}.csv")
    pd.DataFrame(vif_info.dropped).to_csv(
        year_dir / f"vif_dropped_{year}.csv", index=False
    )

    run_summaries.append(
        {
            "year": year,
            "observations": len(gdf_year),
            "features_initial": vif_info.initial_count,
            "features_retained": len(feature_names),
            "features_dropped_vif": len(vif_info.dropped),
            "bandwidth_min": float(np.min(bandwidths)),
            "bandwidth_max": float(np.max(bandwidths)),
            "aicc": float(mgwr_results.aicc),
            "aic": float(mgwr_results.aic),
            "bic": float(mgwr_results.bic),
        }
    )

    print(
        f"[year {year}] obs={len(gdf_year):,}, features={len(feature_names)}, "
        f"bandwidth range=({np.min(bandwidths):.2f}, {np.max(bandwidths):.2f})"
    )

summary_df = pd.DataFrame(run_summaries)
summary_path = OUTPUT_DIR / "mgwr_run_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("\nMGWR run summary:")
print(summary_df)
print(f"Summary written to {summary_path}")

# %% [md]
# ### Notes
# - All predictors are z-scored before MGWR; interpret coefficients accordingly.
# - Inspect the saved `vif_*.csv` files to review multicollinearity diagnostics per year.
# - If any year emits singular warnings, consider tightening the feature set or lowering `MAX_SAMPLE`.

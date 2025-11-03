# %% [md]
# # MGWR Modelling Across Years (2017–2025)
#
# This notebook-style script prepares yearly slices of the feature-enriched HDB resale dataset,
# retains must-have predictors, prunes additional candidates via correlation filtering and VIF,
# scales the design matrix, applies stratified sampling by town to control dataset size, and fits
# Multiscale Geographically Weighted Regression (MGWR) models for each calendar year from 2017
# through 2024. Local parameter estimates and diagnostics are persisted for downstream spatial
# analysis.

# %% [md]
# ## Quick sanity test (optional)
# Set `TEST_MODE = True` to run a lightweight check on a single year.

# %% [md]
# ## 1. Imports and configuration

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
from tqdm.auto import tqdm

project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DATA_DIR  # noqa: E402

YEARS: Sequence[int] = tuple(range(2024, 2025))
MUST_HAVE_FEATURES: Sequence[str] = (
    "floor_area_sqm",
    "remaining_lease",
    "storey_median",
    "flat_type_2 ROOM",
    "flat_type_3 ROOM",
    "flat_type_4 ROOM",
    "flat_type_5 ROOM",
    "flat_type_EXECUTIVE",
    "distance_to_nearest_preschool",
    "distance_to_nearest_mrt_lrt",
    "distance_to_nearest_mall",
    "distance_to_nearest_prischool",
    "dist_to_cbd",
)
ADDITIONAL_FEATURE_LIMIT: int = 10
VIF_THRESHOLD: float = 10.0
MAX_SAMPLE: int | None = 9_500
MIN_FEATURES: int = len(MUST_HAVE_FEATURES) + 1
TEST_MODE: bool = False
TEST_YEAR: int = 2023

OUTPUT_DIR = DATA_DIR / "model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if TEST_MODE:
    YEARS = [TEST_YEAR]
    MAX_SAMPLE = 2_000

# %% [md]
# ## 2. Helper utilities


# %%
@dataclass
class VIFResult:
    kept_features: List[str]
    vif_series: pd.Series
    dropped: List[Dict[str, object]]
    numeric_count: int
    candidate_count: int


def clean_feature_frame(df: pd.DataFrame, drop_cols: Iterable[str]) -> pd.DataFrame:
    feat_df = df.drop(columns=list(drop_cols), errors="ignore")
    feat_df = feat_df.select_dtypes(include=[np.number]).copy()
    feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat_df.dropna(axis=1, inplace=True)
    nunique = feat_df.nunique(dropna=True)
    feat_df = feat_df[nunique[nunique > 1].index]
    return feat_df.astype(float)


def select_features_with_vif(
    X_scaled: np.ndarray,
    feature_names: List[str],
    must_have_set: set[str],
    threshold: float,
    numeric_count: int,
) -> VIFResult:
    remaining = list(range(X_scaled.shape[1]))
    dropped: List[Dict[str, object]] = []
    candidate_count = len(feature_names)

    while remaining:
        X_curr = X_scaled[:, remaining]
        curr_features = [feature_names[i] for i in remaining]

        if len(curr_features) <= len(must_have_set):
            vif_vals = pd.Series(np.nan, index=curr_features, dtype=float)
            return VIFResult(
                curr_features, vif_vals, dropped, numeric_count, candidate_count
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
            drop_global_idx = remaining[drop_idx_local]
            candidate = feature_names[drop_global_idx]
            if candidate in must_have_set:
                dropped.append(
                    {
                        "feature": candidate,
                        "vif": float("inf"),
                        "reason": "singular_must_keep",
                    }
                )
                remaining.pop(drop_idx_local)
                continue
            dropped.append(
                {"feature": candidate, "vif": float("inf"), "reason": "singular"}
            )
            remaining.pop(drop_idx_local)
            continue

        max_vif_idx = int(np.argmax(vif_array))
        max_vif = float(vif_array[max_vif_idx])
        candidate = curr_features[max_vif_idx]

        if candidate in must_have_set or max_vif <= threshold:
            if max_vif <= threshold:
                vif_series = pd.Series(vif_array, index=curr_features)
                return VIFResult(
                    curr_features, vif_series, dropped, numeric_count, candidate_count
                )
            else:
                remaining.pop(max_vif_idx)
                continue

        dropped.append(
            {"feature": candidate, "vif": max_vif, "reason": "above_threshold"}
        )
        remaining.pop(max_vif_idx)

    raise ValueError("All features were removed during VIF selection.")


def prepare_design_matrices(
    gdf_year: gpd.GeoDataFrame,
    drop_cols: Iterable[str],
    target: pd.Series,
    must_have: Sequence[str],
    extra_limit: int,
    vif_threshold: float,
) -> Tuple[np.ndarray, List[str], StandardScaler, VIFResult]:
    missing = [col for col in must_have if col not in gdf_year.columns]
    if missing:
        for col in missing:
            gdf_year[col] = 0
        tqdm.write(f"[info] Added missing must-have columns with zeros: {missing}")
    numeric_df = clean_feature_frame(gdf_year, drop_cols)
    if numeric_df.empty:
        raise ValueError("No numeric features available after cleaning.")

    must_have_set = {col for col in must_have if col in numeric_df.columns}
    missing_must_have = set(must_have) - must_have_set
    if missing_must_have:
        raise KeyError(f"Missing must-have features: {sorted(missing_must_have)}")

    extra_candidates = numeric_df.drop(columns=list(must_have_set), errors="ignore")
    numeric_count = numeric_df.shape[1]

    corr_scores = extra_candidates.apply(
        lambda col: col.corr(target, method="spearman")
    )
    corr_scores = corr_scores.abs().dropna().sort_values(ascending=False)
    extra_cols = corr_scores.index.tolist()[:extra_limit]

    feature_df = numeric_df[list(must_have_set) + extra_cols]

    scaler = StandardScaler()
    X_scaled_all = scaler.fit_transform(feature_df.to_numpy())
    feature_names = list(feature_df.columns)

    vif_result = select_features_with_vif(
        X_scaled_all,
        feature_names,
        must_have_set,
        vif_threshold,
        numeric_count,
    )

    selected_features = vif_result.kept_features
    selected_indices = [feature_names.index(col) for col in selected_features]
    X_selected = X_scaled_all[:, selected_indices]

    scaler.mean_ = scaler.mean_[selected_indices]
    scaler.scale_ = scaler.scale_[selected_indices]

    return X_selected, selected_features, scaler, vif_result


def stratified_sample_by_town(
    gdf_year: gpd.GeoDataFrame,
    max_sample: int,
    random_state: int = 42,
) -> gpd.GeoDataFrame:
    if max_sample is None or len(gdf_year) <= max_sample:
        return gdf_year

    towns = gdf_year["town"].astype("category")
    samples_per_town = np.floor(max_sample * towns.value_counts(normalize=True)).astype(
        int
    )
    samples_per_town = samples_per_town.clip(lower=1)

    sampled_frames = []
    rng = np.random.default_rng(random_state)

    for town, quota in samples_per_town.items():
        town_df = gdf_year[gdf_year["town"] == town]
        if len(town_df) <= quota:
            sampled_frames.append(town_df)
        else:
            sampled_frames.append(town_df.sample(n=quota, random_state=random_state))

    sampled = pd.concat(sampled_frames).sort_values("month")

    if len(sampled) > max_sample:
        sampled = sampled.sample(n=max_sample, random_state=random_state)

    return sampled.sort_values("month").reset_index(drop=True)


# %% [md]
# ## 3. Load dataset and ensure CRS

# %%
full_gdf = gpd.read_parquet(DATA_DIR / "final_dataset.parquet")
if full_gdf.crs is None:
    full_gdf = full_gdf.set_crs("EPSG:3414")
else:
    full_gdf = full_gdf.to_crs("EPSG:3414")

if "town" not in full_gdf.columns:
    base_meta = gpd.read_parquet(DATA_DIR / "base_dataset.parquet")[["txn_id", "town"]]
    base_meta = base_meta.drop_duplicates(subset="txn_id")
    full_gdf = full_gdf.merge(base_meta, on="txn_id", how="left")

full_gdf["month"] = pd.to_datetime(full_gdf["month"])
full_gdf["year"] = full_gdf["month"].dt.year

print(f"Loaded {len(full_gdf):,} records with {full_gdf.shape[1]} columns")

# %% [md]
# ## 4. Iterate through years, fit MGWR, and persist outputs

# %%
DROP_COLUMNS = {"txn_id", "resale_price", "month", "geometry", "year", "town"}

run_summaries: List[Dict[str, object]] = []

for year in tqdm(YEARS, desc="MGWR years"):
    gdf_year = full_gdf[full_gdf["year"] == year].copy()
    if gdf_year.empty:
        tqdm.write(f"[skip] {year}: No data found")
        continue

    if "town" not in gdf_year.columns:
        tqdm.write(f"[skip] {year}: missing 'town' column for stratified sampling")
        continue

    gdf_year = stratified_sample_by_town(gdf_year, MAX_SAMPLE)
    gdf_year = gdf_year.dropna(subset=list(MUST_HAVE_FEATURES) + ["resale_price"])
    if gdf_year.empty:
        tqdm.write(f"[skip] {year}: no rows after dropping NaNs in must-have features")
        continue

    target_log = np.log(gdf_year["resale_price"])

    try:
        X_scaled, selected_features, scaler, vif_info = prepare_design_matrices(
            gdf_year,
            DROP_COLUMNS,
            target_log,
            MUST_HAVE_FEATURES,
            ADDITIONAL_FEATURE_LIMIT,
            VIF_THRESHOLD,
        )
    except (ValueError, KeyError) as err:
        tqdm.write(f"[skip] {year}: {err}")
        continue

    tqdm.write(
        f"[year {year}] obs={len(gdf_year):,}, numeric={vif_info.numeric_count}, "
        f"candidates={vif_info.candidate_count}, kept={len(selected_features)}"
    )

    coords = np.column_stack([gdf_year.geometry.x, gdf_year.geometry.y])
    y = target_log.to_numpy().reshape(-1, 1)

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

    param_names = ["mgwr_intercept"] + [f"mgwr_{col}" for col in selected_features]
    se_names = ["mgwr_se_intercept"] + [f"mgwr_se_{col}" for col in selected_features]

    params = pd.DataFrame(
        mgwr_results.params, columns=param_names, index=gdf_year.index
    )
    std_errors = pd.DataFrame(
        mgwr_results.se_betas, columns=se_names, index=gdf_year.index
    )
    local_r2 = pd.Series(
        mgwr_results.localR2, name="mgwr_local_r2", index=gdf_year.index
    )

    export_cols = [
        "txn_id",
        "month",
        "resale_price",
        "geometry",
        "town",
    ] + selected_features
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
            "features_numeric": vif_info.numeric_count,
            "features_candidates": vif_info.candidate_count,
            "features_retained": len(selected_features),
            "bandwidth_min": float(np.min(bandwidths)),
            "bandwidth_max": float(np.max(bandwidths)),
            "aicc": float(mgwr_results.aicc),
            "aic": float(mgwr_results.aic),
            "bic": float(mgwr_results.bic),
        }
    )

    tqdm.write(
        f"[done {year}] bandwidth range=({np.min(bandwidths):.2f}, {np.max(bandwidths):.2f}), "
        f"AICc={mgwr_results.aicc:.2f}"
    )

summary_df = pd.DataFrame(run_summaries)
summary_path = OUTPUT_DIR / "mgwr_run_summary.csv"
summary_df.to_csv(summary_path, index=False)

tqdm.write("\nMGWR run summary:")
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    tqdm.write(summary_df.to_string(index=False))
tqdm.write(f"Summary written to {summary_path}")

# %% [md]
# ### Notes
# - All predictors are z-scored before MGWR; interpret coefficients accordingly.
# - Inspect the saved `vif_*.csv` files to review multicollinearity diagnostics per year.
# - Stratified sampling ensures each town is represented while capping the per-year dataset size.

# %% [md]
# # Merge all the features into one dataset

# %%
import pandas as pd
import glob
import geopandas as gpd
from pathlib import Path
import sys
import gc
import pyarrow.parquet as pq
from tqdm.auto import tqdm

# %% [md]
# Load main dataset


# %%
project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DATA_DIR, FEAT_DIR

MAIN_DATA_PATH = Path(DATA_DIR / "base_dataset.parquet")
feature_folder = FEAT_DIR

# Load the main dataset
try:
    gdf_merged = gpd.read_parquet(MAIN_DATA_PATH)
    print(f" Initialized with Main Dataset: {MAIN_DATA_PATH.name}")
except FileNotFoundError:
    print(f"Error: Main dataset not found at {MAIN_DATA_PATH}")
    # Handle the error, maybe exit or return an empty DataFrame
    gdf_merged = pd.DataFrame()

# %%
import pandas as pd
import geopandas as gpd
import glob
from pathlib import Path
from tqdm.auto import tqdm  # Import tqdm for the progress bar

# The target CRS for Singapore (SVY21)
TARGET_CRS = "EPSG:3414"

# --- Assuming feature_folder is correctly set based on your first block ---
# -----------------------------------------------------------------------

# Define the two merge keys
FIRST_KEY = "month"
SECOND_KEY = "txn_id"

if not gdf_merged.empty:
    print(f"Current CRS of main dataset: {gdf_merged.crs}")

    # 1. Get a list of all feature parquet files
    feature_files = glob.glob(str(feature_folder / "feat_*.parquet"))
    feature_files = [f for f in feature_files if Path(f).name != "base_dataset.parquet"]

    print(
        f"\nStarting two-stage feature merging (Key 1: '{FIRST_KEY}', Key 2: '{SECOND_KEY}')..."
    )

    # 2. Iterate and merge each feature file

    for file_path in tqdm(feature_files, desc="Merging Features"):
        file_name = Path(file_path).name
        df_feature = None
        temp_merged = None

        try:
            parquet_file = pq.ParquetFile(file_path)
            feature_columns = parquet_file.schema.names
        except Exception as meta_err:
            tqdm.write(f"Skipped {file_name}: Unable to read metadata ({meta_err}).")
            continue

        has_first_key = FIRST_KEY in feature_columns
        has_second_key = SECOND_KEY in feature_columns

        if not has_first_key and not has_second_key:
            tqdm.write(
                f"Skipped {file_name}: Missing both merge keys ('{FIRST_KEY}' and '{SECOND_KEY}')."
            )
            continue

        current_cols = set(gdf_merged.columns)
        value_columns = [
            col for col in feature_columns if col not in {FIRST_KEY, SECOND_KEY}
        ]
        new_value_columns = [col for col in value_columns if col not in current_cols]

        columns_to_read = []
        if has_first_key:
            columns_to_read.append(FIRST_KEY)
        if has_second_key and SECOND_KEY not in columns_to_read:
            columns_to_read.append(SECOND_KEY)
        columns_to_read.extend(new_value_columns)

        if not new_value_columns:
            tqdm.write(f"Skipped {file_name}: No new columns to merge.")
            continue

        try:
            if "geometry" in columns_to_read:
                df_feature = gpd.read_parquet(file_path, columns=columns_to_read)
                if getattr(df_feature, "crs", None) and df_feature.crs != TARGET_CRS:
                    df_feature = df_feature.to_crs(TARGET_CRS)
            else:
                df_feature = pd.read_parquet(file_path, columns=columns_to_read)
        except Exception as read_err:
            tqdm.write(f"Error loading {file_name}: {read_err}. Skipping.")
            continue

        try:
            if has_first_key and has_second_key:
                df_txn = df_feature.drop_duplicates(subset=[SECOND_KEY])
                temp_merged = gdf_merged.merge(
                    right=df_txn,
                    on=SECOND_KEY,
                    how="left",
                    suffixes=("", "_feat_txn"),
                    copy=False,
                )

                unmatched_mask = temp_merged[new_value_columns].isna().all(axis=1)
                if unmatched_mask.any():
                    month_lookup = (
                        df_feature[[FIRST_KEY] + new_value_columns]
                        .drop_duplicates(subset=[FIRST_KEY])
                        .set_index(FIRST_KEY)
                    )
                    month_values = month_lookup.reindex(
                        temp_merged.loc[unmatched_mask, FIRST_KEY]
                    )
                    for col in new_value_columns:
                        temp_merged.loc[unmatched_mask, col] = month_values[
                            col
                        ].to_numpy()

                gdf_merged = temp_merged

            elif has_first_key:
                df_feature = df_feature.drop_duplicates(subset=[FIRST_KEY])
                gdf_merged = gdf_merged.merge(
                    right=df_feature,
                    on=FIRST_KEY,
                    how="left",
                    suffixes=("", "_feat"),
                    copy=False,
                )

            else:
                df_feature = df_feature.drop_duplicates(subset=[SECOND_KEY])
                gdf_merged = gdf_merged.merge(
                    right=df_feature,
                    on=SECOND_KEY,
                    how="left",
                    suffixes=("", "_feat_txn"),
                    copy=False,
                )

            tqdm.write(f"➡️ Merged feature: {file_name}")

        except MemoryError:
            tqdm.write(
                f"General Error during merge of {file_name}: **MemoryError**. Skipping to prevent crash."
            )
            continue
        except Exception as merge_err:
            tqdm.write(
                f"General Error during merge of {file_name}: {merge_err}. Skipping."
            )
            continue
        finally:
            df_feature = None
            temp_merged = None
            gc.collect()

        # Clean up any duplicate columns that may have resulted from the merge attempts
        cols_to_drop = [
            col for col in gdf_merged.columns if col.endswith(("_feat", "_feat_txn"))
        ]
        if cols_to_drop:
            gdf_merged = gdf_merged.drop(columns=cols_to_drop, errors="ignore")

    # 3. Final CRS standardization
    if gdf_merged.crs != TARGET_CRS:
        gdf_merged = gdf_merged.to_crs(TARGET_CRS)
        print(f"\n✅ Reprojected final GeoDataFrame to {TARGET_CRS}")

    # 4. Final summary
    print("\n--- Final Merged GeoDataFrame Summary ---")
    print(f"Total Columns: {len(gdf_merged.columns)}")
    print(f"Total Rows: {len(gdf_merged)}")
    print(f"Final CRS: {gdf_merged.crs.to_string()}")
    gdf_merged.head()

# %%
# 1. Get the list of all feature files (excluding the base) again
feature_files = glob.glob(str(feature_folder / "feat_*.parquet"))
feature_files = [f for f in feature_files if Path(f).name != "base_dataset.parquet"]

# 2. Initialize counts and column lists
base_cols = set(gdf_merged.columns)
num_base_cols = len(base_cols)
expected_new_cols = set()

# 3. Loop through feature files to calculate EXPECTED new columns
print("\n--- Feature Column Validation ---")
for file_path in tqdm(feature_files, desc="Counting Expected Columns"):
    file_name = Path(file_path).name

    try:
        parquet_file = pq.ParquetFile(file_path)
        feature_cols = set(parquet_file.schema.names)
        expected_new_cols.update(feature_cols - {FIRST_KEY, SECOND_KEY})
    except Exception as e:
        tqdm.write(f"Error counting columns in {file_name}: {e}. Skipping.")

# 4. Calculate final expected and actual counts
num_expected_total_cols = num_base_cols + len(expected_new_cols - base_cols)
num_actual_total_cols = len(gdf_merged.columns)
num_actual_added_cols = num_actual_total_cols - num_base_cols


print(f"\n Total columns in MAIN dataset (before merge): {num_base_cols}")
print(f" Total UNIQUE columns expected from all features: {len(expected_new_cols)}")
print("-------------------------------------------------")
print(f" EXPECTED TOTAL columns after merge: **{num_expected_total_cols}**")
print(f" ACTUALLY ADDED columns (Count): **{num_actual_added_cols}**")
print(f" ACTUAL TOTAL columns in final GeoDataFrame: **{num_actual_total_cols}**")

# 5. Check for discrepancies
if num_actual_total_cols == num_expected_total_cols:
    print("\n Validation SUCCESS: Actual column count matches expected count.")
else:
    print(
        f"\n Validation WARNING: Actual column count ({num_actual_total_cols}) does NOT match expected count ({num_expected_total_cols})."
    )
    print(
        "This usually means some columns were not added due to name collisions or data type issues."
    )

# %%
# Set display options to show all columns without truncation
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


print("Displaying the first 10 rows of the merged GeoDataFrame:")
gdf_merged

# %% [md]
# Convert `storey_range` to the median of the lower and upper limit
# %%
gdf_merged["storey_median"] = (
    gdf_merged["storey_range"]
    .str.extractall(r"(\d+)")[0]
    .groupby(level=0)
    .apply(lambda x: x.astype(int).median())
)
gdf_merged[["storey_range", "storey_median"]]

# %% [md]
# One-hot encode `flat_type` for downstream models
# <br>
# and drop `flat_type_1 ROOM`
# %%
flat_type_dummies = pd.get_dummies(
    gdf_merged["flat_type"],
    prefix="flat_type",
    dtype="uint8",
)
gdf_merged = gdf_merged.join(flat_type_dummies)
gdf_merged = gdf_merged.drop(columns=["flat_type", "flat_type_1 ROOM"])
del flat_type_dummies
gc.collect()

# %% [md]
# Drop a few unnecessary cols
# %%
gdf_merged = gdf_merged.drop(
    columns=[
        "block",
        "street_name",
        "flat_model",
        "lease_commence_date",
        "postal_code",
        "x",
        "y",
        "latitude",
        "longitude",
        "__batch__",
        "storey_range",
    ]
)

# %% [md]
# Keep a few cols

# %%
gdf_merged = gdf_merged[
    [
        "txn_id",
        "month",
        "floor_area_sqm",
        "resale_price",
        "remaining_lease",
        "distance_to_nearest_park",
        "parks_access",
        "ppi_non_landed",
        "coe_a",
        "coe_b",
        "marriage_rate",
        "inc_mem_pct_change",
        "distance_to_nearest_mrt_lrt",
        "distance_to_nearest_sports",
        "sports_access",
        "distance_to_nearest_prischool",
        "prischool_access",
        "distance_to_nearest_foodcentre",
        "foodcentre_access",
        "birth_rate",
        "distance_to_nearest_hawker",
        "hawker_access",
        "distance_to_nearest_secschool",
        "secschool_access",
        "distance_to_nearest_mall",
        "mall_access",
        "distance_to_nearest_preschool",
        "preschool_access",
        "unemp_rate",
        "cpi_index",
        "distance_to_nearest_cc",
        "cc_access",
        "sora",
        "ssd_max_rate",
        "ssd_max_hold_yrs",
        "bsd_max_rate_res",
        "sg_absd_2",
        "sg_absd_3",
        "pr_absd_1",
        "pr_absd_2",
        "pr_absd_3",
        "tdsr",
        "storey_median",
        "dist_to_cbd",
    ]
]

# %% [md]
# final set of cols

# %%
pd.set_option("display.max_rows", None)
pd.DataFrame(gdf_merged.dtypes)

# %%
gc.collect()
gdf_merged.to_parquet(DATA_DIR / "final_dataset.parquet")

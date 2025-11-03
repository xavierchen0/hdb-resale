# %% [md]
# # Merge all the features into one dataset

# %%
import pandas as pd
import glob
import numpy as np
import geopandas as gpd
from pathlib import Path
import sys
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

        try:
            # --- Load Data with Fallback ---
            try:
                df_feature = gpd.read_parquet(file_path)
                df_feature = df_feature.to_crs(TARGET_CRS)
            except Exception:
                df_feature = pd.read_parquet(file_path)

            if df_feature is None:
                continue

            # --- DETERMINE MERGE STRATEGY ---

            # Case 1: Has MONTH (Time-series features). Requires two-stage logic.
            if FIRST_KEY in df_feature.columns:
                # --- STAGE 1 MERGE: By 'month' ---
                temp_merged = gdf_merged.merge(
                    right=df_feature,
                    on=FIRST_KEY,
                    how="left",
                    suffixes=("", "_feat"),
                    indicator=True,
                )

                # Identify rows that were NOT merged successfully by month
                # and have the SECOND_KEY for a second attempt
                if SECOND_KEY in df_feature.columns:
                    # Rows in main data that failed to match on month
                    gdf_unmatched_mask = temp_merged["_merge"] == "left_only"
                    gdf_unmatched = gdf_merged[gdf_unmatched_mask].copy()

                    # Rows that were matched in Stage 1
                    gdf_matched = temp_merged[
                        temp_merged["_merge"] != "left_only"
                    ].drop(columns=["_merge"])

                    # --- STAGE 2 MERGE: By 'txn_id' on UNMATCHED rows ---
                    gdf_unmatched_merged = gdf_unmatched.merge(
                        right=df_feature,
                        on=SECOND_KEY,
                        how="left",
                        suffixes=("", "_feat_txn"),
                    )

                    # Final combine: Matched + Second-stage Merged
                    gdf_merged = pd.concat(
                        [gdf_matched, gdf_unmatched_merged]
                    ).drop_duplicates(subset=[SECOND_KEY, FIRST_KEY])

                else:
                    # If SECOND_KEY is missing, just use the month merge result
                    gdf_merged = temp_merged.drop(columns=["_merge"])

            # Case 2: Missing MONTH, but has TXN_ID (Location features)
            elif SECOND_KEY in df_feature.columns:
                # Direct merge by txn_id
                gdf_merged = gdf_merged.merge(
                    right=df_feature,
                    on=SECOND_KEY,
                    how="left",
                    suffixes=("", "_feat_txn"),
                )

            else:
                tqdm.write(
                    f"Skipped {file_name}: Missing both merge keys ('{FIRST_KEY}' and '{SECOND_KEY}')."
                )
                continue

            tqdm.write(f"➡️ Merged feature: {file_name}")

        except MemoryError:
            tqdm.write(
                f"General Error during merge of {file_name}: **MemoryError**. Skipping to prevent crash."
            )
            continue  # Skip to the next file

        except Exception as e:
            tqdm.write(f"General Error during merge of {file_name}: {e}. Skipping.")

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
        # Load as Pandas DataFrame for fast metadata check
        df_feature = pd.read_parquet(file_path)

        # Identify columns that are NOT the merge keys ('month', 'txn_id')
        new_cols = set(df_feature.columns) - {FIRST_KEY, SECOND_KEY}

        # Add new unique columns to the expected set
        expected_new_cols.update(new_cols)

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

# %%
df1 = pd.read_parquet(FEAT_DIR / "feat_coe.parquet")
df1

# %%
# Assuming df1 has columns: month, coe_a, coe_b, coe_c, coe_d, coe_e
# 1. Identify the columns to aggregate (all non-month columns)
coe_cols = [col for col in df1.columns if col != "month"]

# 2. Aggregate df1 to one row per month
# We use .mean() to collapse the transaction-level data into monthly averages.
df_coe_agg = df1.groupby("month", as_index=False)[coe_cols].mean()

print(f"Aggregated COE data reduced from {len(df1)} rows to {len(df_coe_agg)} rows.")

gdf_merged = gdf_merged.merge(right=df_coe_agg, on="month", how="left")

print(f"✅ Successfully merged COE features based on 'month'.")
print(f"Final column count: {len(gdf_merged.columns)}")
gdf_merged.head()

# %%
gdf_merged.to_parquet(DATA_DIR / "final_dataset.parquet")

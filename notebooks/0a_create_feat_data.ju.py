# %% [md]
# # Purpose
# This Notebook serves to create the feature dataset that will be used for modelling.
#
# This is part 1 because the conversion to coordinates takes a long time

# %% [md]
# # Libraries and Setup

# %%
import asyncio
import sys
from pathlib import Path
import math

from tqdm.notebook import tqdm
import pandas as pd
import numpy as np

# %%

# Allow importing from src/
project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import src.open_map_utils as open_map_utils
import src.utils as utils
from src.config import DATA_DIR

# %% [md]
# # Below cells will outline the data transformations performed

# %% [md]
# **Note**: See `0_basic_eda.ju.py` for basic insights and transformations on the HDB Resale dataset.
# ## Import hdb resale dataset

# %%
HDB_TXN_DATA_FP = DATA_DIR / "data.csv"

raw_df = pd.read_csv(HDB_TXN_DATA_FP)
raw_df

# %% [md]
# ## Reading the right data types for each column

# %%
# Read the csv into dataframe
df1 = pd.read_csv(
    HDB_TXN_DATA_FP,
    dtype={
        "town": "category",
        "flat_type": "category",
        "block": "category",
        "street_name": "category",
        "storey_range": "category",
        "floor_area_sqm": "float32",
        "flat_model": "category",
        "remaining_lease": "string",
        "resale_price": "float32",
    },
    parse_dates=["month", "lease_commence_date"],
)
df1.dtypes

# %% [md]
# ## Remove the month October 2025 (2025-10)

# %%
print(f"Initial No. of Observations: {len(df1):,}")

EXCLUDE_MONTH_STRING = "2025-10"

CNT_EXCLUDE_MONTH_ROWS = len(df1[df1["month"] == EXCLUDE_MONTH_STRING])

print(f"Number of rows with {EXCLUDE_MONTH_STRING}: {CNT_EXCLUDE_MONTH_ROWS}")

df1 = df1[df1["month"] != EXCLUDE_MONTH_STRING].copy()
df1.reset_index(drop=True, inplace=True)
print(f"No. of Observations after excluding {EXCLUDE_MONTH_STRING}: {len(df1):,}")

df1.dtypes

# %% [md]
# ## Change `remaining_lease` to months

# %%
# Split the col `remaining_lease`
df1[["year1", "year2", "month1", "month2"]] = df1["remaining_lease"].str.split(
    " ", expand=True
)

# Convert months with `null` with `0`
df1["month1"] = df1["month1"].fillna("0")

# Drop unnecessary cols
df1 = df1.drop(["year2", "month2", "remaining_lease"], axis=1)

# Convert cols "year1" and "month1" to integers
df1["year1"] = df1["year1"].astype("int32")
df1["month1"] = df1["month1"].astype("int32")

# Get col "remaining_lease" as col of months
df1["remaining_lease"] = df1["year1"] * 12 + df1["month1"]

# Drop unncessary cols
df1 = df1.drop(["year1", "month1"], axis=1)

df1

# %%
df1.dtypes

# %% [md]
# ## Deduplication

# %%
df1.sort_values(by="month", ascending=True, inplace=True, ignore_index=True)
df1 = df1.drop_duplicates(ignore_index=True)  # Keep first occurrence

# %% [md]
# ## Create a column for combined address

# %%
df1.loc[:, "full_addr"] = (
    df1["block"].astype(str) + " " + df1["street_name"].astype(str)
)
df1

# %% [md]
# ## Add postal_code, x, y, latitude, longitude columns

# %% [md]
# Call OpenMap API to get postal code, x, y, latitude and longitude
#
# Perfom caching to deal with network issues

# %%
# Get list of addresses
addresses = df1["full_addr"].astype(str).tolist()
chunk_size = 5_000
out_dir = DATA_DIR / "openmap_batches"
out_dir.mkdir(exist_ok=True)

done = {int(p.stem.split("_")[-1]) for p in out_dir.glob("part_*.pkl")}
total_batches = math.ceil(len(addresses) / chunk_size)

pbar = tqdm(total=total_batches, desc="OneMap batches")
pbar.update(len(done))  # reflect resumed progress

for batch_idx, chunk in utils.chunked(addresses, chunk_size):
    if batch_idx in done:
        continue

    try:
        batch_df = await open_map_utils.many_openmap_search(chunk)
        # optional: tag batch number for debugging
        batch_df["__batch__"] = batch_idx

        # Save this chunk
        part_path = out_dir / f"part_{batch_idx:05d}.pkl"
        batch_df.to_pickle(part_path)

    except Exception as e:
        print(f"Batch {batch_idx} failed: {e}")
    finally:
        pbar.update(1)

pbar.close()

# %% [md]
# Load back all the chunks

# %%
parts = sorted(out_dir.glob("part_*.pkl"))

pieces = [pd.read_pickle(p) for p in parts]
addresses_coordinates = pd.concat(pieces, ignore_index=True)
addresses_coordinates

# %% [md]
# Check for null in "full_addr"

# %%
addresses_coordinates[addresses_coordinates["full_addr"].isnull()]

# %% [md]
# Check for zero result and multiple results and NA

# %%
failed = addresses_coordinates[
    addresses_coordinates["x"].apply(
        lambda v: (isinstance(v, list) and (len(v) > 1 or len(v) == 0) or v is pd.NA)
    )
]
failed

# %% [md]
# Fix NA rows first

# %% [md]
# Get na rows

# %%
na_rows = failed[failed.isna().any(axis=1)].reset_index()
na_rows

# %% [md]
# Get results of na rows

# %%
na_rows_addrs = na_rows["full_addr"].to_list()
na_rows_result = await open_map_utils.many_openmap_search(na_rows_addrs)
na_rows_result

# %% [md]
# Combine with na rows dataframe

# %%
na_rows_merged = pd.concat(
    [
        na_rows.drop(
            [
                "__batch__",
                "full_addr",
                "postal_code",
                "x",
                "y",
                "latitude",
                "longitude",
            ],
            axis=1,
        ).reset_index(drop=True),
        na_rows_result.reset_index(drop=True),
    ],
    axis=1,
)
na_rows_merged = na_rows_merged.set_index("index")
na_rows_merged

# %% [md]
# Check that there is no na in na_rows_merged

# %%
na_rows_merged[na_rows_merged.isna().any(axis=1)]

# %% [md]
# Merged with addresses_coordinates dataframe

# %%
addresses_coordinates = addresses_coordinates.rename_axis(index="index")
addresses_coordinates.update(na_rows_merged)

# %% [md]
# Check failed again

# %% [md]
# failed should have no na

# %%
failed_no_na = addresses_coordinates[
    addresses_coordinates["x"].apply(
        lambda v: (isinstance(v, list) and (len(v) > 1 or len(v) == 0) or v is pd.NA)
    )
]
failed_no_na[failed_no_na.isna().any(axis=1)]

# %% [md]
# fix multiple result by first checking number of unique full_addr

# %%
pd.reset_option("display.max_rows", None)
failed_no_na

# %%
failed_no_na["full_addr"].unique()

# %% [md]
# update problematic full addr

# %%
# 8 BOON KENG RD
failed_no_na.loc[
    failed_no_na["full_addr"] == "8 BOON KENG RD",
    ["postal_code", "x", "y", "latitude", "longitude"],
] = [
    "330008",
    "31136.5014144294",
    "33189.1117452513",
    "1.31642489627751",
    "103.861501638313",
]

# 1E CANTONMENT RD
failed_no_na.loc[
    failed_no_na["full_addr"] == "1E CANTONMENT RD",
    ["postal_code", "x", "y", "latitude", "longitude"],
] = [
    "085501",
    "28872.2574625267",
    "28787.8612435274",
    "1.27662168290503",
    "103.841156133831",
]

# 1D CANTONMENT RD
failed_no_na.loc[
    failed_no_na["full_addr"] == "1D CANTONMENT RD",
    ["postal_code", "x", "y", "latitude", "longitude"],
] = [
    "085401",
    "28908.9450131785",
    "28820.8735950608",
    "1.27692023401394",
    "103.841485785921",
]

# 1 TOH YI DR
failed_no_na.loc[
    failed_no_na["full_addr"] == "1 TOH YI DR",
    ["postal_code", "x", "y", "latitude", "longitude"],
] = [
    "591501",
    "21473.9589902703",
    "35748.6328935493",
    "1.33957175314186",
    "103.774678246393",
]

# 208 BT BATOK ST 21
failed_no_na.loc[
    failed_no_na["full_addr"] == "208 BT BATOK ST 21",
    ["postal_code", "x", "y", "latitude", "longitude"],
] = [
    "650208",
    "18595.3245478645",
    "36448.8063004576",
    "1.34590310545134",
    "103.748811814397",
]

# 5 BOON KENG RD
failed_no_na.loc[
    failed_no_na["full_addr"] == "5 BOON KENG RD",
    ["postal_code", "x", "y", "latitude", "longitude"],
] = [
    "330005",
    "31013.6175472504",
    "33175.2953792606",
    "1.31629995826281",
    "103.860397463054",
]

failed_no_na

# %% [md]
# Check failed_no_na all conditions are met

# %%
failed_no_na[
    failed_no_na["x"].apply(
        lambda v: (isinstance(v, list) and (len(v) > 1 or len(v) == 0) or v is pd.NA)
    )
]

# %% [md]
# Merge with addresses_coordinates

# %%
addresses_coordinates.update(failed_no_na)

# %% [md]
# Verify

# %%
addresses_coordinates.loc[[212425, 212461, 214347]]

# %% [md]
# Extract from list

# %%
addresses_coordinates[addresses_coordinates["x"].apply(lambda v: isinstance(v, list))]

# %%
# postal_code
addresses_coordinates["postal_code"] = addresses_coordinates["postal_code"].apply(
    lambda v: v[0] if isinstance(v, list) else v
)

# x
addresses_coordinates["x"] = addresses_coordinates["x"].apply(
    lambda v: v[0] if isinstance(v, list) else v
)

# y
addresses_coordinates["y"] = addresses_coordinates["y"].apply(
    lambda v: v[0] if isinstance(v, list) else v
)

# latitude
addresses_coordinates["latitude"] = addresses_coordinates["latitude"].apply(
    lambda v: v[0] if isinstance(v, list) else v
)

# longitude
addresses_coordinates["longitude"] = addresses_coordinates["longitude"].apply(
    lambda v: v[0] if isinstance(v, list) else v
)

# Check
addresses_coordinates[
    addresses_coordinates["postal_code"].apply(lambda v: isinstance(v, list))
]

# %%
addresses_coordinates

# %% [md]
# merge with main df

# %%
final_df = pd.concat(
    [
        df1.reset_index(drop=True),
        addresses_coordinates.drop(columns=["full_addr"]).reset_index(drop=True),
    ],
    axis=1,
)
final_df

# %% [md]
# Set the right datatype

# %%
final_df.dtypes

# %%
final_df = final_df.astype(
    {
        "full_addr": "category",
        "postal_code": "category",
        "x": "float32",
        "y": "float32",
        "latitude": "float32",
        "longitude": "float32",
    }
)
final_df.dtypes


# %% [md]
# # Export partially completed file

# %%
final_df.to_parquet(DATA_DIR / "partial_dataset_1.parquet")

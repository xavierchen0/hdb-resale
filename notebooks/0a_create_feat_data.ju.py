# %% [md]
# # Purpose
# This Notebook serves to create the feature dataset that will be used for modelling.

# %% [md]
# # Libraries and Setup

# %%
import pandas as pd
import numpy as np
import geopandas

import sys
from pathlib import Path

# Allow importing from src/
project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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

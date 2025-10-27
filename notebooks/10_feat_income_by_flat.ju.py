# %% [md]
# # Feature Engineering:
# 1. Real Change in Average Monthly Household Employment Income (Including Employer CPF Contributions) Among Resident Employed Households by Type of Dwelling (Household Employment Income, Annual 2000-2024)

# %% [md]
# Read the base dataset

# %%
from os import stat_result
import pandas as pd
from pathlib import Path
import sys
import geopandas as gpd
from pandas.core.base import ensure_wrapped_if_datetimelike
import seaborn as sns
import matplotlib.pyplot as plt

# Allow importing from src/
project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DATA_DIR

df = gpd.read_parquet(DATA_DIR / "base_dataset.parquet")
df

# %% [md]
# Read the household income dataset

# %%
HOUSEHOLD_INCOME_PER_TYPE = DATA_DIR / "household_income_per_flat_type.csv"

df_hi_type = pd.read_csv(HOUSEHOLD_INCOME_PER_TYPE)
df_hi_type

# %% [md]
# Drop unneeded rows

# %%
df_long = df_hi_type.iloc[:-6]
df_long

# %% [md]
# Convert to the right data type, and rename PerCent column to month

# %%
df_long["PerCent"] = pd.to_datetime(df_long["PerCent"])
for col in df_long.columns:
    if col != "PerCent":
        df_long[col] = df_long[col].astype("Float64")

df_long = df_long.rename(columns={"PerCent": "month"})
df_long.dtypes


# %% [md]
# Add missing year, 2025, and assume the last year value
# <br>
# Value is not provided yet by the government

# %%
new_rows = pd.DataFrame([{"month": pd.to_datetime("2025-12-31")}])
df_long = pd.concat([df_long, new_rows]).ffill()
df_long

# %% [md]
# Convert to months
# <br>
# Keep the same value throughout the year
# <br>
# because using other interpolation techniques means we imply a pattern
# in real household income change throughout the year which may not be true

# %%
df_long = df_long.set_index("month").sort_index()

df_long = df_long.resample("MS").ffill()
df_long

# %% [md]
# Plot

# %%
df_long.plot()

# %% [md]
# Add 2, 3, 4, 5 year moving average for each flat type

# %%
flat_types = df_long.columns

for col in flat_types:
    df_long[f"income_type_yoy_ma2_{col}"] = df_long[col].rolling(2 * 12).mean()
    df_long[f"income_type_yoy_ma3_{col}"] = df_long[col].rolling(3 * 12).mean()
    df_long[f"income_type_yoy_ma4_{col}"] = df_long[col].rolling(4 * 12).mean()
    df_long[f"income_type_yoy_ma5_{col}"] = df_long[col].rolling(5 * 12).mean()
df_long

# %% [md]
# Add 2, 3, 4, 5 year rolling standard deviation

# %%
for col in flat_types:
    df_long[f"income_type_yoy_std2_{col}"] = df_long[col].rolling(2 * 12).std()
    df_long[f"income_type_yoy_std3_{col}"] = df_long[col].rolling(3 * 12).std()
    df_long[f"income_type_yoy_std4_{col}"] = df_long[col].rolling(4 * 12).std()
    df_long[f"income_type_yoy_std5_{col}"] = df_long[col].rolling(5 * 12).std()
df_long

# %% [md]
# Filter for the correct dates

# %%
START_DATE = "2017-01-01"
END_DATE = "2025-10-01"
START_DATE = pd.to_datetime(START_DATE)
END_DATE = pd.to_datetime(END_DATE)

df_long = df_long.loc[START_DATE:END_DATE]
df_long

# %% [md]
# Drop october 2025

# %%
df_long = df_long.drop(pd.to_datetime("2025-10-01"))
df_long

# %% [md]
# Check for null

# %%
df_long[df_long.isna().any(axis=1)]

# %% [md]
# Check dtype

# %%
df_long.dtypes

# %% [md]
# reset index

# %%
df_long = df_long.reset_index()

# %% [md]
# Final Check

# %%
df_long

# %% [md]
# # Export

# %%
df_long.to_parquet(DATA_DIR / "feat" / "feat_income_flat_type.parquet")

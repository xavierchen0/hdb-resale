# %% [md]
# # Feature Engineering:
# 1. Real Change in Monthly Household Employment Income Per Household Member (Including Employer CPF Contributions) Among Resident Employed Households at Selected Percentiles (Household Employment Income, Annual 2000-2024)

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
# Read the household income datasets

# %%
HOUSEHOLD_INCOME_PER_MEMBER = DATA_DIR / "household_income_per_member.csv"

df_hi_mem = pd.read_csv(HOUSEHOLD_INCOME_PER_MEMBER)
df_hi_mem

# %% [md]
# Household incomer per member

# %%
df_long = df_hi_mem.melt(
    id_vars="PerCent",
    value_vars=[str(y) for y in range(2001, 2025)],
    var_name="year",
    value_name="inc_mem_pct_change",
)
df_long

# %% [md]
# Convert to the right data type

# %%
df_long["PerCent"] = df_long["PerCent"].astype("category")
df_long["year"] = pd.to_datetime(df_long["year"])
df_long.dtypes

# %% [md]
# Keep only the median

# %%
df_long_median = df_long[df_long["PerCent"] == "50th (Median)"].drop(columns="PerCent")
df_long_median

# %% [md]
# Add missing year, 2025, and assume the last year value
# <br>
# Value is not provided yet by the government

# %%
new_rows = pd.DataFrame(
    [{"year": pd.to_datetime("2025-12-31"), "inc_mem_pct_change": 0.8}]
)
df_long_median = pd.concat([df_long_median, new_rows])
df_long_median

# %% [md]
# Convert to months
# <br>
# Keep the same value throughout the year
# <br>
# because using other interpolation techniques means we imply a pattern
# in real household income change throughout the year which may not be true

# %%
df_long_median = (
    df_long_median.set_index("year").sort_index().rename_axis(index="month")
)

df_long_median = df_long_median.resample("MS").ffill()
df_long_median

# %% [md]
# Plot

# %%
df_long_median.plot()

# %% [md]
# Add 2, 3, 4, 5 year moving average

# %%
df_long_median["income_mem_yoy_ma2"] = (
    df_long_median["inc_mem_pct_change"].rolling(2 * 12).mean()
)
df_long_median["income_mem_yoy_ma3"] = (
    df_long_median["inc_mem_pct_change"].rolling(3 * 12).mean()
)
df_long_median["income_mem_yoy_ma4"] = (
    df_long_median["inc_mem_pct_change"].rolling(4 * 12).mean()
)
df_long_median["income_mem_yoy_ma5"] = (
    df_long_median["inc_mem_pct_change"].rolling(5 * 12).mean()
)
df_long_median

# %% [md]
# Add 2, 3, 4, 5 year rolling standard deviation

# %%
df_long_median["income_mem_yoy_std2"] = (
    df_long_median["inc_mem_pct_change"].rolling(2 * 12).std()
)
df_long_median["income_mem_yoy_std3"] = (
    df_long_median["inc_mem_pct_change"].rolling(3 * 12).std()
)
df_long_median["income_mem_yoy_std4"] = (
    df_long_median["inc_mem_pct_change"].rolling(4 * 12).std()
)
df_long_median["income_mem_yoy_std5"] = (
    df_long_median["inc_mem_pct_change"].rolling(5 * 12).std()
)
df_long_median

# %% [md]
# Filter for the correct dates

# %%
START_DATE = "2017-01-01"
END_DATE = "2025-10-01"
START_DATE = pd.to_datetime(START_DATE)
END_DATE = pd.to_datetime(END_DATE)

df_long_median = df_long_median.loc[START_DATE:END_DATE]
df_long_median

# %% [md]
# Drop october 2025

# %%
df_long_median = df_long_median.drop(pd.to_datetime("2025-10-01"))
df_long_median

# %% [md]
# Check for null

# %%
df_long_median[df_long_median.isna().any(axis=1)]

# %% [md]
# Check dtype

# %%
df_long_median.dtypes

# %% [md]
# reset index

# %%
df_long_median = df_long_median.reset_index()

# %% [md]
# Final Check

# %%
df_long_median

# %% [md]
# # Export

# %%
df_long_median.to_parquet(DATA_DIR / "feat" / "feat_income_mem.parquet")

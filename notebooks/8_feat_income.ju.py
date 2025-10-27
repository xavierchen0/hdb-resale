# %% [md]
# # Feature Engineering:
# 1. Real Change in Monthly Household Employment Income (Including Employer CPF Contributions) Among Resident Employed Households at Selected Percentiles (Household Employment Income, Annual 2000-2024)

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
HOUSEHOLD_INCOME = DATA_DIR / "household_income.csv"
HDB_IDX = DATA_DIR / "HDB_Resale_Price_Index.csv"
PPI = DATA_DIR / "Private_Property_Index.csv"

df_hi = pd.read_csv(HOUSEHOLD_INCOME)
df_hdb = pd.read_csv(HDB_IDX)
df_ppi = pd.read_csv(PPI)

# %% [md]
# Household Income dataset

# %%
df_long = df_hi.melt(
    id_vars="PerCent",
    value_vars=[str(y) for y in range(2001, 2025)],
    var_name="year",
    value_name="inc_pct_change",
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
new_rows = pd.DataFrame([{"year": pd.to_datetime("2025-12-31"), "inc_pct_change": 1.4}])
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
df_long_median["income_yoy_ma2"] = (
    df_long_median["inc_pct_change"].rolling(2 * 12).mean()
)
df_long_median["income_yoy_ma3"] = (
    df_long_median["inc_pct_change"].rolling(3 * 12).mean()
)
df_long_median["income_yoy_ma4"] = (
    df_long_median["inc_pct_change"].rolling(4 * 12).mean()
)
df_long_median["income_yoy_ma5"] = (
    df_long_median["inc_pct_change"].rolling(5 * 12).mean()
)
df_long_median

# %% [md]
# Add 2, 3, 4, 5 year rolling standard deviation

# %%
df_long_median["income_yoy_std2"] = (
    df_long_median["inc_pct_change"].rolling(2 * 12).std()
)
df_long_median["income_yoy_std3"] = (
    df_long_median["inc_pct_change"].rolling(3 * 12).std()
)
df_long_median["income_yoy_std4"] = (
    df_long_median["inc_pct_change"].rolling(4 * 12).std()
)
df_long_median["income_yoy_std5"] = (
    df_long_median["inc_pct_change"].rolling(5 * 12).std()
)
df_long_median

# %% [md]
# Add income yoy diff (for detection acceleration and deceleration)

# %%
df_long_median["income_yoy_diff"] = df_long_median["inc_pct_change"].diff()
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
# Add income yoy - hdb yoy to find affordability trend

# %%
# Cleaning of hdb dataset
new_rows = pd.DataFrame(
    [
        {"quarter": "2025-Q3", "index": 203.7},
    ]
)
df_hdb = pd.concat([df_hdb, new_rows], ignore_index=True)
df_hdb.loc[:, "quarter"] = pd.PeriodIndex(
    df_hdb["quarter"], freq="Q"
).end_time.normalize()
df_hdb = df_hdb.sort_values(by="quarter")

df_hdb.rename(
    columns={
        "index": "hdb_resale_index",
    },
    inplace=True,
)

df_hdb = df_hdb.set_index("quarter").sort_index()

df_hdb = df_hdb.resample("ME").interpolate("time").sort_index()
df_hdb

# %%
df_hdb["hdb_yoy_change"] = df_hdb["hdb_resale_index"].pct_change(12)

# filter for specific dates
df_hdb = df_hdb[START_DATE:END_DATE]

# change to start of month
df_hdb.index = df_hdb.index - pd.offsets.MonthEnd(1) + pd.offsets.MonthBegin(1)
df_hdb

# %%
df_long_median["inc_yoy_minus_hdb_yoy"] = (
    df_long_median["inc_pct_change"] - df_hdb["hdb_yoy_change"]
)
df_long_median

# %% [md]
# Add income yoy to hdb yoy

# %%
df_long_median["inc_yoy_to_hdb_yoy"] = (
    df_long_median["inc_pct_change"] / df_hdb["hdb_yoy_change"]
)
df_long_median

# %% [md]
# Add income yoy minus ppi yoy

# %%
# Cleaning
new_rows = pd.DataFrame(
    [
        {"quarter": "2025-Q3", "property_type": "All Residential", "index": 215.8},
        {"quarter": "2025-Q3", "property_type": "Landed", "index": 244.8},
        {"quarter": "2025-Q3", "property_type": "Non-Landed", "index": 209.1},
    ]
)
df_ppi = pd.concat([df_ppi, new_rows], ignore_index=True)

df_ppi["quarter"] = pd.PeriodIndex(df_ppi["quarter"], freq="Q").end_time.normalize()
df_ppi = df_ppi.sort_values(by="quarter")

df_ppi_pivot = df_ppi.pivot_table(
    index="quarter",
    columns="property_type",
    values="index",
)

df_ppi_pivot.rename(
    columns={
        "All Residential": "ppi_all_residential",
        "Landed": "ppi_landed",
        "Non-Landed": "ppi_non_landed",
    },
    inplace=True,
)

df_ppi_pivot = df_ppi_pivot.resample("ME").interpolate("time").sort_index()
df_ppi_pivot

# %%

# change to start of month
df_ppi_pivot.index = (
    df_ppi_pivot.index - pd.offsets.MonthEnd(1) + pd.offsets.MonthBegin(1)
)
df_ppi_pivot

# %%
df_ppi_pivot["ppi_all_res_yoy_change"] = df_ppi_pivot["ppi_all_residential"].pct_change(
    12
)
df_ppi_pivot["ppi_landed_yoy_change"] = df_ppi_pivot["ppi_landed"].pct_change(12)
df_ppi_pivot["ppi_non_landed_yoy_change"] = df_ppi_pivot["ppi_non_landed"].pct_change(
    12
)

df_long_median["inc_yoy_ppi_minus_all_res_yoy"] = (
    df_long_median["inc_pct_change"] - df_ppi_pivot["ppi_all_res_yoy_change"]
)


df_long_median["inc_yoy_ppi_minus_landed_yoy"] = (
    df_long_median["inc_pct_change"] - df_ppi_pivot["ppi_landed_yoy_change"]
)


df_long_median["inc_yoy_ppi_minus_non_landed_yoy"] = (
    df_long_median["inc_pct_change"] - df_ppi_pivot["ppi_non_landed_yoy_change"]
)
df_long_median

# %% [md]
# Find income yoy to ppi yoy

# %%
df_long_median["inc_yoy_ppi_to_all_res_yoy"] = (
    df_long_median["inc_pct_change"] / df_ppi_pivot["ppi_all_res_yoy_change"]
)


df_long_median["inc_yoy_ppi_to_landed_yoy"] = (
    df_long_median["inc_pct_change"] / df_ppi_pivot["ppi_landed_yoy_change"]
)


df_long_median["inc_yoy_ppi_to_non_landed_yoy"] = (
    df_long_median["inc_pct_change"] / df_ppi_pivot["ppi_non_landed_yoy_change"]
)
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
df_long_median.to_parquet(DATA_DIR / "feat" / "feat_income.parquet")

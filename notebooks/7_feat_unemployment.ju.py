# %% [md]
# # Feature Engineering: Unemployment (Seasonally-adjusted) dataset

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
# Read unemployment data

# %%
UNEMPLOYMENT_DATA = DATA_DIR / "Unemployment_Rate_SA.csv"

df_ue = pd.read_csv(UNEMPLOYMENT_DATA)
df_ue

# %% [md]
# Convert to a suitable format

# %%
df_ue_long = df_ue.melt(
    id_vars=["DataSeries"], var_name="month_str", value_name="unemp_rate"
)

df_ue_long

# %% [md]
# Resident == SC + PR
# <br>
# More representative to use the unemployment rate for resident
# <br>
# Total unemployment rate is also composed of foreigners
# <br>
# [link](https://www.singstat.gov.sg/find-data/search-by-theme/population/population-and-population-structure/latest-data)
# <br>
# Assume foreigners do not make up a bulk of the HDB resale transactions because to be eligible for HDB resale flats,
# one has to be married to a SC or PR and have to satisfy several other conditions.

# %%
df_ue_all = df_ue_long[
    df_ue_long["DataSeries"] == "Resident Unemployment Rate, (SA)"
].copy()

df_ue_all["unemp_rate"] = pd.to_numeric(df_ue_all["unemp_rate"])
df_ue_all

# %% [md]
# As of 25 October 2025, the 2025Q3 unemployment results has not been released yet
# Therefore, we will just use the last quarter results

# %%
new_rows = pd.DataFrame([{"month_str": "20253Q", "unemp_rate": 2.8}])

df_ue_all = pd.concat([df_ue_all, new_rows], ignore_index=True)
df_ue_all

# %% [md]
# Fix the quarters formatting

# %%
df_ue_all["quarter_clean"] = df_ue_all["month_str"].str.replace(
    r"(\d{4})(\d)Q", r"\1-Q\2", regex=True
)
df_ue_all

# %% [md]
# Set quarter to be at the end of the quarter

# %%
df_ue_all["quarter_clean"] = pd.PeriodIndex(
    df_ue_all["quarter_clean"], freq="Q"
).end_time.normalize()

df_ue_all = df_ue_all.set_index("quarter_clean").sort_index()

df_ue_all

# %% [md]
# Interpolate to get monthly data

# %%
df_ue_all = df_ue_all.resample("ME").interpolate("time").sort_index()
df_ue_all

# %% [md]
# Drop month_str and "DataSeries" column

# %%
df_ue_all = df_ue_all.drop(["month_str", "DataSeries"], axis=1)
df_ue_all

# %% [md]
# Add mom and yoy change

# %%
df_ue_all["unemp_rate_mom_pct_change"] = df_ue_all["unemp_rate"].pct_change()
df_ue_all["unemp_rate_yoy_pct_change"] = df_ue_all["unemp_rate"].pct_change(12)

df_ue_all

# %% [md]
# Compute rolling volatility 3, 6, 9 and 12 months features

# %%
df_ue_all["unemp_rate_mom_change"] = df_ue_all["unemp_rate"].diff()

df_ue_all["unemp_rate_vol_3m"] = df_ue_all["unemp_rate_mom_change"].rolling(3).std()
df_ue_all["unemp_rate_vol_6m"] = df_ue_all["unemp_rate_mom_change"].rolling(6).std()
df_ue_all["unemp_rate_vol_9m"] = df_ue_all["unemp_rate_mom_change"].rolling(9).std()
df_ue_all["unemp_rate_vol_12m"] = df_ue_all["unemp_rate_mom_change"].rolling(12).std()

df_ue_all

# %% [md]
# Filter for the dates we want

# %%
START_DATE = "2017-01-01"
END_DATE = "2025-10-01"
START_DATE = pd.to_datetime(START_DATE)
END_DATE = pd.to_datetime(END_DATE)

df_ue_all = (
    df_ue_all.loc[START_DATE:END_DATE]
    .reset_index()
    .rename(columns={"quarter_clean": "month"})
)
df_ue_all

# %% [md]
# Create unemp shock dummy to capture suddent spike in unemp
# <br>
# A spike is defined to be more 2 standard deviations away historical mean change

# %%
threshold = (
    df_ue_all["unemp_rate_mom_change"].mean()
    + 2 * df_ue_all["unemp_rate_mom_change"].std()
)

df_ue_all["unemp_shock_dummy"] = (
    df_ue_all["unemp_rate_mom_change"] > threshold
).astype(int)

df_ue_all[df_ue_all["unemp_shock_dummy"] == 1]

# %%
df_ue_all

# %% [md]
# Check for nulls

# %%
df_ue_all[df_ue_all.isna().any(axis=1)]

# %% [md]
# Make the month start with the start of the month

# %%
df_ue_all["month"] = (
    df_ue_all["month"] - pd.offsets.MonthEnd(1) + pd.offsets.MonthBegin(1)
)
df_ue_all

# %% [md]
# Check dtypes

# %%
df_ue_all.dtypes

# %% [md]
# Final check

# %%
df_ue_all

# %% [md]
# # Export

# %%
df_ue_all.to_parquet(DATA_DIR / "feat" / "feat_unemployment.parquet")

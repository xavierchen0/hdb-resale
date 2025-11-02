# %% [md]
# # Feature Engineering:
# Total Fertility Rate (TFR) (Per Female) (Per 1,000 Unmarried Resident Males Aged 15-49 Years) based on resident populations (i.e. Singapore citizens and permanent residents).

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
import requests

project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DATA_DIR

# %% [md]
# Read the birthrate ataset

# %%
dataset_id = "d_e39eeaeadb571c0d0725ef1eec48d166"
url = "https://data.gov.sg/api/action/datastore_search?resource_id=" + dataset_id

response = requests.get(url)
data = response.json()

df_br = pd.DataFrame(data["result"]["records"])
df_br.head()

# %% [md]
# Transform data to year as rows

# %%
df_row_0 = df_br.iloc[[0]]
df_long = df_row_0.melt(ignore_index=True, var_name="Year", value_name="birth_rate")
df_long

## %% [md]
# Check data type

# %%
df_long.dtypes

# # %% [md]
# Tranform cols to right datatype

# %%
df_long = df_long[df_long["Year"].str.match(r"^\d{4}$").fillna(False)]
df_long["birth_rate"] = pd.to_numeric(df_long["birth_rate"])
df_long["Year"] = pd.to_datetime(df_long["Year"], format="%Y")
df_long

# %% [md]
# Add missing year 2025

# %%
new_rows = pd.DataFrame([{"Year": pd.to_datetime("2025-12-31"), "birth_rate": 42}])
df_long = pd.concat([df_long, new_rows])
df_long

# %% [md]
# set month as index

# %%
df_long = df_long.set_index("Year").sort_index().rename_axis(index="month")

df_long = df_long.resample("MS").interpolate("time").sort_index()
df_long

# %%
df_long.plot()

# %% [md]
# Find rolling window mean for 2, 3, 4, 5 months

# %%
df_long["birth_rate_ma2"] = df_long["birth_rate"].rolling(2 * 12).mean()
df_long["birth_rate_ma3"] = df_long["birth_rate"].rolling(3 * 12).mean()
df_long["birth_rate_ma4"] = df_long["birth_rate"].rolling(4 * 12).mean()
df_long["birth_rate_ma5"] = df_long["birth_rate"].rolling(5 * 12).mean()
df_long

# %% [md]
# Find rolling window std for 2, 3, 4, 5 months
# %%
df_long["birth_rate_std2"] = df_long["birth_rate"].rolling(2 * 12).std()
df_long["birth_rate_std3"] = df_long["birth_rate"].rolling(3 * 12).std()
df_long["birth_rate_std4"] = df_long["birth_rate"].rolling(4 * 12).std()
df_long["birth_rate_std5"] = df_long["birth_rate"].rolling(5 * 12).std()
df_long

# %% [md]
# Filter for interested dates

# %%
START_DATE = "2017-01-01"
END_DATE = "2025-09-01"
START_DATE = pd.to_datetime(START_DATE)
END_DATE = pd.to_datetime(END_DATE)

df_long = df_long.loc[START_DATE:END_DATE]
df_long

# %%
df_long[df_long.isna().any(axis=1)]

# %%
df_long.dtypes

# %%
df_long = df_long.reset_index()

# %%
pd.set_option("display.max_rows", None)
df_long

# %%
df_long.to_parquet(DATA_DIR / "feat" / "feat_birth_rate.parquet")

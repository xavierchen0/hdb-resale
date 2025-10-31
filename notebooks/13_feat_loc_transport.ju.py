# %% [md]
# # Feature Engineering: Transport

# %% [md]
# Read the base dataset

# %%
import collections
from itertools import zip_longest
from os import stat_result
import pandas as pd
from pathlib import Path
import sys
import geopandas as gpd
from pandas.core.base import ensure_wrapped_if_datetimelike
import seaborn as sns
import matplotlib.pyplot as plt
import re

# Allow importing from src/
project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DATA_DIR

df = gpd.read_parquet(DATA_DIR / "base_dataset.parquet")
df

# %% [md]
# Check the coordinate system

# %%
df.crs

# %% [md]
# Read the lrt mrt exit data, updated 06 June 2024

# %%
df_mrt_lrt = gpd.read_file(DATA_DIR / "lta_mrt_station_exit.geojson").to_crs(epsg=3414)
df_mrt_lrt

# %% [md]
# Check mrt coordinates

# %%
df_mrt_lrt.crs

# %% [md]
# Extract the MRT and LRT name

# %%
df_mrt_lrt["station_name"] = df_mrt_lrt["Description"].str.extract(
    r"<th>\s*STATION_NA\s*</th>\s*<td>(.*?)</td>", flags=re.IGNORECASE
)
df_mrt_lrt

# %% [md]
# Take the first occurrence; The reason for the duplicates is because it also captures the exits for each station

# %%
df_mrt_lrt = df_mrt_lrt.drop_duplicates(subset="station_name", ignore_index=True)
df_mrt_lrt

# %% [md]
# Drop unneeded columns

# %%
df_mrt_lrt = df_mrt_lrt.drop(columns=["Name", "Description"])
df_mrt_lrt

# %% [md]
# Find nearest mrt or lrt

# %%
df_merged = df.sjoin_nearest(
    df_mrt_lrt, how="left", distance_col="distance_to_nearest_mrt_lrt"
)
df_merged

# %% [md]
# drop unneeded cols

# %%
df_merged = df_merged.drop(columns=["index_right"])
df_merged

# %% [md]
# rename "station_name" to "nearest_mrt_lrt"

# %%
df_merged = df_merged.rename(columns={"station_name": "nearest_mrt_lrt"})
df_merged

# %% [md]
# filter for mrt only

# %%
df_mrt = df_mrt_lrt.loc[df_mrt_lrt["station_name"].str.contains("MRT"), :]
df_mrt

# %% [md]
# Merge to get nearest mrt

# %%
df_merged = df_merged.sjoin_nearest(
    df_mrt, how="left", distance_col="distance_to_nearest_mrt"
)
df_merged

# %% [md]
# rename "station_name" to "nearest_mrt"

# %%
df_merged = df_merged.rename(columns={"station_name": "nearest_mrt"})
df_merged

# %% [md]
# drop unneeded cols

# %%
df_merged = df_merged.drop(columns="index_right")
df_merged

# %% [md]
# check if walkable to mrt and lrt

# %%
df_merged["within_400m_mrt_lrt"] = (
    df_merged["distance_to_nearest_mrt_lrt"] <= 400.0
).astype(int)

df_merged["within_400m_mrt"] = (df_merged["distance_to_nearest_mrt"] <= 400.0).astype(
    int
)

df_merged

# %% [md]
# Add bus stop

# %%
path = (DATA_DIR / "BusStopLocation_Aug2025.zip").resolve()

df_bus = gpd.read_file(f"zip://{path}!BusStopLocation_Aug2025/BusStop.shp").to_crs(
    epsg=3414
)
df_bus

# %% [md]
# Check coordinates system

# %%
df_bus.crs

# %% [md]
# Find nearest bus stop

# %%
df_merged = (
    df_merged.sjoin_nearest(
        df_bus, how="left", distance_col="distance_to_nearest_bus_stop"
    )
    .drop(columns="index_right")
    .rename(columns={"BUS_STOP_N": "bus_stop_num", "LOC_DESC": "bus_desc"})
)
df_merged

# %% [md]
# Keep only relevant columns

# %%
df_merged = df_merged.loc[
    :,
    [
        "txn_id",
        "distance_to_nearest_mrt_lrt",
        "distance_to_nearest_mrt",
        "within_400m_mrt_lrt",
        "within_400m_mrt",
        "distance_to_nearest_bus_stop",
    ],
]
df_merged

# %% [md]
# Check for null

# %%
df_merged[df_merged.isna().any(axis=1)]

# %% [md]
# Check dtype

# %%
df_merged.dtypes

# %% [md]
# Final Check

# %%
df_merged

# %% [md]
# # Export

# %%
df_merged.to_parquet(DATA_DIR / "feat" / "feat_loc_transport.parquet")

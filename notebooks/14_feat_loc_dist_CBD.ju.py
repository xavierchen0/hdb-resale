# %% [md]
# # Feature Engineering: Distance to CBD

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
from shapely.geometry import Point
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
# Create GeoPandas for Raffles Place Park also know as CBD

# %%
g_target = gpd.GeoSeries(
    [Point(103.8516393, 1.2843702)],  # lon, lat order
    crs="EPSG:4326",
).to_crs(3414)
g_target

# %%
df["dist_to_cbd"] = df.geometry.distance(g_target.iloc[0])
df

# %% [md]
# Keep only relevant columns

# %%
df = df.loc[
    :,
    [
        "txn_id",
        "dist_to_cbd",
    ],
]
df

# %% [md]
# Check for null

# %%
df[df.isna().any(axis=1)]

# %% [md]
# Check dtype

# %%
df.dtypes

# %% [md]
# Final Check

# %%
df

# %% [md]
# # Export

# %%
df.to_parquet(DATA_DIR / "feat" / "feat_loc_dist_cbd.parquet")

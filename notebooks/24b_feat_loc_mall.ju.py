# %% [md]
# # Feature Engineering: Create mall features

# %% [md]
# Read the base dataset

# %%
import time
import requests
import pandas as pd
import datetime as dt
import json
import os
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import sys
import numpy as np

project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DATA_DIR
from src.utils import accessibility_score_one_point

df = gpd.read_parquet(DATA_DIR / "base_dataset.parquet")
df

# %% [md]
# Read the malls dataset

# %%
df_mall = gpd.read_parquet(DATA_DIR / "loc_mall.parquet")
df_mall = df_mall.to_crs(epsg=3414)
df_mall

# %% [md]
# check other dataset crs

# %%
df_mall.crs

# %% [md]
# Join with the base dataset to get: distance to nearest location

# %%
hdb_nearest_mall = df.sjoin_nearest(
    df_mall[
        ["BUILDING", "geometry"]
    ],  # Select only necessary columns from the right df
    how="left",
    distance_col="distance_to_nearest_mall",  # name the output distance column
)
hdb_nearest_mall

# %% [md]
# Rename Name to nearest_loc

# %%
final_df = hdb_nearest_mall.rename(
    columns={
        "BUILDING": "nearest_mall",
    }
)

final_df

# %% [md]
# Drop index_right col

# %%
final_df = final_df.drop(columns="index_right")
final_df

# %% [md]
# add count of loc within 500m

# %%
hdb_buffer_500 = final_df[["txn_id", "geometry"]].copy()
hdb_buffer_500["geometry"] = hdb_buffer_500.geometry.buffer(500)

join_500 = gpd.sjoin(
    hdb_buffer_500,
    df_mall[["BUILDING", "geometry"]],
    how="left",
    predicate="contains",
)

# Use .groupby().count() on a column from the RIGHT table
# .count() only counts non-NaN values, giving a true count of 0 for unmatched rows.
count_500 = (
    join_500.groupby("txn_id")["BUILDING"]
    .count()
    .rename("num_malls_500m")
    .reset_index()
)
count_500

# %% [md]
# add count of loc within 1km

# %%
hdb_buffer_1000 = final_df[["txn_id", "geometry"]].copy()
hdb_buffer_1000["geometry"] = hdb_buffer_1000.geometry.buffer(1000)

join_1000 = gpd.sjoin(
    hdb_buffer_1000,
    df_mall[["BUILDING", "geometry"]],
    how="left",
    predicate="contains",
)

# Use .groupby().count() on a column from the RIGHT table
# .count() only counts non-NaN values, giving a true count of 0 for unmatched rows.
count_1000 = (
    join_1000.groupby("txn_id")["BUILDING"]
    .count()
    .rename("num_malls_1000m")
    .reset_index()
)
count_1000

# %%
final_df = final_df.merge(count_500, on="txn_id", how="left")
final_df = final_df.merge(count_1000, on="txn_id", how="left")

final_df

# %% [md]
# calculate accessability score
# <br>
# We define the *hawker accessibility index* for each HDB flat $i$  as:
# <br>
# $$
# A_i = \sum_{j=1}^{N} \exp\left(-\frac{d_{ij}}{\lambda}\right)
# $$
# where:
# - $A_i$ — accessibility score for flat $ i $
# - $ N $ — total number of hawker centres
# - $ d_{ij} $ — distance (in meters) between flat $ i $ and hawker centre $ j $
# - $ \lambda $ — decay parameter (e.g., $ \lambda = 500 $), controlling how quickly accessibility declines with distance
# <br>
# <br>
# Intuitively, this measure gives higher scores to flats that are:
# - **closer** to hawker centres (smaller $ d_{ij} $), and
# - **surrounded by multiple** hawker centres within walking distance.
# <br>
# <br>
# A larger $ \lambda $ means distance matters less (slower decay),
# while a smaller $ \lambda $ means only nearby hawkers contribute significantly.
# %%
mall_points = df_mall.geometry.to_list()

# --- Find the correct list of coordinates ---
mall_coords_list = []
for p in mall_points:
    # Safely check if the point is valid before extracting coordinates
    if p is not None and p.is_valid:
        mall_coords_list.append((p.x, p.y))

# This guarantees the required 2D structure (N rows, 2 columns).
mall_array = np.array(mall_coords_list, dtype=np.float64)

final_df["mall_access"] = final_df.geometry.apply(
    lambda geom: accessibility_score_one_point(geom, mall_array, lam=500)
)
final_df

# %% [md]
# Keep relevant cols

# %%
df_merged = final_df.loc[
    :,
    [
        "txn_id",
        "distance_to_nearest_mall",
        "num_malls_500m",
        "num_malls_1000m",
        "mall_access",
    ],
]
df_merged

# %%
df_merged[df_merged.isna().any(axis=1)]

# %%
df_merged.dtypes

# %%
df_merged

# %%
df_merged.to_parquet(DATA_DIR / "feat" / "feat_loc_mall.parquet")

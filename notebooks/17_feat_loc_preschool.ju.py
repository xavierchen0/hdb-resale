# %% [md]
# # Feature Engineering: Distance to pre-school

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
import numpy as np

project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DATA_DIR
import src.utils as utils

df = gpd.read_parquet(DATA_DIR / "base_dataset.parquet")
df

# %% [md]
# Import the themes dataset

# %%
df_other_loc = gpd.read_parquet(DATA_DIR / "sg_finalfac_loc.parquet")
df_other_loc = df_other_loc.to_crs(epsg=3414)
df_other_loc

# %% [md]
# Filter the facilities DataFrame to only include preschool

# %%
df_preschool = df_other_loc[df_other_loc["Theme"] == "PRESCHOOL"].copy()
df_preschool

# %% [md]
# check themes dataset crs

# %%
df_preschool.crs

# %% [md]
# Join with the base dataset to get: distance to nearest hawker centre

# %%
hdb_nearest_preschool = df.sjoin_nearest(
    df_preschool[
        ["Name", "geometry"]
    ],  # Select only necessary columns from the right df
    how="left",
    distance_col="distance_to_nearest_preschool",  # Name the output distance column
)
hdb_nearest_preschool


# %% [md]
# Rename Name to nearest_preschool

# %%
final_df = hdb_nearest_preschool.rename(
    columns={
        "Name": "nearest_preschool",
    }
)

final_df

# %% [md]
# Drop index_right col

# %%
final_df = final_df.drop(columns="index_right")
final_df

# %% [md]
# add count of preschool within 500m

# %%
hdb_buffer_500 = final_df[["txn_id", "geometry"]].copy()
hdb_buffer_500["geometry"] = hdb_buffer_500.geometry.buffer(500)

join_500 = gpd.sjoin(
    hdb_buffer_500,
    df_preschool[["Name", "geometry"]],
    how="left",
    predicate="contains",
)

# count preschool per txn_id
count_500 = (
    join_500.groupby("txn_id").size().rename("num_preschools_500m").reset_index()
)
count_500

# %% [md]
# add count of preschool within 1km

# %%
hdb_buffer_1000 = final_df[["txn_id", "geometry"]].copy()
hdb_buffer_1000["geometry"] = hdb_buffer_1000.geometry.buffer(1000)

join_1000 = gpd.sjoin(
    hdb_buffer_1000,
    df_preschool[["Name", "geometry"]],
    how="left",
    predicate="contains",
)

count_1000 = (
    join_1000.groupby("txn_id").size().rename("num_preschools_1000m").reset_index()
)
count_1000

# %% [md]
# merge both counts back into gdf_hdb

# %%
final_df = final_df.merge(count_500, on="txn_id", how="left")
final_df = final_df.merge(count_1000, on="txn_id", how="left")

final_df

# %% [md]
# fill flats with no preschools nearby as zero instead of NaN

# %%
final_df[["num_preschools_500m", "num_preschools_1000m"]] = (
    final_df[["num_preschools_500m", "num_preschools_1000m"]].fillna(0).astype(int)
)
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
preschool_points = df_preschool.geometry.to_list()
preschool_array = np.array([(p.x, p.y) for p in preschool_points])

final_df["preschool_access"] = final_df.geometry.apply(
    lambda geom: utils.accessibility_score_one_point(geom, preschool_array, lam=500)
)
final_df

# %% [md]
# Keep relevant cols

# %%
df_merged = final_df.loc[
    :,
    [
        "txn_id",
        "distance_to_nearest_preschool",
        "num_preschools_500m",
        "num_preschools_1000m",
        "preschool_access",
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
df_merged.to_parquet(DATA_DIR / "feat" / "feat_loc_preschool.parquet")

# %% [md]
# # Feature Engineering: Distance to sec school

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
from src.utils import accessibility_score_one_point

df = gpd.read_parquet(DATA_DIR / "base_dataset.parquet")
df

# %% [md]
# Import the secondary school dataset

# %%
df_other_loc = gpd.read_parquet(DATA_DIR / "loc_sec_sch.parquet")
df_other_loc = df_other_loc.to_crs(epsg=3414)
df_other_loc

# %% [md]
# check other dataset crs

# %%
df_other_loc.crs

# %% [md]
# Join with the base dataset to get: distance to nearest location

# %%
hdb_nearest = df.sjoin_nearest(
    df_other_loc[
        ["school_name", "geometry"]
    ],  # Select only necessary columns from the right df
    how="left",
    distance_col="distance_to_nearest_secschool",  # Name the output distance column
)
hdb_nearest

# %% [md]
# Rename Name to nearest_loc

# %%
final_df = hdb_nearest.rename(
    columns={
        "school_name": "nearest_sec_sch",
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
    df_other_loc[["school_name", "geometry"]],
    how="left",
    predicate="contains",
)

# Use .groupby().count() on a column from the RIGHT table (school_name)
# .count() only counts non-NaN values, giving a true count of 0 for unmatched rows.
count_500 = (
    join_500.groupby("txn_id")["school_name"]
    .count()
    .rename("num_secschools_500m")
    .reset_index()
)
# count_500 will now show 0 for transactions with no schools inside the buffer.
count_500

# %% [md]
# add count of loc within 1km

# %%
hdb_buffer_1000 = final_df[["txn_id", "geometry"]].copy()
hdb_buffer_1000["geometry"] = hdb_buffer_1000.geometry.buffer(1000)

join_1000 = gpd.sjoin(
    hdb_buffer_1000,
    df_other_loc[["school_name", "geometry"]],
    how="left",
    predicate="contains",
)

# Use .groupby().count() on a column from the RIGHT table (school_name)
# .count() only counts non-NaN values, giving a true count of 0 for unmatched rows.
count_1000 = (
    join_1000.groupby("txn_id")["school_name"]
    .count()
    .rename("num_secschools_1000m")
    .reset_index()
)
count_1000


# %% [md]
# merge both counts back into final_df

# %%
final_df = final_df.merge(count_500, on="txn_id", how="left")
final_df = final_df.merge(count_1000, on="txn_id", how="left")

final_df

# %% [md]
# fill flats with no loc nearby as zero instead of NaN

# %%
final_df[["num_secschools_500m", "num_secschools_1000m"]] = (
    final_df[["num_secschools_500m", "num_secschools_1000m"]].fillna(0).astype(int)
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
secschool_points = df_other_loc.geometry.to_list()
secschool_array = np.array([(p.x, p.y) for p in secschool_points])

final_df["secschool_access"] = final_df.geometry.apply(
    lambda geom: accessibility_score_one_point(geom, secschool_array, lam=500)
)
final_df

# %% [md]
# Keep relevant cols

# %%
df_merged = final_df.loc[
    :,
    [
        "txn_id",
        "distance_to_nearest_secschool",
        "num_secschools_500m",
        "num_secschools_1000m",
        "secschool_access",
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
df_merged.to_parquet(DATA_DIR / "feat" / "feat_loc_secschool.parquet")

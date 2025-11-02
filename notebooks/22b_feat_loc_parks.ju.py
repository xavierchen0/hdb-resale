# %% [md]
# # Feature Engineering: Distance to parks

# %% [md]
# Read the base dataset

# %%
import collections
from itertools import zip_longest
from os import stat_result
import pandas as pd
from pathlib import Path
import sys
import numpy as np
import requests
import geopandas as gpd

project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DATA_DIR
from src.utils import accessibility_score_one_point

df = gpd.read_parquet(DATA_DIR / "base_dataset.parquet")
df

# %% [md]
# Read the nparks dataset

# %%
df_other_loc = gpd.read_parquet(DATA_DIR / "loc_nparks.parquet")
gdf_nparks = df_other_loc.to_crs(epsg=3414)
gdf_nparks

# %% [md]
# check other dataset crs

# %%
df_other_loc.crs

# %% [md]
# Join with the base dataset to get: distance to nearest location

# %%
hdb_nearest = df.sjoin_nearest(
    gdf_nparks[["NAME", "geometry"]],  # Select only necessary columns from the right df
    how="left",
    distance_col="distance_to_nearest_park",  # Name the output distance column
)
hdb_nearest

# %% [md]
# Rename Name to nearest_loc

# %%
final_df = hdb_nearest.rename(
    columns={
        "NAME": "nearest_park",
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
    gdf_nparks[["NAME", "geometry"]],
    how="left",
    predicate="contains",
)

# Use .groupby().count() on a column from the RIGHT table
# .count() only counts non-NaN values, giving a true count of 0 for unmatched rows.
count_500 = (
    join_500.groupby("txn_id")["NAME"].count().rename("num_parks_500m").reset_index()
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
    gdf_nparks[["NAME", "geometry"]],
    how="left",
    predicate="contains",
)

# Use .groupby().count() on a column from the RIGHT table
# .count() only counts non-NaN values, giving a true count of 0 for unmatched rows.
count_1000 = (
    join_1000.groupby("txn_id")["NAME"].count().rename("num_parks_1000m").reset_index()
)
count_1000

# %% [md]
# merge both counts back into final_df

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
park_points = gdf_nparks.geometry.to_list()

# --- Find the correct list of coordinates ---
park_coords_list = []
for p in park_points:
    # Safely check if the point is valid before extracting coordinates
    if p is not None and p.is_valid:
        park_coords_list.append((p.x, p.y))

# This guarantees the required 2D structure (N rows, 2 columns).
park_array = np.array(park_coords_list, dtype=np.float64)

final_df["parks_access"] = final_df.geometry.apply(
    lambda geom: accessibility_score_one_point(geom, park_array, lam=500)
)
final_df

# %% [md]
# Keep relevant cols

# %%
df_merged = final_df.loc[
    :,
    [
        "txn_id",
        "distance_to_nearest_park",
        "num_parks_500m",
        "num_parks_1000m",
        "parks_access",
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
df_merged.to_parquet(DATA_DIR / "feat" / "feat_loc_parks.parquet")

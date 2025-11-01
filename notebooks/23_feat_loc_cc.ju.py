# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

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

# %%
project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DATA_DIR
from src.score_utils import accessibility_score_one_point

df = gpd.read_parquet(DATA_DIR / "base_dataset.parquet")
df

# %%
df_cc = gpd.read_parquet(DATA_DIR / "cc_dataset.parquet")
df_cc = df_cc.to_crs(epsg=3414)
df_cc

# %%
hdb_nearest_hawker = df.sjoin_nearest(
    df_cc[['Name', 'geometry']], # Select only necessary columns from the right df
    how='left',                       
    distance_col='distance_to_nearest_cc' # Name the output distance column
)

final_df = hdb_nearest_hawker.rename(
    columns={
        'Name': 'nearest_cc',
    }
)

final_df

# %%
final_df = final_df.drop(columns="index_right")
final_df

# %%
hdb_buffer_500 = final_df[["txn_id", "geometry"]].copy()
hdb_buffer_500["geometry"] = hdb_buffer_500.geometry.buffer(500)

join_500 = gpd.sjoin(
    hdb_buffer_500,
    df_cc[["NAME", "geometry"]],
    how="left",
    predicate="contains",
)

# Use .groupby().count() on a column from the RIGHT table
# .count() only counts non-NaN values, giving a true count of 0 for unmatched rows.
count_500 = (
    join_500.groupby("txn_id")["NAME"]
    .count()
    .rename("num_cc_500m")
    .reset_index()
)
# count_500 will now show 0 for transactions with no schools inside the buffer.
count_500

# %%
hdb_buffer_1000 = final_df[["txn_id", "geometry"]].copy()
hdb_buffer_1000["geometry"] = hdb_buffer_1000.geometry.buffer(1000)

join_1000 = gpd.sjoin(
    hdb_buffer_1000,
    df_cc[["NAME", "geometry"]],
    how="left",
    predicate="contains",
)

# Use .groupby().count() on a column from the RIGHT table
# .count() only counts non-NaN values, giving a true count of 0 for unmatched rows.
count_1000 = (
    join_1000.groupby("txn_id")["NAME"]
    .count()
    .rename("num_cc_1000m")
    .reset_index()
)
count_1000

# %%
final_df = final_df.merge(count_500, on="txn_id", how="left")
final_df = final_df.merge(count_1000, on="txn_id", how="left")

final_df

# %%
cc_points = df_cc.geometry.to_list()

# --- Find the correct list of coordinates ---
cc_coords_list = []
for p in cc_points:
    # Safely check if the point is valid before extracting coordinates
    if p is not None and p.is_valid:
        cc_coords_list.append((p.x, p.y))

# This guarantees the required 2D structure (N rows, 2 columns).
cc_array = np.array(cc_coords_list, dtype=np.float64)

final_df["cc_access"] = final_df.geometry.apply(
    lambda geom: accessibility_score_one_point(geom, cc_array, lam=500)
)
final_df

# %%
df_merged = final_df.loc[
    :,
    [
        "txn_id",
        "distance_to_nearest_cc",
        "num_cc_500m",
        "num_cc_1000m",
        "cc_access",
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
df_merged.to_parquet(DATA_DIR / "feat" / "feat_loc_cc.parquet")

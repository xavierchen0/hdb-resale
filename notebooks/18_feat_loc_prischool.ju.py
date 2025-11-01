# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

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

# %%
project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DATA_DIR
from src.score_utils import accessibility_score_one_point

df = gpd.read_parquet(DATA_DIR / "base_dataset.parquet")
df

# %%
df_other_loc = gpd.read_parquet(DATA_DIR / "loc_pri_sch.parquet")
df_other_loc = df_other_loc.to_crs(epsg=3414)
df_other_loc

# %%
df_other_loc.crs

# %%
hdb_nearest = df.sjoin_nearest(
    df_other_loc[
        ["school_name", "geometry"]
    ],  # Select only necessary columns from the right df
    how="left",
    distance_col="distance_to_nearest_prischool",  # Name the output distance column
)
hdb_nearest

# %%
final_df = hdb_nearest.rename(
    columns={
        "school_name": "nearest_pri_sch",
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
    df_other_loc[["school_name", "geometry"]],
    how="left",
    predicate="contains",
)

# count prischool per txn_id
count_500 = (
    join_500.groupby("txn_id").size().rename("num_prischools_500m").reset_index()
)
count_500

# %%
hdb_buffer_1000 = final_df[["txn_id", "geometry"]].copy()
hdb_buffer_1000["geometry"] = hdb_buffer_1000.geometry.buffer(1000)

join_1000 = gpd.sjoin(
    hdb_buffer_1000,
    df_other_loc[["school_name", "geometry"]],
    how="left",
    predicate="contains",
)

count_1000 = (
    join_1000.groupby("txn_id").size().rename("num_prischools_1000m").reset_index()
)
count_1000

# %%
final_df = final_df.merge(count_500, on="txn_id", how="left")
final_df = final_df.merge(count_1000, on="txn_id", how="left")

final_df

# %%
final_df[["num_prischools_500m", "num_prischools_1000m"]] = (
    final_df[["num_prischools_500m", "num_prischools_1000m"]].fillna(0).astype(int)
)
final_df

# %%
prischool_points = df_other_loc.geometry.to_list()
prischool_array = np.array([(p.x, p.y) for p in prischool_points])

final_df["prischool_access"] = final_df.geometry.apply(
    lambda geom: accessibility_score_one_point(geom, prischool_array, lam=500)
)
final_df

# %%
df_merged = final_df.loc[
    :,
    [
        "txn_id",
        "distance_to_nearest_prischool",
        "num_prischools_500m",
        "num_prischools_1000m",
        "prischool_access",
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
df_merged.to_parquet(DATA_DIR / "feat" / "feat_loc_prischool.parquet")

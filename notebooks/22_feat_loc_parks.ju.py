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
dataset_id = "d_0542d48f0991541706b58059381a6eca"
API_BASE_URL = "https://api-open.data.gov.sg/v1/public/api/datasets/"
POLL_DOWNLOAD_URL = API_BASE_URL + dataset_id + "/poll-download"

print("--- Step 1: Retrieving Download URL ---")

poll_response = requests.get(POLL_DOWNLOAD_URL)
poll_json = poll_response.json()

if poll_json.get('code') != 0:
    print(f"Error retrieving download URL: {poll_json.get('errMsg')}")
    raise Exception("API poll failed.")

download_url = poll_json['data']['url']
print("Download URL retrieved successfully.")

# 2. Download the data
data_response = requests.get(download_url)

data_json = data_response.json()
print(f"Data downloaded successfully. Total features: {len(data_json.get('features', []))}")

# %%
print("--- Step 2: Extracting Properties to DataFrame ---")

# Extract the 'properties' dictionary from each 'feature' in the JSON
properties_list = [feature['properties'] for feature in data_json.get('features', [])]

# Create the initial DataFrame
df_nparks = pd.DataFrame(properties_list)

# Check for the existence of X and Y columns
if 'X' not in df_nparks.columns or 'Y' not in df_nparks.columns:
    print(f"Error: Required coordinate columns (X, Y) not found.")
    print("Available columns:", df_nparks.columns.tolist())
    raise Exception("Missing coordinate data in JSON properties.")

# Display the initial DataFrame structure
print(f"DataFrame created with {len(df_nparks)} records.")
df_nparks.head()

# %%
print("--- Step 3: Convert X/Y to Geometry (GeoDataFrame) ---")

# 1. Ensure X and Y coordinates are numeric
df_nparks['X'] = pd.to_numeric(df_nparks['X'])
df_nparks['Y'] = pd.to_numeric(df_nparks['Y'])

# Drop any rows where X or Y could not be read or were invalid
df_nparks.dropna(subset=['X', 'Y'], inplace=True)
print(f"Cleaned DataFrame size: {len(df_nparks)} records.")

# 2. Create the geometry column: map the (X, Y) pairs to Shapely Point objects
# X/Y are (Easting/Northing)
geometry = [
    Point(xy) for xy in zip(
        df_nparks['X'], 
        df_nparks['Y']
    )
]

# 3. Create the GeoDataFrame
# We explicitly set the CRS (Coordinate Reference System) to SVY21 (EPSG:3414)
gdf_nparks = gpd.GeoDataFrame(
    df_nparks.copy(), 
    geometry=geometry,
    crs="EPSG:3414" 
)

# Display the GeoDataFrame, showing the new geometry column
print("\nGeoDataFrame (gdf_nparks) Head:")
gdf_nparks.head()

# %%
project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DATA_DIR
from src.score_utils import accessibility_score_one_point

df = gpd.read_parquet(DATA_DIR / "base_dataset.parquet")
df

# %%
hdb_nearest = df.sjoin_nearest(
    gdf_nparks[
        ["NAME", "geometry"]
    ],  # Select only necessary columns from the right df
    how="left",
    distance_col="distance_to_nearest_park",  # Name the output distance column
)
hdb_nearest

# %%
final_df = hdb_nearest.rename(
    columns={
        "NAME": "nearest_park",
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
    gdf_nparks[["NAME", "geometry"]],
    how="left",
    predicate="contains",
)

# Use .groupby().count() on a column from the RIGHT table
# .count() only counts non-NaN values, giving a true count of 0 for unmatched rows.
count_500 = (
    join_500.groupby("txn_id")["NAME"]
    .count()
    .rename("num_parks_500m")
    .reset_index()
)
# count_500 will now show 0 for transactions with no schools inside the buffer.
count_500

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
    join_1000.groupby("txn_id")["NAME"]
    .count()
    .rename("num_parks_1000m")
    .reset_index()
)
count_1000

# %%
final_df = final_df.merge(count_500, on="txn_id", how="left")
final_df = final_df.merge(count_1000, on="txn_id", how="left")

final_df

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

# %% [md]
# # Feature Engineering: Generate the parks dataset

# %% [md]
# Download parks dataset

# %%
import collections
from itertools import zip_longest
from os import stat_result
import pandas as pd
from pathlib import Path
import sys
import numpy as np
import requests
from shapely import Point
import geopandas as gpd

project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DATA_DIR
import src.utils as utils

dataset_id = "d_0542d48f0991541706b58059381a6eca"
API_BASE_URL = "https://api-open.data.gov.sg/v1/public/api/datasets/"
POLL_DOWNLOAD_URL = API_BASE_URL + dataset_id + "/poll-download"

print("--- Step 1: Retrieving Download URL ---")

poll_response = requests.get(POLL_DOWNLOAD_URL)
poll_json = poll_response.json()

if poll_json.get("code") != 0:
    print(f"Error retrieving download URL: {poll_json.get('errMsg')}")
    raise Exception("API poll failed.")

download_url = poll_json["data"]["url"]
print("Download URL retrieved successfully.")

# 2. Download the data
data_response = requests.get(download_url)

data_json = data_response.json()
print(
    f"Data downloaded successfully. Total features: {len(data_json.get('features', []))}"
)

# %% [md]
# Extract to pandas dataframe

# %%
print("--- Step 2: Extracting Properties to DataFrame ---")

# Extract the 'properties' dictionary from each 'feature' in the JSON
properties_list = [feature["properties"] for feature in data_json.get("features", [])]

# Create the initial DataFrame
df_nparks = pd.DataFrame(properties_list)

# Check for the existence of X and Y columns
if "X" not in df_nparks.columns or "Y" not in df_nparks.columns:
    print(f"Error: Required coordinate columns (X, Y) not found.")
    print("Available columns:", df_nparks.columns.tolist())
    raise Exception("Missing coordinate data in JSON properties.")

# Display the initial DataFrame structure
print(f"DataFrame created with {len(df_nparks)} records.")
df_nparks.head()

# %% [md]
# Convert to geopandas dataframe

# %%
print("--- Step 3: Convert X/Y to Geometry (GeoDataFrame) ---")

# 1. Ensure X and Y coordinates are numeric
df_nparks["X"] = pd.to_numeric(df_nparks["X"])
df_nparks["Y"] = pd.to_numeric(df_nparks["Y"])

# Drop any rows where X or Y could not be read or were invalid
df_nparks.dropna(subset=["X", "Y"], inplace=True)
print(f"Cleaned DataFrame size: {len(df_nparks)} records.")

# 2. Create the geometry column: map the (X, Y) pairs to Shapely Point objects
# X/Y are (Easting/Northing)
geometry = [Point(xy) for xy in zip(df_nparks["X"], df_nparks["Y"])]

# 3. Create the GeoDataFrame
# We explicitly set the CRS (Coordinate Reference System) to SVY21 (EPSG:3414)
gdf_nparks = gpd.GeoDataFrame(df_nparks.copy(), geometry=geometry, crs="EPSG:3414")

# Display the GeoDataFrame, showing the new geometry column
print("\nGeoDataFrame (gdf_nparks) Head:")
gdf_nparks.head()

# %% [md]
# Drop the object id column

# %%
gdf_nparks = gdf_nparks.drop(columns="OBJECTID")
gdf_nparks

# %% [md]
# # Export

# %%
gdf_nparks.to_parquet(DATA_DIR / "loc_nparks.parquet")

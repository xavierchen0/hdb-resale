# %% [md]
# # Feature Engineering: Generate mall data

# %% [md]
# Download the mall locations data

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
from src.open_map_utils import openmap_search
from src.utils import fetch_overpass

QUERY = r"""
[out:json][timeout:60];
area["ISO3166-1"="SG"]->.sg;

(
  // Core: explicitly tagged malls
  node["shop"="mall"](area.sg);
  way["shop"="mall"](area.sg);
  relation["shop"="mall"](area.sg);

  // Broadened: retail/commercial buildings that look like malls by name
  // Use case-insensitive flag ,i (Overpass) instead of (?i)
  nwr["building"~"retail|commercial"](area.sg)
     ["name"~"mall|shopping centre|shopping center|plaza|square|arcade|centrepoint",i];
);
out tags center;
"""


data = fetch_overpass(QUERY)
elements = data.get("elements", [])

rows = []
for el in elements:
    tags = el.get("tags") or {}
    name = tags.get("name")
    if not name:
        continue
    lat = el.get("lat") or (el.get("center") or {}).get("lat")
    lon = el.get("lon") or (el.get("center") or {}).get("lon")
    source = "core" if tags.get("shop") == "mall" else "broadened"
    rows.append({"name": name, "lat": lat, "lon": lon, "source": source})

df = pd.DataFrame(rows)
if not df.empty:
    df["core_rank"] = (df["source"] == "core").astype(int)
    df = (
        df.sort_values(["name", "core_rank"], ascending=[True, False])
        .drop_duplicates("name")
        .drop(columns="core_rank")
        .reset_index(drop=True)
    )

mall_names = df["name"].tolist() if not df.empty else []
mall_names

# %% [md]
# Create mall dataframe

# %%
frames = []

for mall_name in mall_names:
    try:
        r = openmap_search(mall_name)  # your existing function
        r.raise_for_status()  # optional
        j = r.json()
        results = j.get("results") or []
        if not results:
            continue
        first = results[0]  # <-- index 0
        first["query_input"] = mall_name
        frames.append(first)
    except requests.RequestException as e:
        pass

mall_data = pd.DataFrame(frames)

# %% [md]
# Drop duplicated values

# %%
mall_data = mall_data.drop_duplicates("SEARCHVAL")
mall_data

# %% [md]
# Fix null values

# %%
# Replace any 'N/A' strings with a proper NaN value
mall_data["X"] = mall_data["X"].replace("N/A", np.nan)
mall_data["Y"] = mall_data["Y"].replace("N/A", np.nan)

# Convert columns to numeric, forcing any other errors to NaN
mall_data["X"] = pd.to_numeric(mall_data["X"])
mall_data["Y"] = pd.to_numeric(mall_data["Y"])


# %% [md]
# create mall geopandas

# %%
initial_rows = len(mall_data)
# Create a new DataFrame without the rows that have null coordinates
df_clean = mall_data.dropna(subset=["X", "Y"]).copy()
final_rows = len(df_clean)
dropped_rows = initial_rows - final_rows

print(f"Initial rows: {initial_rows}")
print(f"Rows with valid X/Y coordinates: {final_rows}")
print(f"Rows dropped due to missing X/Y: {dropped_rows}")

print("\n--- Creating GeoDataFrame ---")
df_mall = gpd.GeoDataFrame(
    df_clean, geometry=gpd.points_from_xy(df_clean["X"], df_clean["Y"])
)

# OneMap X/Y coordinates are in SVY21 (EPSG:3414)
df_mall.set_crs(epsg=3414, inplace=True)
df_mall

# %%
df_mall[df_mall.isna().any(axis=1)]

# %%
df_mall.dtypes

# %%
df_mall

# %% [md]
# # Export

# %%
df_mall.to_parquet(DATA_DIR / "loc_mall.parquet")

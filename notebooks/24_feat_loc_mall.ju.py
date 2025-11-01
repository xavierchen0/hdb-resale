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
headers = os.environ.get("headers")

# %%
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]

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

def openmap_search(searchVal, returnGeom="Y", getAddrDetails="Y", pageNum=1):
    """
    https://www.onemap.gov.sg/apidocs/search

    Returns query results
    """
    url = "https://www.onemap.gov.sg/api/common/elastic/search"

    params = {
        "searchVal": searchVal,
        "returnGeom": returnGeom,
        "getAddrDetails": getAddrDetails,
        "pageNum": pageNum,
    }

    response = requests.get(url, params=params, headers=headers)

    return response

def fetch_overpass(query: str) -> dict:
    last_err = None
    for url in OVERPASS_ENDPOINTS:
        try:
            r = requests.post(url, data={"data": query}, timeout=120)
            if r.status_code != 200:
                print(f"[warn] {url} → {r.status_code}\n{r.text[:500]}")
                r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_err = e
            continue
    raise last_err

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

# %%
frames = []

for mall_name in mall_names:
    try:
        r = openmap_search(mall_name)   # your existing function
        r.raise_for_status()            # optional
        j = r.json()
        results = j.get("results") or []
        if not results:
            continue
        first = results[0]              # <-- index 0
        first["query_input"] = mall_name
        frames.append(first)
    except requests.RequestException as e:
        pass

mall_data = pd.DataFrame(frames)

# %%
mall_data = mall_data.drop_duplicates("SEARCHVAL")
mall_data

# %%
# Replace any 'N/A' strings with a proper NaN value
mall_data['X'] = mall_data['X'].replace('N/A', np.nan)
mall_data['Y'] = mall_data['Y'].replace('N/A', np.nan)

# Convert columns to numeric, forcing any other errors to NaN
mall_data['X'] = pd.to_numeric(mall_data['X'])
mall_data['Y'] = pd.to_numeric(mall_data['Y'])


initial_rows = len(mall_data)
# Create a new DataFrame without the rows that have null coordinates
df_clean = mall_data.dropna(subset=['X', 'Y']).copy()
final_rows = len(df_clean)
dropped_rows = initial_rows - final_rows

print(f"Initial rows: {initial_rows}")
print(f"Rows with valid X/Y coordinates: {final_rows}")
print(f"Rows dropped due to missing X/Y: {dropped_rows}")

# 3. Create the GeoDataFrame

print("\n--- Creating GeoDataFrame ---")
df_mall = gpd.GeoDataFrame(
    df_clean, 
    geometry=gpd.points_from_xy(df_clean['X'], df_clean['Y'])
)

# OneMap X/Y coordinates are in SVY21 (EPSG:3414)
df_mall.set_crs(epsg=3414, inplace=True)
df_mall

# %%
project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DATA_DIR
from src.score_utils import accessibility_score_one_point

df = gpd.read_parquet(DATA_DIR / "base_dataset.parquet")
df

# %%
hdb_nearest_mall = df.sjoin_nearest(
    df_mall[['BUILDING', 'geometry']], # Select only necessary columns from the right df
    how='left',                       
    distance_col='distance_to_nearest_mall' # name the output distance column
)

final_df = hdb_nearest_mall.rename(
    columns={
        'BUILDING': 'nearest_mall',
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

# %% [md]
# # Add regions, and transaction id and produce base dataset

# %% [md]
# Load partially cleaned dataset

# %%
import pandas as pd
from pathlib import Path
import sys
import geopandas as gpd

# Allow importing from src/
project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DATA_DIR

df = pd.read_parquet(DATA_DIR / "partial_dataset_1.parquet")
df

# %% [md]
# Load regions geojson data

# %%
regions_gdf = gpd.read_file(DATA_DIR / "sg_regions.geojson")
regions_gdf

# %% [md]
# Regions data is using CRS84 which is equivalent to EPSG:4326 which is equivalent to WGS84
# <br>
# Need to convert to Singapore's SVY21 which is equivalent to EPSG:3414

# %%
regions_gdf = regions_gdf.to_crs(crs=3414)
regions_gdf.crs

# %% [md]
# Convert dataframe to geopandas

# %%
gdf = gpd.GeoDataFrame(data=df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs=3414)
gdf

# %% [md]
# Join to get region

# %%
gdf = gdf.sjoin(regions_gdf, how="left")
gdf

# %% [md]
# Remove unncessary column

# %%
gdf = gdf.drop("index_right", axis=1)
gdf

# %% [md]
# Add unique transaction id using the current index

# %%
gdf = gdf.reset_index()
gdf

# %% [md]
# Rename the column

# %%
gdf = gdf.rename(columns={"index": "txn_id"})
gdf

# %% [md]
# Set datatype of region

# %%
gdf["region"] = gdf["region"].astype("category")
gdf.dtypes

# %% [md]
# # Export base dataset

# %%
gdf.to_parquet(DATA_DIR / "base_dataset.parquet")

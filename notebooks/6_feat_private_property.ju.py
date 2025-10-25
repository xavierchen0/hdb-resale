# %% [md]
# # Feature Engineering: Private Property Index

# %% [md]
# Read the base dataset

# %%
import pandas as pd
from pathlib import Path
import sys
import geopandas as gpd
from pandas.core.base import ensure_wrapped_if_datetimelike
import seaborn as sns
import matplotlib.pyplot as plt

# Allow importing from src/
project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DATA_DIR

df = gpd.read_parquet(DATA_DIR / "base_dataset.parquet")
df

# %% [md]
# Read private property dataset

# %%
PPI_DATA = DATA_DIR / "Private_Property_Index.csv"

df_ppi = pd.read_csv(PPI_DATA)
df_ppi

# %% [md]
# Read HDB Resale Price Index

# %%
HDB_RESALE_INDEX = DATA_DIR / "HDB_Resale_Price_Index.csv"

df_hdb = pd.read_csv(HDB_RESALE_INDEX)
df_hdb

# %% [md]
# # PPI
# As of 25 October 2025, the 2025Q3 PPI results were released, but not updated in
# data.gov.sg. Therefore, we will add the 2025Q3 results manually

# %%
new_rows = pd.DataFrame(
    [
        {"quarter": "2025-Q3", "property_type": "All Residential", "index": 215.8},
        {"quarter": "2025-Q3", "property_type": "Landed", "index": 244.8},
        {"quarter": "2025-Q3", "property_type": "Non-Landed", "index": 209.1},
    ]
)
df_ppi = pd.concat([df_ppi, new_rows], ignore_index=True)
df_ppi

# %% [md]
# Convert the quarter column to datetime

# %%
df_ppi["quarter"] = pd.PeriodIndex(df_ppi["quarter"], freq="Q").end_time.normalize()
df_ppi = df_ppi.sort_values(by="quarter")
df_ppi

# %% [md]
# Create a pivot table of months by property type

# %%
df_ppi_pivot = df_ppi.pivot_table(
    index="quarter",
    columns="property_type",
    values="index",
)

df_ppi_pivot.rename(
    columns={
        "All Residential": "ppi_all_residential",
        "Landed": "ppi_landed",
        "Non-Landed": "ppi_non_landed",
    },
    inplace=True,
)

df_ppi_pivot

# %% [md]
# Plot of PPI

# %%
df_ppi_pivot.plot()

# %% [md]
# Since our data is quarterly, we need to convert it to monthly data
# <br>
# The decision is to interpolate them between two quarters because PPI is a constantly adjusting and
# moving value that is only captured quarterly.
# <br>
# Set the quarter as the month end of the quarter
# <br>
# Our monthly PPI values correspond to the end of the month

# %%
df_ppi_pivot = df_ppi_pivot.resample("ME").interpolate("time").sort_index()
df_ppi_pivot

# %% [md]
# Add mom and yoy pct change

# %%
# All Residential
df_ppi_pivot.loc[:, "ppi_all_residential_mom_pct_change"] = df_ppi_pivot[
    "ppi_all_residential"
].pct_change()

df_ppi_pivot.loc[:, "ppi_all_residential_yoy_pct_change"] = df_ppi_pivot[
    "ppi_all_residential"
].pct_change(12)


# Landed
df_ppi_pivot.loc[:, "ppi_landed_mom_pct_change"] = df_ppi_pivot[
    "ppi_landed"
].pct_change()

df_ppi_pivot.loc[:, "ppi_landed_yoy_pct_change"] = df_ppi_pivot[
    "ppi_landed"
].pct_change(12)

# Non-Landed
df_ppi_pivot.loc[:, "ppi_non_landed_mom_pct_change"] = df_ppi_pivot[
    "ppi_non_landed"
].pct_change()

df_ppi_pivot.loc[:, "ppi_non_landed_yoy_pct_change"] = df_ppi_pivot[
    "ppi_non_landed"
].pct_change(12)

df_ppi_pivot

# %% [md]
# # HDB
# As of 25 October 2025, the 2025Q3 HDB results were released, but not updated in
# data.gov.sg. Therefore, we will add the 2025Q3 results manually

# %%
new_rows = pd.DataFrame(
    [
        {"quarter": "2025-Q3", "index": 203.7},
    ]
)
df_hdb = pd.concat([df_hdb, new_rows], ignore_index=True)
df_hdb

# %% [md]
# Convert the quarter column to datetime

# %%
df_hdb.loc[:, "quarter"] = pd.PeriodIndex(
    df_hdb["quarter"], freq="Q"
).end_time.normalize()
df_hdb = df_hdb.sort_values(by="quarter")
df_hdb

# %% [md]
# Rename index to hdb_resale_index

# %%
df_hdb.rename(
    columns={
        "index": "hdb_resale_index",
    },
    inplace=True,
)

df_hdb

# %% [md]
# Set index as the quarter

# %%
df_hdb = df_hdb.set_index("quarter").sort_index()
df_hdb

# %% [md]
# Plot of PPI

# %%
df_hdb.plot()

# %% [md]
# Since our data is quarterly, we need to convert it to monthly data
# <br>
# The decision is to interpolate them between two quarters because HDB Resale Price Index is a constantly adjusting and
# moving value that is only captured quarterly.
# <br>
# Set the quarter as the month end of the quarter
# <br>
# Our monthly HDB Resale Price Index values correspond to the end of the month

# %%
df_hdb = df_hdb.resample("ME").interpolate("time").sort_index()
df_hdb

# %% [md]
# # For both HDB and PPI
# Filter for the months we are looking
# Both HDB and PPI share the same base of 1Q2009 = 100

# %%
START_DATE = "2017-01-01"
END_DATE = "2025-10-01"
START_DATE = pd.to_datetime(START_DATE)
END_DATE = pd.to_datetime(END_DATE)

df_ppi_pivot = df_ppi_pivot.loc[START_DATE:END_DATE]

df_ppi_pivot

# %%
df_hdb = df_hdb.loc[START_DATE:END_DATE]

df_hdb

# %% [md]
# Add price gap ratio feature
# <br>
# This says: how many “units” of HDB price for one “unit” of private price?
# <br>
# High ratio → private much more expensive than HDB → stronger spillover demand into HDB.
# <br>
# Private property prices are a substitute and an anchor.
# If condos get expensive, some buyers get priced out and turn to
# <br>
# resale HDB → resale demand ↑ → HDB prices ↑.
# <br>

# %%
df_ppi_pivot.loc[:, "price_gap_ratio_all_residential"] = (
    df_ppi_pivot["ppi_all_residential"] / df_hdb["hdb_resale_index"]
)

df_ppi_pivot.loc[:, "price_gap_ratio_landed"] = (
    df_ppi_pivot["ppi_landed"] / df_hdb["hdb_resale_index"]
)

df_ppi_pivot.loc[:, "price_gap_ratio_non_landed"] = (
    df_ppi_pivot["ppi_non_landed"] / df_hdb["hdb_resale_index"]
)

df_ppi_pivot

# %% [md]
# Add price gap diff feature
# <br>
# Same concept as above

# %%
df_ppi_pivot.loc[:, "price_gap_diff_all_residential"] = (
    df_ppi_pivot["ppi_all_residential"] - df_hdb["hdb_resale_index"]
)

df_ppi_pivot.loc[:, "price_gap_diff_landed"] = (
    df_ppi_pivot["ppi_landed"] - df_hdb["hdb_resale_index"]
)

df_ppi_pivot.loc[:, "price_gap_diff_non_landed"] = (
    df_ppi_pivot["ppi_non_landed"] - df_hdb["hdb_resale_index"]
)

df_ppi_pivot

# %% [md]
# Verify na

# %%
df_ppi_pivot[df_ppi_pivot.isna().any(axis=1)]

# %% [md]
# Make the month start with the start of the month

# %%
df_ppi_pivot.index = (
    df_ppi_pivot.index - pd.offsets.MonthEnd(1) + pd.offsets.MonthBegin(1)
)
df_ppi_pivot

# %% [md]
# drop index and change to month

# %%
df_ppi_pivot = df_ppi_pivot.reset_index().rename(columns={"quarter": "month"})
df_ppi_pivot

# %% [md]
# Verify datatype

# %%
df_ppi_pivot.dtypes

# %% [md]
# # Export

# %%
df_ppi_pivot.to_parquet(DATA_DIR / "feat_private_property.parquet")

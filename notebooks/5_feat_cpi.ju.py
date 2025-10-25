# %% [md]
# # Feature Engineering: CPI data

# %% [md]
# Read the base dataset

# %%
import pandas as pd
from pathlib import Path
import sys
import geopandas as gpd
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
# Read the CPI data

# %%
CPI_DATA = DATA_DIR / "CPI.csv"

df_cpi = pd.read_csv(CPI_DATA)
df_cpi

# %% [md]
# Convert to a nicer format

# %%
df_cpi_long = df_cpi.melt(
    id_vars=["DataSeries"], var_name="month_str", value_name="cpi_index"
)

df_cpi_long

# %% [md]
# Keep only "All Items" as we only want to know the general inflation level

# %%
df_cpi_all = df_cpi_long[df_cpi_long["DataSeries"] == "All Items"].copy(deep=True)
df_cpi_all

# %% [md]
# Convert month to datetime
# convert the format from "YYYYMon" (e.g., "2025Aug") to "YYYY-MM-01".

# %%
df_cpi_all["month"] = pd.to_datetime(df_cpi_all["month_str"], format="%Y%b")
df_cpi_all

# %% [md]
# Ensure cpi_index is numeric

# %%
df_cpi_all["cpi_index"] = pd.to_numeric(df_cpi_all["cpi_index"])
df_cpi_all.dtypes

# %% [md]
# Sort by month

# %%
df_cpi_all.sort_values(by="month", ascending=True, inplace=True)
df_cpi_all


# %% [md]
# Filter for the correct months
# <br>
# Set one year before so that we can capture pct change

# %%
START_DATE = "2016-01-01"
END_DATE = "2025-09-01"
START_DATE = pd.to_datetime(START_DATE)
END_DATE = pd.to_datetime(END_DATE)

df_cpi_final = df_cpi_all[
    (df_cpi_all["month"] >= START_DATE) & (df_cpi_all["month"] <= END_DATE)
][["month", "cpi_index"]].reset_index(drop=True)

df_cpi_final

# %% [md]
# Add missing dates: September 2025 and using last month value

# %%
last_cpi_value = df_cpi_final["cpi_index"].iloc[-1]

new_row = pd.DataFrame(
    {
        "month": [END_DATE],
        "cpi_index": [last_cpi_value],
    }
)

df_cpi_final = pd.concat([df_cpi_final, new_row], ignore_index=True).sort_values(
    by="month"
)
df_cpi_final

# %% [md]
# Add mom and yoy pct change<br>
# Removes base-year bias <br>
# Captures real inflation momentum <br>

# %%
df_cpi_final["cpi_mom_pct"] = df_cpi_final["cpi_index"].pct_change()
df_cpi_final["cpi_yoy_pct"] = df_cpi_final["cpi_index"].pct_change(12)
df_cpi_final

# %% [md]
# Filter for the months we actually need

# %%
FINAL_START = "2017-01-01"
END_DATE = "2025-09-01"
FINAL_START = pd.to_datetime(FINAL_START)
END_DATE = pd.to_datetime(END_DATE)

df_cpi_final = df_cpi_final[
    (df_cpi_final["month"] >= FINAL_START) & (df_cpi_final["month"] <= END_DATE)
].reset_index(drop=True)

df_cpi_final

# %% [md]
# # Export data

# %%
df_cpi_final.to_parquet(DATA_DIR / "feat_cpi.parquet")

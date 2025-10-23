# %% [md]
# # Feature Engineering: COE data

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
# Read COE data

# %%
COE_DATA = DATA_DIR / "COE.csv"

df_coe = pd.read_csv(
    COE_DATA,
    dtype={
        "bidding_no": "Int32",
        "vehicle_class": "category",
        "premium": "Int32",
    },
    parse_dates=["month"],
).sort_values(by="month")
df_coe

# %% [md]
# Filter date range related to hdb resale dataset

# %%
START_DATE = "2017-01-01"
END_DATE = "2025-09-01"
START_DATE = pd.to_datetime(START_DATE)
END_DATE = pd.to_datetime(END_DATE)

df_coe = df_coe[(df_coe["month"] >= START_DATE) & (df_coe["month"] <= END_DATE)]
df_coe

# %% [md]
# Get the mean premium for each month because there are two bidding rounds

# %%
df_coe_pivot = df_coe.pivot_table(
    index="month", columns="vehicle_class", values="premium", aggfunc="mean"
).reset_index()
df_coe_pivot

# %% [md]
# Rename columns

# %%
df_coe_pivot.rename(
    columns={
        "Category A": "coe_a",
        "Category B": "coe_b",
        "Category C": "coe_c",
        "Category D": "coe_d",
        "Category E": "coe_e",
    },
    inplace=True,
)
df_coe_pivot

# %% [md]
# From the chart below, the COE prices for each category is about the same with some slight increase

# %%
df_long = df_coe_pivot.melt(
    id_vars="month",
    value_vars=["coe_a", "coe_b", "coe_c", "coe_d", "coe_e"],
    var_name="category",
    value_name="coe_price",
)

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_long, x="month", y="coe_price", hue="category", marker="o")

# %% [md]
# Add equally spaced values between months
# <br>
# ‘time’: Works on daily and higher resolution data to interpolate given length of interval.

# %%
# Add missing dates
new_rows = pd.DataFrame(
    [
        {"month": pd.to_datetime("2020-04-01")},
        {"month": pd.to_datetime("2020-05-01")},
        {"month": pd.to_datetime("2020-06-01")},
    ]
)

df_coe_pivot = pd.concat([df_coe_pivot, new_rows], ignore_index=True)

# Sort by month
df_coe_pivot = df_coe_pivot.sort_values(by="month")

# Interpolate
df_coe_pivot = df_coe_pivot.set_index("month")
for category in ["coe_a", "coe_b", "coe_c", "coe_d", "coe_e"]:
    df_coe_pivot[category] = df_coe_pivot[category].interpolate(method="time")

# Verify
ls = [
    pd.to_datetime("2020-03-01"),
    pd.to_datetime("2020-04-01"),
    pd.to_datetime("2020-05-01"),
    pd.to_datetime("2020-06-01"),
    pd.to_datetime("2020-07-01"),
]
df_coe_pivot[df_coe_pivot.index.isin(ls)]

# %% [md]
# Merge coe data to main df to get features

# %%
df_merged = pd.merge(df, df_coe_pivot.reset_index(), on="month", how="left")
df_merged

# %% [md]
# Check for null values

# %%
df_merged.loc[
    df_merged.isna().any(axis=1), ["month", "coe_a", "coe_b", "coe_c", "coe_d", "coe_e"]
]

# %% [md]
# # Export

# %%
df_export = df_merged[
    [
        "month",
        "coe_a",
        "coe_b",
        "coe_c",
        "coe_d",
        "coe_e",
    ]
]
df_export

# %%
df_export.to_parquet(DATA_DIR / "feat_coe.parquet")

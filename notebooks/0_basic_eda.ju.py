# %% [md]
# # Purpose
# This Jupyter notebook serves to generate the HTML data report from ydata-profiling package, while performing some basic EDA and cleaning.

# %% [md]
# # Generate Report

# %%
import sys
from pathlib import Path
import pandas as pd
# !!! Don't create report with current environment; there is an issue
# Use the already generated html file
# from ydata_profiling import ProfileReport

# Allow importing from src/
project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DATA_DIR

# Ensure you are at the root directory
HDB_TXN_DATA_FP = DATA_DIR / "data.csv"
HDB_TXN_REPORT_FP = "data/hdb_txn_raw_report.html"

raw_df = pd.read_csv(HDB_TXN_DATA_FP)
raw_df
# profile = ProfileReport(raw_df, title="EDA on Raw Dataset")
# profile.to_file(HDB_TXN_REPORT_FP)

# %% [md]
# # Insights
# *Tip*: below is best understood with the report opened side by side

# %% [md]
# ## # 1: No missing data
# - In the overview section

# %% [md]
# ## # 2: Quite a bit of duplicated rows so we need to remove them
# - In the overview/"Duplicate rows" sections
# - There is a possibility that these duplicated transactions are actually distinct HDB units which were are in the same `storey_range` and are actually sold at the same price in `resale_price`.
# - Or they are actually a mistake.
# - Considering that they represent a small subset of 0.1% of the whole dataset, it will not significantly influence the modelling result

# %%
# For e.g 1
raw_df[
    (raw_df["town"] == "SEMBAWANG")
    & (raw_df["flat_type"] == "4 ROOM")
    & (raw_df["block"] == "103B")
    & (raw_df["month"] == "2025-04")
]

# %%
# For e.g 2
raw_df[
    (raw_df["town"] == "BUKIT MERAH")
    & (raw_df["flat_type"] == "4 ROOM")
    & (raw_df["block"] == "106")
    & (raw_df["month"] == "2017-01")
]

# %%
# For e.g 3
raw_df[
    (raw_df["town"] == "CENTRAL AREA")
    & (raw_df["flat_type"] == "3 ROOM")
    & (raw_df["block"] == "271")
    & (raw_df["month"] == "2017-01")
]

# %% [md]
# ## # 3: It is better to enforce the correct pandas datatype
# - Better than simply using `read_csv()`
# | column                 | Correct Datatype  | comments  |
# | :-                     | :-                | :-        |
# | `month`                | DatetimeTZDtype   | -         |
# | `town`                 | Categorical       | -         |
# | `flat_type`            | Categorical       | -         |
# | `block`                | Categorical       | Though there are many different distinct blocks, cardinality represent `1.3%` of the whole dataset, so it is still okay to set it as a `Categorical` datatype.          |
# | `street_name`          | Categorical       | Though there are many different distinct street names, cardinality represent `0.3%` of the whole dataset, so it is still okay to set it as a `Categorical` datatype.          |
# | `storey_range`         | Categorical       |-         |
# | `floor_area_sqm`       | Float32Dtype      |-         |
# | `flat_model`           | Categorical       |-         |
# | `lease_commence_date`  | DatetimeTZDtype   | `read_csv()` recognise this col as `int64`, but it should be recognised as date datatype         |
# | `remaining_lease`      | DatetimeTZDtype   | Transformation is necessary to become a date datatype as currently it is a string        |

# %%
raw_df.dtypes

# %% [md]
# # Transformations

# %% [md]
# ## Reading the right datatype
# - Reading csv with right data types
# - Transformation for col `remaining_lease`
# - There are some rows with no months, so we assume `0` months
#   - Check:
#   ```python
#   df1[df1["month1"].isnull()]
#   ```
# - All rows have years
#   - Check:
#   ```python
#   df1[df1["year1"].isnull()]
#   ```
# - Verify transformation
#   - Check:
#   ```python
#   df1[(df1["month2"].isnull()) & (df1["month1"] != "0")]
#   ```
# - Treat col `remaining_lease` as a col of months

# %%
# Read the csv into dataframe
df1 = pd.read_csv(
    HDB_TXN_DATA_FP,
    dtype={
        "town": "category",
        "flat_type": "category",
        "block": "category",
        "street_name": "category",
        "storey_range": "category",
        "floor_area_sqm": "float32",
        "flat_model": "category",
        "remaining_lease": "string",
        "resale_price": "float32",
    },
    parse_dates=["month", "lease_commence_date"],
)
df1.dtypes

# %% [md]
# ## Remove the month October 2025 (2025-10)

# %%
print(f"Initial No. of Observations: {len(df1):,}")

EXCLUDE_MONTH_STRING = "2025-10"

CNT_EXCLUDE_MONTH_ROWS = len(df1[df1["month"] == EXCLUDE_MONTH_STRING])

print(f"Number of rows with {EXCLUDE_MONTH_STRING}: {CNT_EXCLUDE_MONTH_ROWS}")

df1 = df1[df1["month"] != EXCLUDE_MONTH_STRING].copy()
df1.reset_index(drop=True, inplace=True)
print(f"No. of Observations after excluding {EXCLUDE_MONTH_STRING}: {len(df1):,}")

df1.dtypes

# %% [md]
# ## Change `remaining_lease` to months

# %%
# Split the col `remaining_lease`
df1[["year1", "year2", "month1", "month2"]] = df1["remaining_lease"].str.split(
    " ", expand=True
)

# Convert months with `null` with `0`
df1["month1"] = df1["month1"].fillna("0")

# Drop unnecessary cols
df1 = df1.drop(["year2", "month2", "remaining_lease"], axis=1)

# Convert cols "year1" and "month1" to integers
df1["year1"] = df1["year1"].astype("int32")
df1["month1"] = df1["month1"].astype("int32")

# Get col "remaining_lease" as col of months
df1["remaining_lease"] = df1["year1"] * 12 + df1["month1"]

# Drop unncessary cols
df1 = df1.drop(["year1", "month1"], axis=1)

df1

# %%
df1.dtypes

# %% [md]
# ## Deduplication
# - Keep the first occurrence
# - Check removed rows:
# ```python
# removed = df1[df1.duplicated(keep=False)] # Show all duplicated rows
# ```

# %%
df1.sort_values(by="month", ascending=True, inplace=True, ignore_index=True)
df1 = df1.drop_duplicates(ignore_index=True)  # Keep first occurrence

# %% [md]
# ## Add region and towns

# %%
towns_regions = pd.read_csv("data/sg_towns_regions.csv")

# Map each town to a region
df1 = df1.merge(towns_regions, how="left", on="town")

# Check for towns with null regions
# - ['CENTRAL AREA', 'KALLANG/WHAMPOA']
# df1.loc[df1['region'].isnull(),"town"].unique().tolist()

# Update null regions
df1.loc[df1["town"] == "CENTRAL AREA", "region"] = "CENTRAL REGION"
df1.loc[df1["town"] == "KALLANG/WHAMPOA", "region"] = "CENTRAL REGION"

# Verify there is no more rows with null regions
# df1.loc[df1["region"].isnull(), "town"].unique().tolist()

# Convert to categorical column
df1["region"] = df1["region"].astype("category")
df1.dtypes

# %% [md]
# # Export as pickle file

# %%
df1

# %%
df1.to_pickle("data/cleaned_rawdata.pkl")

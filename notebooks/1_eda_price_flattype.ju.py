# %% [md]
# # Purpose
# Perform EDA on the dataset

# %% [md]
# # Basic Data Exploration

# %% [md]
# ## Data table

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import json
import sys
from pathlib import Path

sys.path.append(Path().resolve() / "src")

from src.config import DATA_DIR

df = pd.read_pickle(DATA_DIR / "cleaned_data.pkl")
df

# %% [md]
# ## Data types

# %%
df.dtypes

# %% [md]
# ## Flat Types

# %%
df["flat_type"].unique()

# %% [md]
# ## Flat Models

# %%
print(list(df["flat_model"].unique()))

# %% [md]
# # Resale Price Trend
# 1. There is a clear and distinct resale price segmentation betweeent the flat types; This makes sense because the flat type varies with the size of the flat and there is a common belief that the larger the size of the house, the higher the resale value of the house.
# 2. Executive flats are discontinued in 1995.

# %%
df_monthly_flattype = (
    df.groupby(["month", "flat_type"])["resale_price"].mean().reset_index()
)
df_monthly_med = df.groupby("month")["resale_price"].median().reset_index()
df_monthly_avg = df.groupby("month")["resale_price"].mean().reset_index()

plt.figure(figsize=(12, 6))

sns.lineplot(
    data=df_monthly_flattype,
    x="month",
    y="resale_price",
    hue="flat_type",
)
sns.lineplot(
    data=df_monthly_avg,
    x="month",
    y="resale_price",
    linestyle="--",
    label="Average",
    linewidth=3,
)
sns.lineplot(
    data=df_monthly_med,
    x="month",
    y="resale_price",
    linestyle="-.",
    label="Median",
    linewidth=3,
)

plt.title("Monthly Average HDB Resale Price by Flat Type")
plt.xlabel("Month")
plt.ylabel("Resale Price (SGD)")
plt.tight_layout()

# %% [md]
# # Pearson/Spearman Correlation of different flat types timeseries
# **Note**:
# - [source](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html#pandas.DataFrame.corr)
# - columns with NaN values are not included in the correlation analysis

# %% [md]
# ### 1. Key Finding
# - The Pearson correlation matrix shows extremely high linear co-movement (ρ ≈ 0.9–1.0) across almost all flat types.
# - The Spearman correlation matrix similarly confirms strong monotonic relationships, except for the multi-generation flats, which exhibit noticeably weaker correlations (ρ ≈ 0.55–0.7).
# - This indicates that the entire resale market moves together, with multi-generation flats as the main outlier segment.
# ### 2. Interpretation
# - **Mainstream Flats (1-, 2-, 3-, 4-, 5-room, Executive)**
#   - All display near-perfect correlation, suggesting they are influenced by the same macroeconomic and policy factors, such as:
#     - Interest-rate movements and mortgage affordability
#     - Cooling measures (LTV limits, ABSD, MSR)
#     - BTO supply delays and upgrader demand cycles
#   - The similar correlation strengths imply a unified resale market where price changes propagate evenly across flat sizes.
# - **Multi-Generation Flats (3Gen)**
#   - Show the lowest co-movement with other types (ρ ≈ 0.55–0.7 Spearman; ≈ 0.75–0.8 Pearson).
# - Possible reasons:
#   - Extremely small market share (~3.9 % of all transactions) → low monthly sample size causes high volatility.
#   - Distinct buyer profile — targeted at extended families; resale demand depends more on specific household needs than general affordability.
#   - Recent introduction (post-2013, [link](https://www.factually.gov.sg/corrections-and-clarifications/factually220224)) — shorter trading history and inconsistent monthly activity reduce statistical correlation.
#   - Heterogeneous pricing factors — larger floor areas, unique layouts, and limited geographical distribution drive idiosyncratic prices.
# ### 3. Analytical Implications
# - For time-series modeling or forecasting, the 3- to 5-room and Executive flats can be treated as a single coherent market segment.
# - Multi-generation flats should either be:
#   - Modeled separately as a niche sub-market, or
#   - Excluded from correlation-based analyses to avoid distortion.
# - The distinction highlights market segmentation by household type, not by flat size alone.

# %%
pivot = df_monthly_flattype.pivot(
    index="month", columns="flat_type", values="resale_price"
)
pivot = pivot.sort_index()
pivot.head()

# %% [md]
# ### High pairwise correlation
# %%
corr = pivot.corr(method="pearson")
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=0.5, vmax=1)

# %%
corr = pivot.corr(
    method="spearman"
)  # Only cares about the rank (whether up together, down together)
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=0.5, vmax=1)

# %% [md]
# ### Percentage of "MULTI-GENERATION" transactions in dataset

# %%
len(df[df["flat_type"] == "MULTI-GENERATION"]) / len(df) * 100

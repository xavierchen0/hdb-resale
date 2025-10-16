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
import numpy as np

project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DATA_DIR

df = pd.read_pickle(DATA_DIR / "cleaned_rawdata.pkl")
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

# %% [md]
# # Year on Year Growth change

# %% [md]
# ## All

# %%
df_monthly_flattype_pct = df_monthly_flattype.sort_values(
    ["flat_type", "month"]
).assign(
    yoy_growth=df_monthly_flattype.groupby("flat_type")["resale_price"].pct_change(
        periods=12
    )
    * 100
)

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_monthly_flattype_pct, x="month", y="yoy_growth", hue="flat_type")
plt.axhline(y=0, color="gray", linestyle="--", linewidth=3)

# %% [md]
# ## Narrow focus

# %%
plt.figure(figsize=(12, 6))
filter = (
    # (df_monthly_flattype_pct["flat_type"] == "1 ROOM")
    (df_monthly_flattype_pct["flat_type"] == "2 ROOM")
    | (df_monthly_flattype_pct["flat_type"] == "3 ROOM")
)
sns.lineplot(
    data=df_monthly_flattype_pct[filter], x="month", y="yoy_growth", hue="flat_type"
)
plt.axhline(y=0, color="gray", linestyle="--", linewidth=3)

# %% [md]
# # CAGR per flat type
# **Market Insight: Compound Annual Growth Rate (CAGR) by Flat Type (2017–2024)**

# **Overview**
# The chart compares the compound annual growth rate (CAGR) of HDB resale prices across different flat types between 2017 and 2024.
# It shows how various flat segments have appreciated at different rates over the long term, reflecting structural differences in demand, affordability, and policy effects.

# **Key Findings**
# - 3-room flats recorded the highest CAGR at approximately **7.5%**, outperforming all other categories.
# - Mid-sized flats (4-room and 5-room) appreciated steadily at around **5.3%**, indicating strong and consistent upgrader demand.
# - Multi-generation flats achieved a solid **5.9% CAGR**, showing niche but healthy growth despite lower transaction volumes.
# - 2-room flats saw moderate appreciation (**5.3%**), while 1-room flats lagged slightly (**3.6%**), reflecting limited market size and liquidity.
# - Executive flats underperformed significantly with only **1.25% CAGR**, highlighting weak resale demand and affordability constraints in this segment.

# **Interpretation**
# 1. The dominance of 3-room flats suggests increasing demand for smaller, more affordable homes—likely driven by policy grants, affordability pressures, and changing household structures.
# 2. 4-room and 5-room flats continue to anchor the resale market, showing steady, broad-based appreciation aligned with upgrader family demand.
# 3. Executive flats’ weak growth points to financing limitations and reduced buyer interest in larger, high-priced units amid rising interest rates.
# 4. The healthy performance of multi-generation flats indicates rising appeal of multigenerational living and scarcity premiums for large units post-pandemic.

# **Market Implications**
# - The market’s growth leadership shifted toward smaller and mid-sized flats, signaling a structural change in housing demand patterns.
# - Affordability and policy support (e.g., Enhanced CPF Housing Grant, BTO delays) have likely amplified demand for 3-room and 4-room flats.
# - Future price resilience may be strongest in these core segments, while larger flats may continue to underperform due to limited affordability and liquidity.

# **Conclusion**
# Between 2017 and 2024, the HDB resale market experienced broad-based price growth, but appreciation was uneven across flat types.
# 3-room flats emerged as the top-performing segment, reflecting a long-term demand shift toward affordability and efficiency.
# The consistent growth in 4-room and 5-room flats underscores their continued role as the market’s backbone, while the lag in executive flats highlights potential saturation in the high-end public housing segment.

# %%
df_flattype_cagr = (
    df.groupby("flat_type")
    .agg(
        start_price=("resale_price", "first"),
        end_price=("resale_price", "last"),
        n_years=("month", lambda x: (x.max() - x.min()).days / 365),
    )
    .assign(
        cagr=lambda d: ((d["end_price"] / d["start_price"]) ** (1 / d["n_years"]) - 1)
        * 100
    )
)

plt.figure(figsize=(12, 6))

ax = sns.barplot(
    data=df_flattype_cagr,
    x="flat_type",
    y="cagr",
)

avg_cagr = df_flattype_cagr["cagr"].mean()
ax.axhline(
    y=avg_cagr,
    color="gray",
    linestyle="--",
    linewidth=3,
    label="0",
)

ax.text(x=-0.5, y=avg_cagr + 0.1, s=f"Avg: {avg_cagr:.2f}%")

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f%%", label_type="edge", padding=3)

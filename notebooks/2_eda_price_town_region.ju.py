# %% [md]
# # Explore price against town and region

# %%
import pandas as pd
import sys
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from src.config import DATA_DIR

df = pd.read_pickle(DATA_DIR / "cleaned_rawdata.pkl")
df.head()

# %% [md]
# # Average Price vs Region
# - Clear indication of which region commands the higher resale price

# %%
df_monthly_region = df.groupby(["month", "region"])["resale_price"].mean().reset_index()

plt.figure(figsize=(12, 6))

sns.lineplot(data=df_monthly_region, x="month", y="resale_price", hue="region")

plt.title("Resale Price over time of each region")

# %% [md]
# # Correlation analysis
# - Strong correlation across regions

# %%
pivot = df_monthly_region.pivot(index="month", columns="region", values="resale_price")
pivot.sort_index()
pivot.head()

# %%
corr = pivot.corr(method="pearson")
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=0.5, vmax=1)

# %%
corr = pivot.corr(method="spearman")
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=0.5, vmax=1)

# %% [md]
# # Average Price vs Town

# %% [md]
# ## West Region

# %%
df_monthly_town = (
    df[df["region"] == "WEST REGION"]
    .groupby(["month", "town"])["resale_price"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(12, 6))

sns.lineplot(data=df_monthly_town, x="month", y="resale_price", hue="town")

# %% [md]
# ### Correlation analysis
# - Strong correlation across regions

# %%
pivot = df_monthly_town.pivot(index="month", columns="town", values="resale_price")
pivot.sort_index()
pivot.head()

# %%
corr = pivot.corr(method="pearson")
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=0.5, vmax=1)

# %%
corr = pivot.corr(method="spearman")
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=0.5, vmax=1)

# %% [md]
# ## North Region

# %%
df_monthly_town = (
    df[df["region"] == "NORTH REGION"]
    .groupby(["month", "town"])["resale_price"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(12, 6))

sns.lineplot(data=df_monthly_town, x="month", y="resale_price", hue="town")

# %% [md]
# ### Correlation analysis
# - Strong correlation across regions

# %%
pivot = df_monthly_town.pivot(index="month", columns="town", values="resale_price")
pivot.sort_index()
pivot.head()

# %%
corr = pivot.corr(method="pearson")
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=0.5, vmax=1)

# %%
corr = pivot.corr(method="spearman")
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=0.5, vmax=1)

# %% [md]
# ## East Region

# %%
df_monthly_town = (
    df[df["region"] == "EAST REGION"]
    .groupby(["month", "town"])["resale_price"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(12, 6))

sns.lineplot(data=df_monthly_town, x="month", y="resale_price", hue="town")

# %% [md]
# ### Correlation analysis
# - Strong correlation across regions

# %%
pivot = df_monthly_town.pivot(index="month", columns="town", values="resale_price")
pivot.sort_index()
pivot.head()

# %%
corr = pivot.corr(method="pearson")
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=0.5, vmax=1)

# %%
corr = pivot.corr(method="spearman")
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=0.5, vmax=1)

# %% [md]
# ## North east Region

# %%
df_monthly_town = (
    df[df["region"] == "NORTH-EAST REGION"]
    .groupby(["month", "town"])["resale_price"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(12, 6))

sns.lineplot(data=df_monthly_town, x="month", y="resale_price", hue="town")

# %% [md]
# ### Correlation analysis
# - Strong correlation across regions

# %%
pivot = df_monthly_town.pivot(index="month", columns="town", values="resale_price")
pivot.sort_index()
pivot.head()

# %%
corr = pivot.corr(method="pearson")
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=0.5, vmax=1)

# %%
corr = pivot.corr(method="spearman")
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=0.5, vmax=1)


# %% [md]
# ## Central Region

# %%
df_monthly_town = (
    df[df["region"] == "CENTRAL REGION"]
    .groupby(["month", "town"])["resale_price"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(12, 6))

sns.lineplot(data=df_monthly_town, x="month", y="resale_price", hue="town")

# %% [md]
# ### Correlation analysis
# ### Why the Central Region correlation map looks like this
# - Thin samples in some towns (e.g., `CENTRAL AREA`, `MARINE PARADE`) lead to noisy monthly means; noise weakens pairwise correlations.
# - Highly heterogeneous stock: older towns (e.g., `QUEENSTOWN`, `TOA PAYOH`) with pronounced lease-decay effects vs. renewal/younger blocks in `BUKIT MERAH` / `KALLANG/WHAMPOA`.
# - Micro-location premia (CBD/MRT proximity, modernised precincts, sea views) drive idiosyncratic month-to-month shifts that do not synchronize across towns.
# - Policy segmentation (e.g., PLH from 2021) narrows buyer pools for prime projects, creating dynamics less aligned with non-PLH towns.
# - Asynchronous supply (BTO/SPR completions, upgrading cycles) varies by town and time, introducing timing mismatches in price moves.

# %%
pivot = df_monthly_town.pivot(index="month", columns="town", values="resale_price")
pivot.sort_index()
pivot.head()

# %%
corr = pivot.corr(method="pearson")
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=0.5, vmax=1)

# %%
corr = pivot.corr(method="spearman")
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=0.5, vmax=1)

# %% [md]
# ### Quick diagnostics: sample size and volatility
# The table below shows each Central Region town's transaction depth and volatility metrics.
# Lower median monthly counts and higher short-horizon volatility typically correspond to weaker cross-town correlations.

# %%
central = df[df["region"] == "CENTRAL REGION"].copy()

# Monthly counts per town
counts = central.groupby(["town", "month"]).size().rename("n").reset_index()

counts_summary = counts.groupby("town").agg(
    months_with_txn=("n", "size"),
    total_txn=("n", "sum"),
    median_monthly_n=("n", "median"),
)

# Volatility of MoM % change in avg resale price (higher = noisier)
avg_price = (
    central.groupby(["town", "month"])
    .agg(avg_price=("resale_price", "mean"))
    .reset_index()
)
avg_price = avg_price.sort_values(["town", "month"]).assign(
    mom_pct=lambda d: d.groupby("town")["avg_price"].pct_change() * 100
)

vol = avg_price.groupby("town").agg(
    mom_pct_std=("mom_pct", "std"),
    mom_pct_iqr=("mom_pct", lambda x: x.quantile(0.75) - x.quantile(0.25)),
)

diagnostics = counts_summary.join(vol)
diagnostics.sort_values(["median_monthly_n", "mom_pct_std"], ascending=[True, False])

# %% [md]
# # Descriptive summary
# - Remember that it is still a time series, for brief overview only

# %% [md]
# ### Region table

# %%
df.groupby("region")["resale_price"].describe().sort_values(by="mean", ascending=False)

# %% [md]
# ### Town table

# %%
df.groupby("town")["resale_price"].describe().sort_values(by="mean", ascending=False)

# %% [md]
# # Visualise all

# %%
sns.histplot(data=df, x="resale_price", hue="region", kde=True, bins=30)

# %% [md]
# ### West

# %%
region = "WEST REGION"
subset = df[df["region"] == region]

stats = (
    subset["resale_price"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .to_frame()
    .rename(columns={"resale_price": region})
)
stats.loc["skew"] = skew(subset["resale_price"], bias=False)
stats.loc["kurtosis"] = kurtosis(subset["resale_price"], bias=False)

print(stats)

sns.histplot(
    data=subset,
    x="resale_price",
    hue="region",
    kde=True,
    bins=30,
)

# %% [md]
# ### East

# %%
region = "EAST REGION"
subset = df[df["region"] == region]

stats = (
    subset["resale_price"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .to_frame()
    .rename(columns={"resale_price": region})
)
stats.loc["skew"] = skew(subset["resale_price"], bias=False)
stats.loc["kurtosis"] = kurtosis(subset["resale_price"], bias=False)

print(stats)

sns.histplot(
    data=subset,
    x="resale_price",
    hue="region",
    kde=True,
    bins=30,
)

# %% [md]
# ### North

# %%
region = "NORTH REGION"
subset = df[df["region"] == region]

stats = (
    subset["resale_price"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .to_frame()
    .rename(columns={"resale_price": region})
)
stats.loc["skew"] = skew(subset["resale_price"], bias=False)
stats.loc["kurtosis"] = kurtosis(subset["resale_price"], bias=False)

print(stats)

sns.histplot(
    data=subset,
    x="resale_price",
    hue="region",
    kde=True,
    bins=30,
)

# %% [md]
# ### North-East

# %%
region = "NORTH-EAST REGION"
subset = df[df["region"] == region]

stats = (
    subset["resale_price"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .to_frame()
    .rename(columns={"resale_price": region})
)
stats.loc["skew"] = skew(subset["resale_price"], bias=False)
stats.loc["kurtosis"] = kurtosis(subset["resale_price"], bias=False)

print(stats)

sns.histplot(
    data=subset,
    x="resale_price",
    hue="region",
    kde=True,
    bins=30,
)


# %% [md]
# ### Central

# %%
region = "CENTRAL REGION"
subset = df[df["region"] == region]

stats = (
    subset["resale_price"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .to_frame()
    .rename(columns={"resale_price": region})
)
stats.loc["skew"] = skew(subset["resale_price"], bias=False)
stats.loc["kurtosis"] = kurtosis(subset["resale_price"], bias=False)

print(stats)

sns.histplot(
    data=subset,
    x="resale_price",
    hue="region",
    kde=True,
    bins=30,
)

# %% [md]
# ### Visualise town

# %%
towns = df["town"].unique().tolist()

for town in towns:
    subset = df[df["town"] == town]

    stats = (
        subset["resale_price"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .to_frame()
        .rename(columns={"resale_price": town})
    )
    stats.loc["skew"] = skew(subset["resale_price"], bias=False)
    stats.loc["kurtosis"] = kurtosis(subset["resale_price"], bias=False)

    from IPython.display import display

    display(stats.T.round(2))
    plt.figure(figsize=(6, 4))
    sns.histplot(data=subset, x="resale_price", kde=True, bins=30, color="steelblue")
    plt.title(f"{town} — Resale Price Distribution")
    plt.xlabel("Resale Price (SGD)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

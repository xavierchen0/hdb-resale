# %% [md]
# # Explore price against town and region

# %%
import pandas as pd
import sys
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(Path().resolve() / "src")

from src.config import DATA_DIR

df = pd.read_pickle(DATA_DIR / "cleaned_data.pkl")
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

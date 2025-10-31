# %% [md]
# # Feature Engineering: SORA

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

# %% [markdown]
# Read the monthly average SORA rates dataset

# %%
SORA_RATES = DATA_DIR / "monthly_avg_sora_rates.csv"
df_sora = pd.read_csv(SORA_RATES)

# %% [markdown]
# Convert 'Month' to datetime64[ns]

# %%
df_sora["Month"] = pd.to_datetime(df_sora["Month"])

# %% [md]
# Rename columns

# %%
df_sora = df_sora.rename(columns={"Month": "month", "SORA": "sora"})
df_sora

# %% [md]
# Remove october 2025

# %%
df_sora = (
    df_sora.set_index("month").drop(index=pd.to_datetime("2025-10-01")).reset_index()
)
df_sora

# %% [md]
# Sort the dataframe by month

# %%
df_sora = df_sora.sort_values(by="month")
df_sora

# %% [markdown]
# Check dtypes

# %%
df_sora.dtypes

# %% [markdown]
# Display df

# %%
df_sora

# %% [markdown]
# # Export

# %%
df_sora.to_parquet(DATA_DIR / "feat" / "feat_sora_rates.parquet")

# %% [md]
# # Perform time series decomposition of resale price

# %% [md]
# # Read data

# %%
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import sys
from pathlib import Path
import matplotlib.pyplot as plt

project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DATA_DIR

df = pd.read_pickle(DATA_DIR / "cleaned_rawdata.pkl")
df

# %% [md]
# # Average Resale Price TSD using LOESS

# %%
average_monthly_resale = df.groupby("month")["resale_price"].mean()
average_monthly_resale

# %%
stl = STL(average_monthly_resale, period=12, seasonal=13)

res = stl.fit()

res.plot()

plt.show()

# %% [md]
# # Log transform resale price

# %%
df["log_resale"] = np.log(df["resale_price"])

average_monthly_log_resale = df.groupby("month")["log_resale"].mean()
average_monthly_log_resale

# %%
stl = STL(average_monthly_log_resale, period=12, seasonal=13)

res = stl.fit()

res.plot()

plt.show()

# %% [md]
# # Average resale price / sqm

# %%
df["price_per_sqm"] = df["resale_price"] / df["floor_area_sqm"]

average_monthly_price_per_sqm = df.groupby("month")["price_per_sqm"].mean()
average_monthly_price_per_sqm

# %%
stl = STL(average_monthly_price_per_sqm, period=12, seasonal=13)

res = stl.fit()

res.plot()

plt.show()

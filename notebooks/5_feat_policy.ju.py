# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python (.venv)
#     language: python
#     name: .venv
# ---

# %%
import sys
from pathlib import Path
import pandas as pd

project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# %%
from src.config import DATA_DIR
pd.set_option('display.max_columns', None)
df = pd.read_parquet(DATA_DIR / "base_dataset.parquet")

df
df.to_csv("output_dataset.csv" , index=False)


# %% [markdown]
# # PURPOSE
# This notebook serves to create the features for policies that is to be merged with the main dataset

# %% [markdown]
# ## Identified Policies Impacting HDB Resale Prices (2017-2025)
#
# | Date (YYYY-MM) | Policy | Why it should move resale prices | Link |
# |---|---|---|---|
# | 2018-02 | Proximity Housing Grant (PHG) enhanced | Subsidies raise effective income, lifting reservation prices—strongest for first‑timers and smaller/mid‑sized units. | [source](https://www.mnd.gov.sg/newsroom/press-releases/view/closer-families-stronger-ties-enhanced-proximity-housing-grant-to-help-more-families-live-closer-together) |
# | 2019-05 | CPF usage & HDB loan rules shifted to the “lease-to-age-95” test | Credit constraint channel: reduced borrowing capacity lowers reservation prices, with larger effects where prices are high or leases are short. CPF usage changes shift the effective downpayment/loan mix; expect price effects concentrated in older‑lease submarkets. | [source](https://www.mnd.gov.sg/newsroom/press-releases/view/more-flexibility-to-buy-a-home-for-life-while-safeguarding-retirement-adequacy) |
# | 2019-09 | Enhanced CPF Housing Grant (EHG) & higher income ceilings | Credit constraint channel: reduced borrowing capacity lowers reservation prices, with larger effects where prices are high or leases are short. Higher grants raise first‑timer purchasing power, supporting prices for smaller/mid‑sized flats and in adjacent towns. Subsidies raise effective income, lifting reservation prices—strongest for first‑timers and smaller/mid‑sized units. CPF usage changes shift the effective downpayment/loan mix; expect price effects concentrated in older‑lease submarkets. Increased BTO cadence diverts demand from resale and eases prices; lulls do the opposite. | [source](https://www.mnd.gov.sg/newsroom/press-releases/view/more-affordable-and-accessible-homes-for-singaporeans-enhanced-cpf-housing-grant-for-first-timers-and-higher-income-ceilings-1) |
# | 2021-12 | Measures To Cool The Property Market | Demand‑suppression via higher transaction costs reduces investor/upgrader demand, exerting near‑term downward pressure on resale prices—especially for larger flats and upgrader‑heavy towns. Credit tightening lowers effective purchasing power and borrowing capacity, compressing feasible bid prices; stronger effects for high‑priced and older‑lease units. Credit constraint channel: reduced borrowing capacity lowers reservation prices, with larger effects where prices are high or leases are short. | [source](https://www.mas.gov.sg/news/media-releases/2021/measures-to-cool-the-property-market) |
# | 2022-09 | Measures to Promote Sustainable Conditions in the Property Market by Ensuring Prudent Borrowing and Moderating Demand | Credit tightening lowers effective purchasing power and borrowing capacity, compressing feasible bid prices; stronger effects for high‑priced and older‑lease units. Credit constraint channel: reduced borrowing capacity lowers reservation prices, with larger effects where prices are high or leases are short. Mortgage‑servicing caps bind more households, curbing demand and pressuring prices—stronger in higher‑priced submarkets. Subsidies raise effective income, lifting reservation prices—strongest for first‑timers and smaller/mid‑sized units. | [source](https://www.hdb.gov.sg/about-us/news-and-publications/press-releases/29092022-propertymeasures2022) |
# | 2023-02 | Budget 2023: higher CPF Housing Grant for resale first-timers | Subsidies raise effective income, lifting reservation prices—strongest for first‑timers and smaller/mid‑sized units. CPF usage changes shift the effective downpayment/loan mix; expect price effects concentrated in older‑lease submarkets. | [source](https://www.hdb.gov.sg/business/estate-agents-and-salespersons/letters-to-keos/Increased-CPF-Housing-Grant-for-resale-flat-buyers) |
# | 2024-08 | Measures to Cool the HDB Resale Market and Provide More Support for First-Time Home Buyers | Credit tightening lowers effective purchasing power and borrowing capacity, compressing feasible bid prices; stronger effects for high‑priced and older‑lease units. Credit constraint channel: reduced borrowing capacity lowers reservation prices, with larger effects where prices are high or leases are short. Subsidies raise effective income, lifting reservation prices—strongest for first‑timers and smaller/mid‑sized units. | [source](https://www.hdb.gov.sg/about-us/news-and-publications/press-releases/Measures-to-Cool-the-HDB-Resale-Market-and-Provide-More-Support-for-First-Time-Home-Buyers) |
# | 2024-10 | New Flat Classification Framework | Stronger classification restrictions lower resale flexibility/liquidity premia, moderating prices and shifting demand to substitutes. Prime/Plus‑style restrictions reduce resale liquidity and speculative premia, redirecting demand to substitute towns. Increased BTO cadence diverts demand from resale and eases prices; lulls do the opposite. | [source](https://www.hdb.gov.sg/about-us/news-and-publications/press-releases/New-Flat-Classification-Framework) |

# %% [markdown]
# ### Create Dataframe for Policies

# %%
import pandas as pd
import datetime as dt

# %%
policy_allocation = ({
    'test_01':"2017-01",
    'phg_18':'2018-02',         # Proximity Housing Grant (PHG) enhanced
    'lta95_19': '2019-05',      # CPF usage & HDB loan rules shifted to the “lease-to-age-95” test
    'ehg_19': '2019-09',        # Enhanced CPF Housing Grant (EHG) & higher income ceilings
    'cooling_21':'2021-12',     # Measures To Cool The Property Market
    'promote_22':'2022-09',     # Measures to Promote Sustainable Conditions in the Property Market by Ensuring Prudent Borrowing and Moderating Demand
    'budget_23': '2023-02',     # Budget 2023: higher CPF Housing Grant for resale first-timers
    'cooling_24':'2024-08',     # Measures to Cool the HDB Resale Market and Provide More Support for First-Time Home Buyers
    'flatclass_24':'2024-10'    # New Flat Classification Framework
})

POLICY_COLUMN_ORDER = list(policy_allocation.keys())

# %%
df_policies = pd.DataFrame(
    list(policy_allocation.items()), 
    columns=['policyname', 'month']
)

df_policies

# %%
df_policies['month'] = pd.to_datetime(df_policies['month'])
df_policies['Value'] = 1

df_policies

# %%
START_DATE = "2017-01-01"
END_DATE = "2025-09-01"
START_DATE = pd.to_datetime(START_DATE)
END_DATE = pd.to_datetime(END_DATE)
full_month_range = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')

df_policy_wide = df_policies.pivot_table(
    index='month',
    columns='policyname',
    values='Value'
)

df_policy_wide

# %%
df_policy_wide = df_policy_wide.reindex(full_month_range)
df_policy_wide.ffill(inplace=True)
df_policy_event_wide = df_policy_wide.fillna(0).astype(int)

df_policy_event_wide


# %%
df_policy_event_wide.reset_index(names=['month'], inplace=True)
df_policy_event_wide

# %%
# merge the df
df_merged = pd.merge(
    df,
    df_policy_event_wide,
    on="month",
    how="left"
)

df_merged

# %%
#df_merged.to_parquet("../data/base_dataset2.parquet")

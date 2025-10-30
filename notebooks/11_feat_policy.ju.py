# %% [md]
# # Feature Engineering: HDB related policies

# %% [md]
# Read the base dataset

# %%
from os import stat_result
import pandas as pd
from pathlib import Path
import sys
import geopandas as gpd
from pandas.core.base import ensure_wrapped_if_datetimelike
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt

# Allow importing from src/
project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DATA_DIR

df = gpd.read_parquet(DATA_DIR / "base_dataset.parquet")
df

# %% [markdown]
# Identified policies impacting hdb resale prices (2017-2025)
#
# | date (yyyy-mm) | policy | why it should move resale prices | link |
# |---|:---|:---|---|
# | 2010-02 | Introduction of SSD | First introduce SSD Not applied to HDB resale, but can make hdb resale more attractive now considering that private properties will be more expensive | [source](https://www.mas.gov.sg/news/media-releases/2011/measures-to-maintain-a-stable-and-sustainable-property-market) | [source](https://www.mas.gov.sg/news/media-releases/2010/measures-to-ensure-a-stable-and-sustainable-property-market)|
# | 2010-08 | Revision of SSD | First introduce SSD Not applied to HDB resale, but can make hdb resale more attractive now considering that private properties will be more expensive | [source](https://www.mas.gov.sg/news/media-releases/2011/measures-to-maintain-a-stable-and-sustainable-property-market) | [source](https://www.mas.gov.sg/news/media-releases/2010/measures-to-maintain-a-stable-and-sustainable-property-market)|
# | 2011-01 | Seller's Stamp Duty (SSD) tightened | In January 2011, the government raised Seller’s Stamp Duty rates and extended the holding period, while tightening loan-to-value limits to curb speculative property purchases. These cooling measures reduced speculative demand and tightened credit, indirectly tempering HDB resale price growth through weaker market sentiment and lower affordability pressures. That discourages speculative “buy now, flip in 12 months” behavior. | [source](https://www.mas.gov.sg/news/media-releases/2011/measures-to-maintain-a-stable-and-sustainable-property-market) |
# | 2011-12 | Additional Buyer's Stamp Duty for a Stable and Sustainable Property Market |In December 2011 the government introduced the Additional Buyer’s Stamp Duty (ABSD), a tax payable in addition to the regular Buyer’s Stamp Duty for residential property purchases. The ABSD aims to curb speculative buying and investment demand—especially by foreigners and buyers acquiring multiple homes—to ensure the housing market remains more affordable and aligned with fundamentals. | [source](https://www.mas.gov.sg/news/media-releases/2011/absd-for-a-stable-and-sustainable-property-market) |
# | 2013-01 | ABSD raised again | | [source](https://www.mas.gov.sg/news/media-releases/2013/additional-measures-to-ensure-a-stable-and-sustainable-property-market) |
# | 2013-06 | MAS Introduces Debt Servicing Framework for Property Loans | Banks now must ensure a borrower’s total monthly debt obligations stay within a fixed threshold (initially 60% of gross monthly income), counting not just the housing loan but also car loans, credit cards, etc. | [source](https://www.mas.gov.sg/news/media-releases/2013/mas-introduces-debt-servicing-framework-for-property-loans)|
# | 2014-03 | HDB kills open COV quoting | This cooled runaway expectations in hot estates, and it’s widely associated with the start of a downtrend in HDB resale prices after 2013 highs. | [source](https://www.mnd.gov.sg/newsroom/parliament-matters/q-as/view/written-answer-by-ministry-of-national-development-on-whether-there-is-a-trend-of-increasing-cash-over-valuation-(cov)-for-resale-hdb-flats-what-are-the-reasons-for-the-increasing-trend-and-what-has-been-the-rate-of-increase-in-the-past-one-year) |
# | 2017-03 | SSD partially relaxed | This makes it cheaper to sell “earlier.” | [source](https://www.mas.gov.sg/news/media-releases/2017/joint-press-release-on-measures-relating-to-residential-property) |
# | 2018-02 | Buyer’s Stamp Duty (BSD) top tier | This increase raises the upfront transaction cost for buyers of higher-value homes, reducing affordability and speculative demand, which in turn may moderate resale price growth in the public and private housing markets. | [source](https://sg.finance.yahoo.com/news/singapore-budget-2018-buyer-stamp-100803181.html?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8&guce_referrer_sig=AQAAAE2mGkWYZXGDp6FcVzAVtJ7GPiBP-2evlqG1TahqN2Tz2wO2QA42pK0YTVeU3tNONL61Wipv1CuV3eBEOepyFGy3weqRyl9zYeny5cl_s8XCSpeHrD-4pbD1D14NZ_xN-bjCWfGqPiPzlQsD1gwZenW4KTI1Uf4rTnynn9Thsfhb) |
# | 2018-02 | proximity housing grant (phg) enhanced | subsidies raise effective income, lifting reservation prices—strongest for first‑timers and smaller/mid‑sized units. | [source](https://www.mnd.gov.sg/newsroom/press-releases/view/closer-families-stronger-ties-enhanced-proximity-housing-grant-to-help-more-families-live-closer-together) |
# | 2019-05 | cpf usage & hdb loan rules shifted to the “lease-to-age-95” test | credit constraint channel: reduced borrowing capacity lowers reservation prices, with larger effects where prices are high or leases are short. cpf usage changes shift the effective downpayment/loan mix; expect price effects concentrated in older‑lease submarkets. | [source](https://www.mnd.gov.sg/newsroom/press-releases/view/more-flexibility-to-buy-a-home-for-life-while-safeguarding-retirement-adequacy) |
# | 2019-09 | enhanced cpf housing grant (ehg) & higher income ceilings | credit constraint channel: reduced borrowing capacity lowers reservation prices, with larger effects where prices are high or leases are short. higher grants raise first‑timer purchasing power, supporting prices for smaller/mid‑sized flats and in adjacent towns. subsidies raise effective income, lifting reservation prices—strongest for first‑timers and smaller/mid‑sized units. cpf usage changes shift the effective downpayment/loan mix; expect price effects concentrated in older‑lease submarkets. increased bto cadence diverts demand from resale and eases prices; lulls do the opposite. | [source](https://www.mnd.gov.sg/newsroom/press-releases/view/more-affordable-and-accessible-homes-for-singaporeans-enhanced-cpf-housing-grant-for-first-timers-and-higher-income-ceilings-1) |
# | 2021-12 | measures to cool the property market | demand‑suppression via higher transaction costs reduces investor/upgrader demand, exerting near‑term downward pressure on resale prices—especially for larger flats and upgrader‑heavy towns. credit tightening lowers effective purchasing power and borrowing capacity, compressing feasible bid prices; stronger effects for high‑priced and older‑lease units. credit constraint channel: reduced borrowing capacity lowers reservation prices, with larger effects where prices are high or leases are short. | [source](https://www.mas.gov.sg/news/media-releases/2021/measures-to-cool-the-property-market) |
# | 2022-09 | measures to promote sustainable conditions in the property market by ensuring prudent borrowing and moderating demand | credit tightening lowers effective purchasing power and borrowing capacity, compressing feasible bid prices; stronger effects for high‑priced and older‑lease units. credit constraint channel: reduced borrowing capacity lowers reservation prices, with larger effects where prices are high or leases are short. mortgage‑servicing caps bind more households, curbing demand and pressuring prices—stronger in higher‑priced submarkets. subsidies raise effective income, lifting reservation prices—strongest for first‑timers and smaller/mid‑sized units. | [source](https://www.hdb.gov.sg/about-us/news-and-publications/press-releases/29092022-propertymeasures2022) |
# | 2023-02 | budget 2023: higher cpf housing grant for resale first-timers | subsidies raise effective income, lifting reservation prices—strongest for first‑timers and smaller/mid‑sized units. cpf usage changes shift the effective downpayment/loan mix; expect price effects concentrated in older‑lease submarkets. | [source](https://www.hdb.gov.sg/business/estate-agents-and-salespersons/letters-to-keos/increased-cpf-housing-grant-for-resale-flat-buyers) |

# %%
policy_allocation = {
    "ssd_intro_2010": "2010-02",  # introduction to SSD
    "ssd_rev_2010": "2010-08",  # revision to SSD
    "ssd_hike_2011": "2011-01",  # SSD tightened
    "absd_into_2011": "2011-12",  # ABSD introduced
    "absd_hike_2013": "2013-01",  # ABSD raised again
    "tdsr_2013": "2013-06",  # Debt servicing framework
    "cov_rule_change_2014": "2014-03",  # COV quoting removed
    "ssd_relax_2017": "2017-03",  # SSD partially relaxed
    "bsd_hike_2018": "2018-02",  # BSD top tier raised
    "phg_18": "2018-02",  # Proximity Housing Grant (PHG) enhanced
    "lta95_19": "2019-05",  # CPF usage & HDB loan rules shifted to the “lease-to-age-95” test
    "ehg_19": "2019-09",  # Enhanced CPF Housing Grant (EHG) & higher income ceilings
    "cooling_21": "2021-12",  # Measures To Cool The Property Market
    "promote_22": "2022-09",  # Measures to Promote Sustainable Conditions in the Property Market by Ensuring Prudent Borrowing and Moderating Demand
    "budget_23": "2023-02",  # Budget 2023: higher CPF Housing Grant for resale first-timers
    "cooling_24": "2024-08",  # Measures to Cool the HDB Resale Market and Provide More Support for First-Time Home Buyers
}

POLICY_COLUMN_ORDER = list(policy_allocation.keys())

# %% [md]
# Create policies dataframe

# %%
df_policies = pd.DataFrame(
    list(policy_allocation.items()), columns=["policyname", "month"]
)

df_policies

# %% [md]
# Change to the right data type for month

# %%
df_policies["month"] = pd.to_datetime(df_policies["month"])
df_policies

# %% [md]
# Create a column of 1s to be used as a boolean indicator; transformed later

# %%
df_policies["Value"] = 1
df_policies

# %% [md]
# Transform to pivot table

# %%
df_policy_wide = df_policies.pivot_table(
    index="month", columns="policyname", values="Value"
)

df_policy_wide

# %% [md]
# markdown

# %%
START_DATE = "2017-01-01"
END_DATE = "2025-09-01"
START_DATE = pd.to_datetime(START_DATE)
END_DATE = pd.to_datetime(END_DATE)

full_month_range = pd.date_range(
    start=pd.to_datetime("2010-01-01"), end=END_DATE, freq="MS"
)

df_policy_wide = df_policy_wide.reindex(full_month_range)
df_policy_wide.ffill(inplace=True)
df_policy_event_wide = df_policy_wide.fillna(0).astype(int)

df_policy_event_wide

# %% [md]
# rename index to month

# %%
df_policy_event_wide.reset_index(names=["month"], inplace=True)
df_policy_event_wide

# %% [md]
# Add ssd values using ssd max rate <br>
# e.g. The SSD will be applied at the standard ad valorem stamp duty rates for the conveyance, assignment or transfer of property: 1% for the first 180,000 of the consideration, 2% for the next 180,000, and 3% for the balance.
# <br>
# [source](https://www.iras.gov.sg/taxes/stamp-duty/for-property/selling-or-disposing-property/seller%27s-stamp-duty-(ssd)-for-residential-property)

# %%
ssd_max_rates = {"2010-02": 3, "2011-01": 16, "2017-03": 12, "2025-07": 16}

df_rate = pd.DataFrame(
    list(ssd_max_rates.items()), columns=["start_date", "ssd_max_rate"]
)
df_rate["start_date"] = pd.to_datetime(df_rate["start_date"])
df_rate

# %% [md]
# Add ssd max hold before ssd is no longer applied

# %%
ssd_max_hold_periods = {
    "2010-02": 1,
    "2010-09": 3,
    "2011-01": 4,
    "2017-03": 3,
    "2025-07": 4,
}

df_hold = pd.DataFrame(
    list(ssd_max_hold_periods.items()), columns=["start_date", "ssd_max_hold_yrs"]
)
df_hold["start_date"] = pd.to_datetime(df_hold["start_date"])
df_hold

# %% [md]
# Generate full monthly index

# %%
start = min(df_rate["start_date"].min(), df_hold["start_date"].min())
end = pd.Timestamp("2025-12-01")  # adjust to your dataset end
monthly_index = pd.date_range(start, end, freq="MS")

df_month = pd.DataFrame(index=monthly_index)
df_month

# %% [md]
# Create the merged dataframe

# %%
df_month = (
    df_month.merge(df_rate, how="outer", left_index=True, right_on="start_date")
    .sort_values("start_date")
    .ffill()
    .set_index("start_date")
    .merge(df_hold, how="outer", left_index=True, right_on="start_date")
    .sort_values("start_date")
    .ffill()
    .set_index("start_date")
)
df_month

# %% [md]
# reset the dataframe and rename the start date column to month

# %%
df_month = df_month.reset_index().rename(columns={"start_date": "month"})
df_month

# %% [md]
# Merge with the og dataframe

# %%
df_policy_event_wide = df_policy_event_wide.merge(df_month, on="month", how="left")
df_policy_event_wide

# %% [md]
# Add bsd values for residential and non-residential

# %%
bsd_max_rate_res = {"2010-01": 3, "2018-02": 4, "2023-02": 6}

df_bsd_res = pd.DataFrame(
    list(bsd_max_rate_res.items()), columns=["start_date", "bsd_max_rate_res"]
)
df_bsd_res["start_date"] = pd.to_datetime(df_bsd_res["start_date"])
df_bsd_res

# %%
bsd_max_rate_non_res = {"2010-01": 3, "2018-02": 3, "2023-02": 5}

df_bsd_non_res = pd.DataFrame(
    list(bsd_max_rate_non_res.items()), columns=["start_date", "bsd_max_rate_non_res"]
)
df_bsd_non_res["start_date"] = pd.to_datetime(df_bsd_non_res["start_date"])
df_bsd_non_res

# %% [md]
# Generate full monthly index

# %%
start = min(df_bsd_res["start_date"].min(), df_bsd_non_res["start_date"].min())
end = pd.Timestamp("2025-12-01")  # adjust to your dataset end
monthly_index = pd.date_range(start, end, freq="MS")

df_month = pd.DataFrame(index=monthly_index)
df_month

# %% [md]
# Create the merged dataframe

# %%
df_month = (
    df_month.merge(df_bsd_res, how="outer", left_index=True, right_on="start_date")
    .sort_values("start_date")
    .ffill()
    .set_index("start_date")
    .merge(df_bsd_non_res, how="outer", left_index=True, right_on="start_date")
    .sort_values("start_date")
    .ffill()
    .set_index("start_date")
)
df_month

# %% [md]
# reset the dataframe and rename the start date column to month

# %%
df_month = df_month.reset_index().rename(columns={"start_date": "month"})
df_month

# %% [md]
# Merge with the og dataframe

# %%
df_policy_event_wide = df_policy_event_wide.merge(df_month, on="month", how="left")
df_policy_event_wide

# %% [md]
# Add LTV and ABSD and maximum loan tenure
# <br>
# [source](https://www.mas.gov.sg/publications/macroprudential-policies-in-singapore)
# <br>
# [source](https://dollarsandsense.sg/history-property-cooling-measures-singapore-years/)
# <br>
# focused on individuals, not non-individuals (corporation, trusts)

# %% [md]
# ltv_0_false_above_30_above_65 refers to no existing loans AND False to (loan tenure > 30 years or beyond age 65)
# <br>
# ltv_0_true_above_30_above_65 refers to no existing loans AND True to (loan tenure > 30 years or beyond age 65)
# <br>
# ltv_1_false_above_30_above_65 refers to 1 existing loan AND False to (loan tenure > 30 years or beyond age 65)
# <br>
# ltv_1_true_above_30_above_65 refers to 1 exisiting loan AND True to (loan tenure > 30 years or beyond age 65)
# <br>
# sg_absd_2 refers to absd applied to 2nd property for singaporeans
# <br>
# sg_absd_3 refers to absd applied to 3rd property and above for singaporeans
# <br>
# pr_absd_1 refers to absd applied to 1st property PRs
# <br>
# pr_absd_2 refers to absd applied to 2nd property and above for PRs
# <br>
# pr_absd_3 refers to absd applied to 3rd property and above for PRs
#


# %%
ltv_0_false_above_30_above_65 = {
    "2010-01": 90,
    "2010-02": 80,
    "2010-08": 70,
    "2018-07": 75,
}

ltv_0_true_above_30_above_65 = {
    "2010-01": 90,
    "2010-02": 80,
    "2010-08": 70,
    "2012-10": 60,
    "2018-07": 55,
}

ltv_1_false_above_30_above_65 = {
    "2010-01": 90,
    "2010-02": 80,
    "2010-08": 70,
    "2012-10": 40,
    "2013-01": 50,
    "2018-07": 45,
}

ltv_1_true_above_30_above_65 = {
    "2010-01": 90,
    "2010-02": 80,
    "2010-08": 60,
    "2012-10": 40,
    "2013-01": 30,
    "2018-07": 25,
}

sg_absd_2 = {
    "2010-01": 0,
    "2013-01": 7,
    "2018-07": 12,
    "2021-12": 17,
    "2023-04": 20,
}

sg_absd_3 = {
    "2010-01": 0,
    "2011-12": 3,
    "2013-01": 10,
    "2018-07": 15,
    "2021-12": 25,
    "2023-04": 30,
}

pr_absd_1 = {
    "2010-01": 0,
    "2013-01": 5,
}

pr_absd_2 = {
    "2010-01": 0,
    "2011-12": 3,
    "2013-01": 10,
    "2018-07": 15,
    "2021-12": 25,
    "2023-04": 30,
}

pr_absd_3 = {
    "2010-01": 0,
    "2011-12": 3,
    "2013-01": 10,
    "2018-07": 15,
    "2021-12": 30,
    "2023-04": 35,
}

tdsr = {"2010-01": 0, "2013-06": 60, "2021-12": 55}

df_ltv_0_false_above_30_above_65 = pd.DataFrame(
    list(ltv_0_false_above_30_above_65.items()),
    columns=["start_date", "ltv_0_false_above_30_above_65"],
)
df_ltv_0_false_above_30_above_65["start_date"] = pd.to_datetime(
    df_ltv_0_false_above_30_above_65["start_date"]
)

df_ltv_0_true_above_30_above_65 = pd.DataFrame(
    list(ltv_0_true_above_30_above_65.items()),
    columns=["start_date", "ltv_0_true_above_30_above_65"],
)
df_ltv_0_true_above_30_above_65["start_date"] = pd.to_datetime(
    df_ltv_0_true_above_30_above_65["start_date"]
)

df_ltv_1_false_above_30_above_65 = pd.DataFrame(
    list(ltv_1_false_above_30_above_65.items()),
    columns=["start_date", "ltv_1_false_above_30_above_65"],
)
df_ltv_1_false_above_30_above_65["start_date"] = pd.to_datetime(
    df_ltv_1_false_above_30_above_65["start_date"]
)

df_ltv_1_true_above_30_above_65 = pd.DataFrame(
    list(ltv_1_true_above_30_above_65.items()),
    columns=["start_date", "ltv_1_true_above_30_above_65"],
)
df_ltv_1_true_above_30_above_65["start_date"] = pd.to_datetime(
    df_ltv_1_true_above_30_above_65["start_date"]
)

df_sg_absd_2 = pd.DataFrame(
    list(sg_absd_2.items()),
    columns=["start_date", "sg_absd_2"],
)
df_sg_absd_2["start_date"] = pd.to_datetime(df_sg_absd_2["start_date"])

df_sg_absd_3 = pd.DataFrame(
    list(sg_absd_3.items()),
    columns=["start_date", "sg_absd_3"],
)
df_sg_absd_3["start_date"] = pd.to_datetime(df_sg_absd_3["start_date"])

df_pr_absd_1 = pd.DataFrame(
    list(pr_absd_1.items()),
    columns=["start_date", "pr_absd_1"],
)
df_pr_absd_1["start_date"] = pd.to_datetime(df_pr_absd_1["start_date"])

df_pr_absd_2 = pd.DataFrame(
    list(pr_absd_2.items()),
    columns=["start_date", "pr_absd_2"],
)
df_pr_absd_2["start_date"] = pd.to_datetime(df_pr_absd_2["start_date"])

df_pr_absd_3 = pd.DataFrame(
    list(pr_absd_3.items()),
    columns=["start_date", "pr_absd_3"],
)
df_pr_absd_3["start_date"] = pd.to_datetime(df_pr_absd_3["start_date"])

df_tdsr = pd.DataFrame(
    list(tdsr.items()),
    columns=["start_date", "tdsr"],
)
df_tdsr["start_date"] = pd.to_datetime(df_tdsr["start_date"])

df_arr = [
    df_ltv_0_false_above_30_above_65,
    df_ltv_0_true_above_30_above_65,
    df_ltv_1_false_above_30_above_65,
    df_ltv_1_true_above_30_above_65,
    df_sg_absd_2,
    df_sg_absd_3,
    df_pr_absd_1,
    df_pr_absd_2,
    df_pr_absd_3,
    df_tdsr,
]
# %% [md]
# Generate full monthly index

# %%
df_arr_idxed = [d.set_index("start_date").sort_index() for d in df_arr]

df_policy = pd.concat(df_arr_idxed, axis=1).sort_index()

start = df_policy.index.min()
end = pd.Timestamp("2025-12-01")  # your chosen end
monthly_index = pd.date_range(start, end, freq="MS")

df_policy = df_policy.reindex(monthly_index).ffill()

df_policy.index.name = "start_date"
df_policy

# %% [md]
# reset the dataframe and rename the start date column to month

# %%
df_policy = df_policy.reset_index().rename(columns={"start_date": "month"})
df_policy

# %% [md]
# Merge with the og dataframe

# %%
df_policy_event_wide = df_policy_event_wide.merge(df_policy, on="month", how="left")
df_policy_event_wide

# %% [md]
# Filter for interested dates

# %%
df_policy_event_wide = df_policy_event_wide.loc[
    df_policy_event_wide["month"].isin(pd.date_range(START_DATE, END_DATE, freq="MS"))
]
df_policy_event_wide

# %% [md]
# Check for null

# %%
df_policy_event_wide[df_policy_event_wide.isna().any(axis=1)]

# %% [md]
# Check dtype

# %%
df_policy_event_wide.dtypes

# %% [md]
# Final Check

# %%
df_policy_event_wide

# %% [md]
# # Export

# %%
df_policy_event_wide.to_parquet(DATA_DIR / "feat" / "feat_policy.parquet")

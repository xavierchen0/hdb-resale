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

df = pd.read_pickle(DATA_DIR / "cleaned_rawdata.pkl")
df

# %%
COE_DATA = DATA_DIR / "COE.csv"

df_coe = pd.read_csv(COE_DATA)
df_coe

# %%
df_coe["month"] = pd.to_datetime(df_coe["month"])
START_DATE = "2017-01-01"
END_DATE = "2025-09-01"
START_DATE = pd.to_datetime(START_DATE)
END_DATE = pd.to_datetime(END_DATE)

df_coe1 = df_coe[
    (df_coe["month"] >= START_DATE) & 
    (df_coe["month"] <= END_DATE)
].copy().reset_index()
df_coe1

# %%
# # filter cat A and B
# relevant_categories = ["Category A", "Category B"]
# df_coe_filtered = df_coe1[df_coe1["vehicle_class"].isin(relevant_categories)].copy().reset_index()
# df_coe_filtered

# %%
#take the mean of both bids
df_coe_pivot = df_coe1.pivot_table(
    index="month",
    columns="vehicle_class",
    values="premium",
    aggfunc="mean"
).reset_index()

df_coe_pivot.rename(columns={
    "Category A": "coe_a",
    "Category B": "coe_b",
    "Category C": "coe_c",
    "Category D": "coe_d",
    "Category E": "coe_e"
}, inplace=True)

# merge the df
df_merged1 = pd.merge(
    df,
    df_coe_pivot,
    on="month",
    how="left"
)

df_merged1

# %%
CPI_DATA = DATA_DIR / "CPI.csv"

df_cpi = pd.read_csv(CPI_DATA)
df_cpi.dtypes

# %%
df_cpi_long = df_cpi.melt(
    id_vars=["DataSeries"],
    var_name="month_str",
    value_name="cpi_index"
)

df_cpi_long

# %%
df_cpi_all = df_cpi_long[df_cpi_long["DataSeries"] == "All Items"].copy()


# convert the format from "YYYYMon" (e.g., "2025Aug") to "YYYY-MM-01".
df_cpi_all["month"] = pd.to_datetime(
    df_cpi_all["month_str"], 
    format="%Y%b"
)
df_cpi_all["cpi_index"] = pd.to_numeric(df_cpi_all["cpi_index"])
df_cpi_all["cpi_index"] = df_cpi_all["cpi_index"].astype('float64')
df_cpi_all

# %%
df_cpi_all.sort_values(by='month', ascending=True, inplace=True)
df_cpi_all['cpi_pct_change'] = (df_cpi_all['cpi_index'].pct_change() * 100)

df_cpi_all

# %%
df_cpi_final = df_cpi_all[
    (df_cpi_all["month"] >= START_DATE) & 
    (df_cpi_all["month"] <= END_DATE)][["month", "cpi_pct_change", "cpi_index"]].copy()

df_cpi_final

# %%
last_cpi_value = df_cpi_final["cpi_index"].iloc[-1] 
    
new_row = pd.DataFrame({
    "month": [END_DATE], 
    "cpi_index": [last_cpi_value],
    "cpi_pct_change" : 0
})
    
df_cpi_final = pd.concat([df_cpi_final, new_row], ignore_index=True)
df_cpi_final

# %%
# merge
df_merged2 = pd.merge(
    df_merged1,
    df_cpi_final,
    on="month",
    how="left"
)

df_merged2

# %%
PPI_DATA = DATA_DIR / "Private_Property_Index.csv"

df_ppi = pd.read_csv(PPI_DATA)
df_ppi

# %%
#df_ppi_filtered = df_ppi[df_ppi["property_type"] == "All Residential"].copy()

# Filter by date range
df_ppi_filtered = df_ppi[
    (df_ppi["quarter"] >= "2017-Q1") & 
    (df_ppi["quarter"] <= "2025-Q3") 
].copy()

df_ppi_filtered


# %%
from utils.convert_quarters import convert_quarter_to_date

df_ppi_filtered["month_start"] = df_ppi_filtered["quarter"].apply(convert_quarter_to_date)
df_ppi_filtered

# %%
df_ppi_upsampled = df_ppi_filtered.set_index("month_start")
df_ppi_upsampled

# %%
df_ppi_pivot = df_ppi_upsampled.pivot_table(
    index="month_start",
    columns="property_type",
    values="index",
)

df_ppi_pivot.rename(columns={
    "All Residential": "all_residential",
    "Landed": "landed",
    "Non-Landed": "non_landed",
}, inplace=True)

df_ppi_pivot


# %%
# date_range = pd.date_range(start=df_ppi_pivot.index.min(), end=END_DATE, freq="MS")

# # Re-index the DataFrame to the complete range
# df_ppi_monthly = df_ppi_pivot.resample("MS").last().reindex(date_range) 
# df_ppi_monthly.ffill(inplace=True) 
# df_ppi_monthly.reset_index(names=["month"], inplace=True)
# df_ppi_monthly.rename(columns={'index': 'private_property_index'}, inplace=True)

# df_ppi_monthly

# %%
date_range = pd.date_range(start=df_ppi_pivot.index.min(), end=END_DATE, freq="MS")

# Re-index the DataFrame to the complete range
df_ppi_monthly = df_ppi_pivot.resample("MS").last().reindex(date_range) 
df_ppi_monthly.ffill(inplace=True) 

df_ppi_monthly

# %%
# for merging
df_ppi_monthly.reset_index(names=["month"], inplace=True)

# merge the df
df_merged3 = pd.merge(
    df_merged2,
    df_ppi_monthly,
    on="month",
    how="left"
)

df_merged3

# %%
unemployment_rate_sa_DATA = DATA_DIR / "Unemployment_Rate_SA.csv"

df_ue = pd.read_csv(unemployment_rate_sa_DATA)
df_ue

# %%
df_ue_long = df_ue.melt(
    id_vars=["DataSeries"],
    var_name="month_str",
    value_name="unemployment_rate"
)

df_ue_long

# %%
df_ue_all = df_ue_long[df_ue_long["DataSeries"] == "Total Unemployment Rate, (SA)"].copy()
df_ue_all["unemployment_rate"] = pd.to_numeric(df_ue_all["unemployment_rate"])
df_ue_all

# %%
df_ue_all['quarter_clean'] = (
    df_ue_all['month_str']
    .str.replace(r'(\d{4})(\d)Q', r'\1-Q\2',regex=True)
)

df_ue_all["month_start"] = df_ue_all["quarter_clean"].apply(
    lambda x: convert_quarter_to_date(x) if pd.notna(x) else pd.NaT
)

df_ue_all

# %%
# Filter by date range
df_ue_filtered = df_ue_all[
    (df_ue_all["month_start"] >= "2017-Q1") & 
    (df_ue_all["month_start"] <= "2025-Q3") 
].copy()

df_ue_filtered.sort_values(by='month_start', ascending=True, inplace=True)
df_ue_filtered

# %%
df_ue_upsampled = df_ue_filtered.set_index("month_start")
date_range = pd.date_range(start=df_ue_upsampled.index.min(), end=END_DATE, freq="MS")
# Re-index the DataFrame to the complete range
df_ue_monthly = df_ue_upsampled.resample("MS").last().reindex(date_range) 
df_ue_monthly.ffill(inplace=True) 

df_ue_monthly

# %%
df_ue_monthly.reset_index(names=["month"], inplace=True)
df_ue_monthly.rename(columns={'unemployment_rate': 'unemployment_rate_sa'}, inplace=True)

df_ue_monthly

# %%
df_ue_to_merge = df_ue_monthly[['month', 'unemployment_rate_sa']]
df_ue_to_merge

# %%
# merge the df
df_merged4 = pd.merge(
    df_merged3,
    df_ue_to_merge,
    on="month",
    how="left"
)

df_merged4

# %%
HOUSEHOLD_INCOME = DATA_DIR / "Household_Income.csv"

df_hi = pd.read_csv(HOUSEHOLD_INCOME)
df_hi

# %%
columns_to_drop = [
    "CumulativeChangefrom2013to2023",
    "CumulativeChangefrom2013to2018",
    "CumulativeChangefrom2018to2023",
    "AnnualisedChangefrom2013to2023",
    "AnnualisedChangefrom2013to2018",
    "AnnualisedChangefrom2018to2023"
]

df_hi.drop(columns=columns_to_drop, inplace=True)
df_hi

# %%
df_hi_filtered = df_hi[df_hi["PerCent"] == "50th (Median)"].copy()
df_hi_long = df_hi_filtered.melt(
    id_vars="PerCent", 
    var_name="Year",
    value_name="50th (Median)"
)
df_hi_long.rename(columns={"50th (Median)": "income_change"}, inplace=True)
df_hi_long.drop(columns=["PerCent"], inplace=True)
df_hi_long

# %%
df_hi_long['Year'] = pd.to_datetime(df_hi_long['Year'], format='%Y')
df_hi_long['income_change'] = df_hi_long['income_change'].astype(float)

# %%
# Filter by date range
df_hi_filtered = df_hi_long[
    (df_hi_long["Year"] >= "2017-Q1") & 
    (df_hi_long["Year"] <= "2025-Q3") 
].copy()

df_hi_filtered

# %%
df_hi_upsampled = df_hi_filtered.set_index("Year")
date_range = pd.date_range(start=df_hi_upsampled.index.min(), end=END_DATE, freq="MS")
# Re-index the DataFrame to the complete range
df_hi_monthly = df_hi_upsampled.resample("MS").last().reindex(date_range) 
df_hi_monthly.ffill(inplace=True)

df_hi_monthly

# %%
#overwrite 2025 with annuliased change of 1.3
ANNUALISED_VALUE = 1.3
start_of_2025 = pd.to_datetime('2025-01-01')
df_hi_monthly.loc[df_hi_monthly.index >= start_of_2025, "income_change"] = ANNUALISED_VALUE

df_hi_monthly

# %%
df_hi_monthly.reset_index(names=["month"], inplace=True)
df_hi_monthly.rename(columns={'income_change': 'income_change'}, inplace=True)

df_hi_monthly

# %%
# merge the df
df_merged5 = pd.merge(
    df_merged4,
    df_hi_monthly,
    on="month",
    how="left"
)

df_merged5

# %%
df_merged5.to_parquet("../data/newfeats_data.parquet")

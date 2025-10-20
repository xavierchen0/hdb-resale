#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
from pathlib import Path
import pandas as pd

project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# In[ ]:


from src.config import DATA_DIR

df = pd.read_pickle(DATA_DIR / "cleaned_rawdata.pkl")
df


# In[ ]:


COE_DATA = DATA_DIR / "COE.csv"

df_coe = pd.read_csv(COE_DATA)
df_coe


# In[ ]:


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


# In[ ]:


# filter cat A and B
relevant_categories = ["Category A", "Category B"]
df_coe_filtered = df_coe1[df_coe1["vehicle_class"].isin(relevant_categories)].copy().reset_index()
df_coe_filtered


# In[ ]:


#take the mean of both bids
df_coe_pivot = df_coe_filtered.pivot_table(
    index="month",
    columns="vehicle_class",
    values="premium",
    aggfunc="mean"
).reset_index()

df_coe_pivot.rename(columns={
    "Category A": "coe_a",
    "Category B": "coe_b"
}, inplace=True)

# merge the df
df_merged1 = pd.merge(
    df,
    df_coe_pivot,
    on="month",
    how="left"
)

df_merged1


# In[ ]:


CPI_DATA = DATA_DIR / "CPI.csv"

df_cpi = pd.read_csv(CPI_DATA)
df_cpi.dtypes


# In[ ]:


df_cpi_long = df_cpi.melt(
    id_vars=["DataSeries"],
    var_name="month_str",
    value_name="cpi_index"
)

df_cpi_long


# In[ ]:


df_cpi_all = df_cpi_long[df_cpi_long["DataSeries"] == "All Items"].copy()


# convert the format from "YYYYMon" (e.g., "2025Aug") to "YYYY-MM-01".
df_cpi_all["month"] = pd.to_datetime(
    df_cpi_all["month_str"], 
    format="%Y%b"
)
df_cpi_all["cpi_index"] = pd.to_numeric(df_cpi_all["cpi_index"], errors="coerce")
df_cpi_all.dtypes


# In[ ]:


df_cpi_final = df_cpi_all[
    (df_cpi_all["month"] >= START_DATE) & 
    (df_cpi_all["month"] <= END_DATE)
][["month", "cpi_index"]].copy()

df_cpi_final


# In[ ]:


last_cpi_value = df_cpi_final["cpi_index"].iloc[0] 

new_row = pd.DataFrame({
    "month": [END_DATE], 
    "cpi_index": [last_cpi_value]
})

df_cpi_final = pd.concat([df_cpi_final, new_row], ignore_index=True)
df_cpi_final


# In[ ]:


# merge
df_merged2 = pd.merge(
    df_merged1,
    df_cpi_final,
    on="month",
    how="left"
)

df_merged2


# In[ ]:


PPI_DATA = DATA_DIR / "Private_Property_Index.csv"

df_ppi = pd.read_csv(PPI_DATA)
df_ppi


# In[ ]:


df_ppi_filtered = df_ppi[df_ppi["property_type"] == "All Residential"].copy()

# Filter by date range
df_ppi_filtered = df_ppi_filtered[
    (df_ppi_filtered["quarter"] >= "2017-Q1") & 
    (df_ppi_filtered["quarter"] <= "2025-Q3") 
].copy()

df_ppi_filtered


# In[ ]:


from utils.convert_quarters import convert_quarter_to_date

df_ppi_filtered["month_start"] = df_ppi_filtered["quarter"].apply(convert_quarter_to_date)
df_ppi_upsampled = df_ppi_filtered.set_index("month_start")
df_ppi_monthly = df_ppi_upsampled["index"].resample("M").ffill().resample("MS").last().to_frame()

df_ppi_monthly


# In[ ]:


date_range = pd.date_range(start=df_ppi_monthly.index.min(), end=END_DATE, freq="MS")

# Re-index the DataFrame to the complete range
df_ppi_monthly = df_ppi_monthly.resample("MS").last().reindex(date_range) 
df_ppi_monthly.ffill(inplace=True) 
df_ppi_monthly.reset_index(names=["month"], inplace=True)
df_ppi_monthly.rename(columns={'index': 'private_property_index'}, inplace=True)

df_ppi_monthly


# In[ ]:


df_merged3 = pd.merge(
    df_merged2, 
    df_ppi_monthly,
    on="month",
    how="left"
)

df_merged3


# In[ ]:


df_merged3.to_pickle("../data/feat_data.pkl")


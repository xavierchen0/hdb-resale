# %%
import pandas as pd;
import numpy as np;
import statsmodels.api as sm;
import statsmodels.stats.api as sms;
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from statsmodels.compat import lzip
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.diagnostic import acorr_breusch_godfrey as bg

# %%
project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    
from src.config import DATA_DIR
from utils.OLS_test import run_simple_ols_per_variable

df = pd.read_pickle(DATA_DIR / "cleaned_rawdata.pkl")
df

# %%
df["ln_remaining_lease"] = np.log(df["remaining_lease"])
df["ln_floor_area_sqm"] = np.log(df["floor_area_sqm"])
df["ln_resale_price"] = np.log(df["resale_price"])

dependent_var = ["resale_price","ln_resale_price"]
numerical_var = ["remaining_lease","ln_remaining_lease","floor_area_sqm","ln_floor_area_sqm"]
categorical_var = ["town", "flat_type", "storey_range", "flat_model", "region"]
independent_vars = numerical_var + categorical_var
print(independent_vars)

#%%
dummies = pd.get_dummies(df[categorical_var])
dummies = dummies.astype(int) 
data_for_ols = pd.concat([df[dependent_var + numerical_var], dummies], axis=1)
data_for_ols

#%%
columns_to_drop = [
    # Reference Categories to be dropped
    'town_ANG MO KIO', 
    'flat_type_1 ROOM', 
    'storey_range_01 TO 03', 
    'flat_model_Improved',
    'region_CENTRAL REGION',
]
all_independent_cols = numerical_var + list(dummies.columns)

run_simple_ols_per_variable(df, dependent_var, numerical_var, categorical_var)

#%%
# from the list of variables to use in the model.
final_independent_vars = [col for col in all_independent_cols if col not in columns_to_drop]
X = data_for_ols[final_independent_vars]
Y = data_for_ols[dependent_var]

all_town_cols = [col for col in data_for_ols.columns if col.startswith("town_")]
town_cols_to_use = [col for col in all_town_cols if col != "town_ANG MO KIO"]
X_town_dummies = data_for_ols[town_cols_to_use]

for yvar in dependent_var:
     y = data_for_ols[yvar]
     for num_var in numerical_var:
          X_numerical = data_for_ols[[num_var]]
          X_combined =  pd.concat([X_numerical, X_town_dummies], axis=1)
          try: 
               result = sm.OLS(y,sm.add_constant(X_combined),missing="drop").fit()
               print("="*80)
               print(f"MULTIPLE OLS MODEL: {yvar} vs. ({num_var} + Town Dummies)")
               print("="*80) 
               print(result.summary())
          except Exception as e:
               print(f"\nAn error occurred during OLS fitting for model with {num_var}: {e}")


#%%
X_dropped1 = X.drop(columns=["ln_remaining_lease","ln_floor_area_sqm"]).copy()
X_dropped2 = X.drop(columns=["remaining_lease","floor_area_sqm"]).copy()
Y_dropped1 = Y.drop(columns=["ln_resale_price"]).copy()
Y_dropped2 = Y.drop(columns=["resale_price"]).copy()

x_combinedx = [X_dropped1,X_dropped2]
for yvar in dependent_var:
    y = data_for_ols[yvar]
    for xvar in x_combinedx:
        try:
            result1 = sm.OLS(y,sm.add_constant(xvar),missing="drop").fit()
            print(result1.summary())
        except Exception as e:
            print(f"\nAn error occurred during OLS fitting for model with {num_var}: {e}")


#%%
#GOLDFIELD-QUANDT test
result1 = sm.OLS(Y_dropped1, sm.add_constant(X_dropped1.astype(float)), missing="drop").fit()
result1.summary()
GQ = sms.het_goldfeldquandt(result1.resid, result1.model.exog)
lzip(['Fstat', 'pval'], GQ)

#%%
# WHITE test
result1 = sm.OLS(Y_dropped2, sm.add_constant(X_dropped2.astype(float)), missing="drop").fit()
result1.summary()
wtest = het_white(result1.resid, result1.model.exog)
labels = ['Lagrange Multiplier statistic:', 'LM test\'s p-value:', 'F-statistic:', 'F-test\'s p-value:']
lzip(labels, wtest)


#%%
#Effect of additional logarithm filter on heterogeneity White's test results
df["lnln_resale_price"] = np.log(data_for_ols["ln_resale_price"]+1)
df["lnln_remaining_lease"] = np.log(df["ln_remaining_lease"]+1)
df["lnln_resale_price"] = np.log(df["ln_resale_price"]+1)

result = sm.OLS(df["lnln_resale_price"], sm.add_constant(df["lnln_remaining_lease"]), missing="drop").fit()
print(result.summary())
result = sm.OLS(df["lnln_resale_price"], sm.add_constant(df["lnln_resale_price"]), missing="drop").fit()
print(result.summary())

#%%
# BRESUCH-GODFREY
result3 = sm.OLS(Y_dropped2, sm.add_constant(X_dropped2.astype(float)), missing="drop").fit()
result3.summary()

print(bg(result3))
print(bg(result3, nlags=5))
import pandas as pd;
import statsmodels.api as sm;

def run_simple_ols_per_variable(data, dependent_var, numerical_vars, categorical_vars):

    # Run OLS for each variable separately
    for yvar in dependent_var:
        y = data[yvar]
        for var in numerical_vars + categorical_vars:
            
            print("\n" + "="*70)
            print(f"Simple OLS Regression: {yvar} vs {var}")
            print("="*70)
            
            if var in categorical_vars:
                try:
                    # convert categorical to 0/1 
                    X_temp = pd.get_dummies(data[var], prefix=var, drop_first=True).astype(int)
                except Exception as e:
                    print(f"Error creating dummies for {var}: {e}")
                    continue
                    
            elif var in numerical_vars:
                # use column directly for numerical 
                X_temp = data[[var]].copy()
                
            else:
                print(f"Skipping variable {var}: Type not recognized.")
                continue

            X_temp = sm.add_constant(X_temp)
            try:
                results = sm.OLS(y, sm.add_constant(X_temp), missing='drop').fit()
                print(results.summary())
                
            except Exception as e:
                print(f"An error occurred during OLS fitting for {var}: {e}")
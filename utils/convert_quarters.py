import pandas as pd

def convert_quarter_to_date(quarter_str):
    return pd.Period(quarter_str, freq="Q").start_time
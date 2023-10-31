import pandas as pd

# Use as utility
def get_airplane_params(df:pd.DataFrame) -> dict:
    airplane_params = {}
    for index, row in df.iterrows():
        airplane_params[row["var_name"]] = float(row["var_val"])

    airplane_params["mass"] = float(12) # kg
    
    return airplane_params


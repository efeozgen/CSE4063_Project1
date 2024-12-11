import pandas as pd

def detect_nulls(df):
    return pd.DataFrame({
        "nulls": df.isnull().sum(),
        "Empty Strings": df.applymap(lambda x: x == None).sum()
    })
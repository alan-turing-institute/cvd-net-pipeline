# utils/io.py
def load_csv(path):
    import pandas as pd
    return pd.read_csv(path)

def save_csv(df, path):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

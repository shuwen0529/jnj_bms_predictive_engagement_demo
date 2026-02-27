import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def build_preprocess(cat_cols, num_cols):
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])

def split_out_of_time(df: pd.DataFrame, time_col="month", train_max=9, test_min=10):
    train_df = df[df[time_col] <= train_max].copy()
    test_df = df[df[time_col] >= test_min].copy()
    return train_df, test_df

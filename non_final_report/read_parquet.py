# Bunch of stuff to check the parquet file looks OK


import pandas as pd

df = pd.read_parquet('cleaned_all.parquet')

print(df.head())
print(df.tail(5))
print(df.columns)

from cleaning.config import RAW_FILE, ID_COLS, DATE_COL, TARGET_COL
"""
df = pd.read_csv(RAW_FILE, nrows=1_000_000, low_memory=False)
cols = [c for c in df.columns if c not in set(ID_COLS + [DATE_COL, TARGET_COL])]
row_na = df[cols].isna().sum(axis=1)
print("rows with >50 NaNs:", (row_na > 50).mean())
print("rows with >100 NaNs:", (row_na > 100).mean())
print(row_na.describe())

"""
df_clean = pd.read_parquet('cleaned_all.parquet')

# Exclude IDs, date/target, and *_was_missing flags from NaN check
exclude = set(ID_COLS + [DATE_COL, TARGET_COL])
cols_num = [c for c in df_clean.columns if c not in exclude and not c.endswith('_was_missing')]
nan_per_col = df_clean[cols_num].isna().sum().sort_values(ascending=False)
print("Top columns by NaN in cleaned (should be near zero):")
print(nan_per_col.head(10))

row_na_clean = df_clean[cols_num].isna().sum(axis=1)
print("Cleaned: rows with any NaN:", (row_na_clean > 0).mean())
print(row_na_clean.describe())


df_raw = pd.read_csv(RAW_FILE, nrows=1_000_000, low_memory=False)
feat_cols = [c for c in df_raw.columns if c not in set(ID_COLS + [DATE_COL, TARGET_COL])]
row_missing_rate = df_raw[feat_cols].isna().mean(axis=1)
print("rows with >66% missing:", (row_missing_rate > 0.66).mean())
print(row_missing_rate.describe())
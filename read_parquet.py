import pandas as pd

df = pd.read_parquet('cleaned_all.parquet')

print(df.head())
print(df.tail(5))
print(df.columns)
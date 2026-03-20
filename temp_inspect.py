import pandas as pd
import os
PROJECT_ROOT=r"C:\Users\nicol\OneDrive\Bureau\Finance"
os.chdir(PROJECT_ROOT)
df=pd.read_csv('data/processed/features_pred.csv', index_col=0, parse_dates=True)
returns=df['log_return'].dropna()
print(type(returns.index))
print(returns.index[:5])

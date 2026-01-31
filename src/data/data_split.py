import pandas as pd

def walk_forward_split(df, train_size=0.7):
    split_idx = int(len(df) * train_size)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test

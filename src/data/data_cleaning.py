import pandas as pd
import numpy as np
import os
import yaml

def load_config():
    with open("src/config/config.yaml", "r") as file:
        return yaml.safe_load(file)

def clean_market_data(df):
    df = df.copy()

    # Conversion explicite de Date
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df.set_index("Date", inplace=True)

    # Tri temporel
    df.sort_index(inplace=True)

    # Rendements logarithmiques
    df["log_return"] = np.log(df["Close"]).diff()

    df.dropna(inplace=True)
    return df

def save_clean_data(df, path, filename):
    os.makedirs(path, exist_ok=True)
    df.to_csv(os.path.join(path, filename))

if __name__ == "__main__":
    config = load_config()

    df_raw = pd.read_csv(
        os.path.join(config["paths"]["raw_data"], "prices.csv")
    )

    print("Preview before cleaning:")
    print(df_raw.head())

    df_clean = clean_market_data(df_raw)

    print("Preview after cleaning:")
    print(df_clean.head())

    save_clean_data(
        df_clean,
        config["paths"]["interim_data"],
        "market_clean.csv"
    )

    print("Clean market data saved successfully.")

import yfinance as yf
import pandas as pd
import os
import yaml

def load_config():
    with open("src/config/config2.yaml", "r") as file:
        return yaml.safe_load(file)

def download_market_data(ticker, start, end):
    tk = yf.Ticker(ticker)

    df = tk.history(
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False
    )

    if df.empty:
        raise ValueError("No data returned from Yahoo Finance")

    df.reset_index(inplace=True)
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    return df

def save_raw_data(df, path, filename):
    os.makedirs(path, exist_ok=True)
    df.to_csv(os.path.join(path, filename), index=False)

if __name__ == "__main__":
    config = load_config()

    df = download_market_data(
        ticker=config["data"]["ticker"],
        start=config["data"]["start_date"],
        end=config["data"]["end_date"]
    )

    print(df.head())
    print("Shape:", df.shape)

    save_raw_data(df, config["paths"]["raw_data"], "prices_pred.csv")
    print("Raw market data saved correctly.")

import yfinance as yf

df = yf.download(
    "^GSPC",
    start="2020-01-01",
    end="2020-12-31",
    progress=False
)

print(df)
print("Shape:", df.shape)

import yfinance as yf

ticker = "AMZN"
df = yf.download(ticker, period="5y", interval="1d")

print(df.head())
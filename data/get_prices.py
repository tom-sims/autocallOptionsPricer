import yfinance as yf
prices = yf.download("^GSPC", start="2020-01-01")
prices[["Close"]].to_csv("data/spx_prices.csv")


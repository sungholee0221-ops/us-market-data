"""Why is the latest session missing? Compare yfinance call styles."""
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

print("yfinance", yf.__version__, "pandas", pd.__version__)
T = ["SPY", "AAPL", "UNH", "^GSPC", "^VIX"]
tomorrow = (date.today() + timedelta(days=1)).isoformat()

def tail(label, df):
    if df is None or df.empty:
        print(f"{label}: EMPTY"); return
    print(f"--- {label}: index tail")
    print(df.tail(3).iloc[:, :6] if df.shape[1] > 6 else df.tail(3))

tail("download period=2y group_by=ticker", yf.download(T, period="2y", interval="1d", auto_adjust=True, group_by="ticker", threads=True, progress=False))
tail("download period=5d", yf.download(T, period="5d", interval="1d", auto_adjust=True, progress=False)["Close"])
tail("download start/end explicit", yf.download(T, start="2026-11-01", end=tomorrow, interval="1d", auto_adjust=True, progress=False)["Close"])
tail("Ticker(SPY).history 5d", yf.Ticker("SPY").history(period="5d", auto_adjust=True))
tail("download 1h period=2d", yf.download(["SPY"], period="2d", interval="1h", auto_adjust=True, progress=False)["Close"])
fi = yf.Ticker("SPY").fast_info
print("fast_info last_price", fi.get("lastPrice"), "previous_close", fi.get("previousClose"))

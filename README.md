# us-market-data

Daily adjusted OHLCV for the S&P 500, Nasdaq-100, and a set of index/sector ETFs, fetched with yfinance by GitHub Actions after each US session.

- Code: `main` branch
- Data: `data` branch, a single commit replaced daily
  - `prices.parquet` — `date, ticker, open, high, low, close, volume` (split/dividend adjusted, ~2 years)
  - `universe.csv` — tickers and source list
  - `meta.json` — last trading date, ticker count, failed tickers

```bash
git clone --depth 1 -b data https://github.com/<owner>/us-market-data
```

Data comes from Yahoo Finance via yfinance and is provided as-is for personal research.

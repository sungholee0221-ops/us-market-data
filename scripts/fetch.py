"""Daily US equity bars: S&P 500 + Nasdaq-100 + extra ETFs.

Output (written to OUT_DIR, default ./out):
  prices.parquet  long format: date, ticker, open, high, low, close, volume
                  split/dividend adjusted (yfinance auto_adjust=True)
  universe.csv    ticker, name, source
  meta.json       generated_at, last_date, tickers, rows, failed
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get("OUT_DIR", ROOT / "out"))
PERIOD = os.environ.get("PERIOD", "2y")
CHUNK = 100
UA = {"User-Agent": "Mozilla/5.0 (us-market-data fetcher)"}

SP500_CSV = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
NDX_WIKI = "https://en.wikipedia.org/wiki/Nasdaq-100"
NDX_API = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
PREV_UNIVERSE = "https://raw.githubusercontent.com/sungholee0221-ops/us-market-data/data/universe.csv"
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")
WIKI_UA = "us-market-data/1.0 (https://github.com/sungholee0221-ops/us-market-data)"


def yahoo_symbol(t: str) -> str:
    return t.strip().upper().replace(".", "-")


def _ndx_nasdaq_api() -> pd.DataFrame:
    r = requests.get(
        NDX_API, timeout=30,
        headers={"User-Agent": BROWSER_UA, "Accept": "application/json, text/plain, */*",
                 "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"},
    )
    r.raise_for_status()
    rows = r.json()["data"]["data"]["rows"]
    return pd.DataFrame({"ticker": [x["symbol"] for x in rows],
                         "name": [x.get("companyName", "") for x in rows]})


def _ndx_wikipedia() -> pd.DataFrame:
    r = requests.get(NDX_WIKI, timeout=30, headers={"User-Agent": WIKI_UA})
    r.raise_for_status()
    for t in pd.read_html(io.StringIO(r.text)):
        if isinstance(t.columns, pd.MultiIndex):
            t.columns = t.columns.get_level_values(-1)
        cols = {str(c).strip().lower(): c for c in t.columns}
        tk = next((cols[k] for k in cols if k.startswith(("ticker", "symbol"))), None)
        nm = next((cols[k] for k in cols if k.startswith(("company", "security"))), None)
        if tk is not None and nm is not None and len(t) >= 90:
            return pd.DataFrame({"ticker": t[tk], "name": t[nm]})
    raise ValueError("no constituents table")


def _ndx_previous() -> pd.DataFrame:
    u = pd.read_csv(io.StringIO(requests.get(PREV_UNIVERSE, timeout=30).text))
    u = u[u["source"] == "ndx100"]
    if u.empty:
        raise ValueError("no ndx100 rows in previous universe")
    return u[["ticker", "name"]]


def load_ndx100() -> pd.DataFrame | None:
    """Nasdaq API -> Wikipedia -> last published list. Additive: None if all fail."""
    for label, fn in (("nasdaq-api", _ndx_nasdaq_api), ("wikipedia", _ndx_wikipedia),
                      ("previous-data-branch", _ndx_previous)):
        try:
            df = fn()
            if len(df) >= 90:
                print(f"ndx100: {len(df)} tickers from {label}")
                df["source"] = "ndx100"
                return df
            print(f"WARN ndx100 {label}: only {len(df)} rows", file=sys.stderr)
        except Exception as e:
            print(f"WARN ndx100 {label} failed: {e}", file=sys.stderr)
    return None


def load_universe() -> pd.DataFrame:
    frames = []

    sp = pd.read_csv(io.StringIO(requests.get(SP500_CSV, headers=UA, timeout=30).text))
    frames.append(pd.DataFrame({"ticker": sp["Symbol"], "name": sp["Security"], "source": "sp500"}))

    ndx = load_ndx100()
    if ndx is not None:
        frames.append(ndx)

    extra = pd.read_csv(ROOT / "universe" / "extra.csv")
    extra["source"] = "extra"
    frames.append(extra[["ticker", "name", "source"]])

    u = pd.concat(frames, ignore_index=True)
    u["ticker"] = u["ticker"].map(yahoo_symbol)
    u = u.drop_duplicates("ticker", keep="first").sort_values("ticker").reset_index(drop=True)
    return u


def download(tickers: list[str]) -> pd.DataFrame:
    df = yf.download(
        tickers, period=PERIOD, interval="1d", auto_adjust=True,
        group_by="ticker", threads=True, progress=False,
    )
    if df.empty:
        return pd.DataFrame()
    if not isinstance(df.columns, pd.MultiIndex):  # single ticker
        df.columns = pd.MultiIndex.from_product([tickers, df.columns])
    long = df.stack(level=0, future_stack=True).reset_index()
    long.columns = [str(c).lower() for c in long.columns]
    long = long.rename(columns={"level_1": "ticker", "price": "ticker"})
    if "ticker" not in long.columns:
        long = long.rename(columns={long.columns[1]: "ticker"})
    long = long.dropna(subset=["close"])
    return long[["date", "ticker", "open", "high", "low", "close", "volume"]]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    u = load_universe()
    tickers = u["ticker"].tolist()
    print(f"universe: {len(tickers)} tickers")

    parts, missing = [], []
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i : i + CHUNK]
        got = download(chunk)
        parts.append(got)
        missing += sorted(set(chunk) - set(got["ticker"]) if not got.empty else chunk)
        time.sleep(2)

    if missing:  # one retry, smaller batches
        print(f"retry {len(missing)}: {missing[:20]}")
        time.sleep(10)
        still = []
        for i in range(0, len(missing), 20):
            chunk = missing[i : i + 20]
            got = download(chunk)
            parts.append(got)
            still += sorted(set(chunk) - set(got["ticker"]) if not got.empty else chunk)
            time.sleep(3)
        missing = still

    px = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    px["date"] = pd.to_datetime(px["date"]).dt.date
    px["volume"] = px["volume"].fillna(0).astype("int64")
    for c in ["open", "high", "low", "close"]:
        px[c] = px[c].astype("float64").round(4)
    px = px.sort_values(["ticker", "date"]).reset_index(drop=True)

    got_n = px["ticker"].nunique()
    if got_n < 0.9 * len(tickers):
        print(f"ERROR only {got_n}/{len(tickers)} tickers downloaded; not publishing", file=sys.stderr)
        return 1

    px.to_parquet(OUT_DIR / "prices.parquet", index=False, compression="zstd")
    u.to_csv(OUT_DIR / "universe.csv", index=False)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_date": str(px["date"].max()),
        "period": PERIOD,
        "adjusted": True,
        "tickers": int(got_n),
        "rows": int(len(px)),
        "failed": missing,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
NDX_GH = "https://raw.githubusercontent.com/Gary-Strauss/NASDAQ100_Constituents/master/data/nasdaq100_constituents.csv"
PREV_UNIVERSE = "https://raw.githubusercontent.com/sungholee0221-ops/us-market-data/data/universe.csv"
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")
WIKI_UA = "us-market-data/1.0 (https://github.com/sungholee0221-ops/us-market-data)"


def yahoo_symbol(t: str) -> str:
    return t.strip().upper().replace(".", "-")


def _ndx_nasdaq_api() -> pd.DataFrame:
    r = requests.get(
        NDX_API, timeout=10,
        headers={"User-Agent": BROWSER_UA, "Accept": "application/json, text/plain, */*",
                 "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"},
    )
    r.raise_for_status()
    rows = r.json()["data"]["data"]["rows"]
    return pd.DataFrame({"ticker": [x["symbol"] for x in rows],
                         "name": [x.get("companyName", "") for x in rows]})


def _ndx_github() -> pd.DataFrame:
    n = pd.read_csv(io.StringIO(requests.get(NDX_GH, timeout=30).text))
    return pd.DataFrame({"ticker": n["Ticker"], "name": n["Company"]})


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
    for label, fn in (("nasdaq-api", _ndx_nasdaq_api), ("github-list", _ndx_github), ("wikipedia", _ndx_wikipedia),
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


INDICES = ["^GSPC", "^IXIC", "^DJI", "^SOX", "^VIX"]
FNG_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{start}"
RAW_DATA = "https://raw.githubusercontent.com/sungholee0221-ops/us-market-data/data/{name}"


def _intraday(tickers: list[str], interval: str, period: str) -> pd.DataFrame:
    """Long frame: ts (America/New_York), ticker, open, high, low, close, volume."""
    df = yf.download(tickers, period=period, interval=interval, auto_adjust=True,
                     group_by="ticker", threads=True, progress=False, prepost=False)
    if df.empty:
        return pd.DataFrame()
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_product([tickers, df.columns])
    long = df.stack(level=0, future_stack=True).reset_index()
    long.columns = [str(c).lower() for c in long.columns]
    long = long.rename(columns={long.columns[0]: "ts", long.columns[1]: "ticker"}).dropna(subset=["close"])
    long["ts"] = pd.to_datetime(long["ts"])
    long["ts"] = (long["ts"].dt.tz_convert("America/New_York") if long["ts"].dt.tz is not None
                  else long["ts"].dt.tz_localize("UTC").dt.tz_convert("America/New_York"))
    return long[["ts", "ticker", "open", "high", "low", "close", "volume"]]


def fill_from_intraday(px: pd.DataFrame) -> pd.DataFrame:
    """Yahoo sometimes serves the latest daily bar of stocks/ETFs with null OHLC for hours
    after the close, while indices and hourly bars are complete. Rebuild those bars from 1h
    data; the next run replaces them with the official daily bar."""
    sessions = sorted(px.loc[px["ticker"] == "^GSPC", "date"].unique())[-5:]
    have = set(zip(px["ticker"], px["date"]))
    tickers = [t for t in px["ticker"].unique() if not t.startswith("^")]
    need = sorted({t for t in tickers for d in sessions if (t, d) not in have})
    if not need:
        return px
    print(f"intraday fill: {len(need)} tickers missing a recent session")
    parts = []
    for i in range(0, len(need), CHUNK):
        got = _intraday(need[i : i + CHUNK], "1h", "5d")
        if not got.empty:
            parts.append(got)
        time.sleep(2)
    if not parts:
        return px
    h = pd.concat(parts, ignore_index=True)
    h["date"] = h["ts"].dt.date
    bars = (h.sort_values("ts").groupby(["ticker", "date"])
            .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                 close=("close", "last"), volume=("volume", "sum")).reset_index())
    bars = bars[bars["date"].isin(sessions)]
    bars = bars[[(t, d) not in have for t, d in zip(bars["ticker"], bars["date"])]]
    print(f"intraday fill: added {len(bars)} bars for dates {sorted({str(d) for d in bars['date']})}")
    return pd.concat([px, bars[px.columns]], ignore_index=True)


def fetch_index_intraday(out: Path) -> None:
    try:
        d = _intraday(INDICES, "5m", "5d")
        d["date"] = d["ts"].dt.date
        last = d.groupby("ticker")["date"].transform("max")
        d = d[d["date"] == last]
        d["ts"] = d["ts"].dt.strftime("%Y-%m-%dT%H:%M")
        d[["ticker", "ts", "close"]].round(4).to_csv(out / "intraday.csv", index=False)
        print(f"intraday indices: {len(d)} rows")
    except Exception as e:
        print(f"WARN intraday indices failed: {e}", file=sys.stderr)
        _copy_previous("intraday.csv", out)


def _copy_previous(name: str, out: Path) -> None:
    try:
        r = requests.get(RAW_DATA.format(name=name), timeout=30)
        r.raise_for_status()
        (out / name).write_bytes(r.content)
        print(f"kept previous {name}")
    except Exception as e:
        print(f"WARN no previous {name}: {e}", file=sys.stderr)


def fetch_marketcaps(tickers: list[str], out: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    def one(t):
        try:
            return t, float(yf.Ticker(t).fast_info["marketCap"] or 0)
        except Exception:
            return t, None

    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(one, tickers))
    mc = pd.DataFrame(rows, columns=["ticker", "market_cap"]).dropna()
    mc = mc[mc["market_cap"] > 0]
    print(f"market caps: {len(mc)}/{len(tickers)}")
    if len(mc) < 0.8 * len(tickers):
        _copy_previous("marketcap.csv", out)
        return
    mc.to_csv(out / "marketcap.csv", index=False)


def fetch_fear_greed(out: Path) -> None:
    start = (datetime.now(timezone.utc).date() - pd.Timedelta(days=400)).isoformat()
    try:
        r = requests.get(FNG_URL.format(start=start), timeout=30, headers={
            "User-Agent": BROWSER_UA, "Accept": "application/json",
            "Referer": "https://edition.cnn.com/", "Origin": "https://edition.cnn.com"})
        r.raise_for_status()
        j = r.json()
        fg = j["fear_and_greed"]
        summary = {k: fg.get(k) for k in ["score", "rating", "timestamp", "previous_close",
                                         "previous_1_week", "previous_1_month", "previous_1_year"]}
        (out / "fng.json").write_text(json.dumps(summary, indent=2))
        hist = pd.DataFrame(j["fear_and_greed_historical"]["data"])
        hist["date"] = pd.to_datetime(hist["x"], unit="ms").dt.date
        hist = hist.rename(columns={"y": "score"}).groupby("date").last().reset_index()
        hist[["date", "score", "rating"]].round(2).to_csv(out / "fng_history.csv", index=False)
        print(f"fear & greed: {summary['score']:.1f} {summary['rating']}, {len(hist)} days")
    except Exception as e:
        print(f"WARN fear & greed failed: {e}", file=sys.stderr)
        _copy_previous("fng.json", out)
        _copy_previous("fng_history.csv", out)


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
    px = fill_from_intraday(px)
    px["volume"] = px["volume"].fillna(0).astype("int64")
    for c in ["open", "high", "low", "close"]:
        px[c] = px[c].astype("float64").round(4)
    # drop partial sessions (e.g. a run before the open returns a bar for a handful of tickers)
    per_day = px.groupby("date")["ticker"].nunique()
    print("tickers per day (last 4):", {str(k): int(v) for k, v in per_day.tail(4).items()})
    partial = per_day[per_day < 0.5 * per_day.max()].index
    if len(partial):
        print(f"dropping partial dates: {[str(d) for d in partial]}")
        px = px[~px["date"].isin(partial)]
    px = px.sort_values(["ticker", "date"]).reset_index(drop=True)

    got_n = px["ticker"].nunique()
    if got_n < 0.9 * len(tickers):
        print(f"ERROR only {got_n}/{len(tickers)} tickers downloaded; not publishing", file=sys.stderr)
        return 1

    px.to_parquet(OUT_DIR / "prices.parquet", index=False, compression="zstd")
    u.to_csv(OUT_DIR / "universe.csv", index=False)
    fetch_index_intraday(OUT_DIR)
    fetch_marketcaps(u.loc[u["source"] != "extra", "ticker"].tolist(), OUT_DIR)
    fetch_fear_greed(OUT_DIR)
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

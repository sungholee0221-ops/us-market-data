"""Trend Template + RS percentile screen over the data branch.

usage: python screen.py PRICES_PARQUET [TICKER ...]
  no tickers -> full universe (stocks only), sorted by RS
Prints CSV to stdout.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

ETF_LIKE = {
    "SPY", "QQQ", "IWM", "DIA", "RSP", "XLK", "XLV", "XLF", "XLI", "XLE", "XLY", "XLP",
    "XLU", "XLB", "XLRE", "XLC", "SMH", "IBB", "XBI", "GLD", "SGOV", "TLT", "^VIX",
}


def rs_score(c: pd.Series) -> float:
    """IBD-style weighted 12m performance: 40% last 3m, 20% each prior quarter."""
    if len(c) < 253:
        return np.nan
    r = lambda n: c.iloc[-1] / c.iloc[-1 - n] - 1  # noqa: E731
    return 0.4 * r(63) + 0.2 * r(126) + 0.2 * r(189) + 0.2 * r(252)


def per_ticker(g: pd.DataFrame) -> dict:
    g = g.sort_values("date")
    c, h, l = g["close"], g["high"], g["low"]
    out = {"ticker": g["ticker"].iat[0], "date": g["date"].iat[-1], "bars": len(g), "close": c.iat[-1]}
    if len(g) < 252:
        out["verdict"] = "insufficient"
        return out
    s50, s150, s200 = (c.rolling(n).mean() for n in (50, 150, 200))
    hi52, lo52 = h.iloc[-252:].max(), l.iloc[-252:].min()
    px = c.iat[-1]
    out.update(
        sma50=s50.iat[-1], sma150=s150.iat[-1], sma200=s200.iat[-1],
        hi52=hi52, lo52=lo52,
        pct_from_hi=px / hi52 - 1, pct_above_lo=px / lo52 - 1,
        pct_vs_sma50=px / s50.iat[-1] - 1,
        ema21=c.ewm(span=21, adjust=False).mean().iat[-1],
        ret_1m=px / c.iat[-22] - 1, ret_3m=px / c.iat[-64] - 1,
        sma200_slope_21d=s200.iat[-1] / s200.iat[-22] - 1,
        rs_raw=rs_score(c),
    )
    out.update(
        c1=px > s150.iat[-1] and px > s200.iat[-1],
        c2=s150.iat[-1] > s200.iat[-1],
        c3=s200.iat[-1] > s200.iat[-22],
        c4=s50.iat[-1] > s150.iat[-1] and s50.iat[-1] > s200.iat[-1],
        c5=px > s50.iat[-1],
        c6=px >= 1.30 * lo52,
        c7=px >= 0.75 * hi52,
    )
    return out


def screen(prices: pd.DataFrame, tickers: list[str] | None = None) -> pd.DataFrame:
    rows = [per_ticker(g) for _, g in prices.groupby("ticker", sort=False)]
    df = pd.DataFrame(rows)
    stocks = ~df["ticker"].isin(ETF_LIKE) & df["rs_raw"].notna()
    df.loc[stocks, "rs_pct"] = df.loc[stocks, "rs_raw"].rank(pct=True).mul(99).round()
    df["c8"] = df["rs_pct"] >= 70
    conds = [f"c{i}" for i in range(1, 9)]
    ok = df[conds].fillna(False).astype(bool)
    df["passed"] = ok.sum(axis=1)
    df["failed_conds"] = ok.apply(lambda r: ",".join(c[1:] for c in conds if not r[c]), axis=1)
    df["verdict"] = np.select(
        [df["verdict"].eq("insufficient") if "verdict" in df else False, df["passed"].eq(8), df["passed"].eq(7)],
        ["insufficient", "PASS", "NEAR"], "FAIL",
    )
    if tickers:
        want = [t.upper().replace(".", "-") for t in tickers]
        df = df.set_index("ticker").reindex(want).reset_index()
    else:
        df = df[~df["ticker"].isin(ETF_LIKE)].sort_values("rs_pct", ascending=False)
    return df


if __name__ == "__main__":
    px = pd.read_parquet(sys.argv[1])
    res = screen(px, sys.argv[2:] or None)
    pd.set_option("display.width", 250)
    res.round(4).to_csv(sys.stdout, index=False)

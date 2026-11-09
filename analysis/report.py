"""Build dashboard CSVs from prices + a private plan file.

usage: python report.py PRICES_PARQUET PLAN_JSON SP500_CSV EVENTS_CSV OUT_DIR

PLAN_JSON (kept outside this public repo):
{
  "theme": "...", "budget": 6000, "as_of": "2026-11-09",
  "positions": [{"ticker": "TMO", "sector": "Health Care", "amount": 600,
                 "entry": "now|pullback", "status": "planned|bought|skipped",
                 "note": "..."}],
  "tranches": [{"id": "T1", "date": "2026-11-11", "pct": 40,
                "condition": "...", "status": "pending|done|skipped"}]
}
Writes: picks.csv, sectors.csv, etfs.csv, leaders.csv, tranches.csv,
        events.csv, run.csv
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import screen  # noqa: E402

ETFS = {
    "SPY": "S&P 500", "RSP": "S&P 500 Equal Wt", "QQQ": "Nasdaq-100", "IWM": "Russell 2000",
    "XLV": "Health Care", "XLF": "Financials", "XLI": "Industrials", "XLE": "Energy",
    "XLK": "Technology", "SMH": "Semiconductors", "XLP": "Staples", "XLU": "Utilities",
    "XLY": "Discretionary", "XLC": "Comm Services", "XLB": "Materials", "XLRE": "Real Estate",
    "IBB": "Biotech", "GLD": "Gold", "TLT": "20Y+ Treasury",
}
EXTENDED = 0.08      # > 8% above 50d SMA -> wait for 21EMA pullback
STOP_PCT = 0.08      # Minervini max loss


def etf_table(px: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, g in px[px["ticker"].isin(ETFS)].groupby("ticker"):
        c = g.sort_values("date")["close"].reset_index(drop=True)
        h = g.sort_values("date")["high"].reset_index(drop=True)
        s50, s200 = c.rolling(50).mean().iat[-1], c.rolling(200).mean().iat[-1]
        rows.append({
            "etf": t, "label": ETFS[t], "close": c.iat[-1],
            "ret_1w": c.iat[-1] / c.iat[-6] - 1, "ret_1m": c.iat[-1] / c.iat[-22] - 1,
            "ret_3m": c.iat[-1] / c.iat[-64] - 1,
            "vs_sma50": c.iat[-1] / s50 - 1, "vs_sma200": c.iat[-1] / s200 - 1,
            "from_high": c.iat[-1] / h.iloc[-252:].max() - 1,
        })
    return pd.DataFrame(rows).sort_values("ret_1m", ascending=False)


def main(prices, plan_path, sp_path, events_path, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    px = pd.read_parquet(prices)
    plan = json.loads(Path(plan_path).read_text())
    sp = pd.read_csv(sp_path)
    sp["ticker"] = sp["Symbol"].str.replace(".", "-", regex=False)
    meta = sp[["ticker", "Security", "GICS Sector", "GICS Sub-Industry"]].rename(
        columns={"Security": "name", "GICS Sector": "sector", "GICS Sub-Industry": "industry"})

    full = screen.screen(px).merge(meta, on="ticker", how="left")
    data_date = str(px["date"].max())

    # picks
    pos = pd.DataFrame(plan["positions"])
    pos["ticker"] = pos["ticker"].str.upper().str.replace(".", "-", regex=False)
    picks = pos.merge(full.drop(columns=["sector"]), on="ticker", how="left")
    picks["entry_zone"] = [
        "pullback to 21EMA" if (v or 0) > EXTENDED else "buyable"
        for v in picks["pct_vs_sma50"]
    ]
    picks["stop_8pct"] = picks["close"] * (1 - STOP_PCT)
    cols = ["ticker", "name", "sector", "amount", "status", "verdict", "failed_conds", "rs_pct",
            "close", "pct_from_hi", "pct_vs_sma50", "ema21", "sma50", "stop_8pct", "entry_zone", "note"]
    picks[[c for c in cols if c in picks]].round(4).to_csv(out / "picks.csv", index=False)

    # sectors
    sec = (full.dropna(subset=["sector"]).groupby("sector")
           .agg(stocks=("ticker", "size"), passing=("verdict", lambda s: (s == "PASS").sum()))
           .reset_index())
    sec["pass_rate"] = (sec["passing"] / sec["stocks"]).round(3)
    sec.sort_values("pass_rate", ascending=False).to_csv(out / "sectors.csv", index=False)

    etf_table(px).round(4).to_csv(out / "etfs.csv", index=False)

    lead = (full[full["verdict"] == "PASS"].sort_values("rs_pct", ascending=False)
            .groupby("sector").head(8))
    lead[["ticker", "name", "sector", "industry", "rs_pct", "close", "pct_from_hi", "pct_vs_sma50",
          "ret_1m"]].round(4).to_csv(out / "leaders.csv", index=False)

    pd.DataFrame(plan["tranches"]).to_csv(out / "tranches.csv", index=False)

    ev = pd.read_csv(events_path)
    ev = ev[pd.to_datetime(ev["date"]).dt.date >= date.fromisoformat(plan.get("as_of", data_date))]
    ev.to_csv(out / "events.csv", index=False)

    spy = full[full["ticker"].isin(["SPY"])]
    pd.DataFrame([{
        "theme": plan.get("theme", ""), "budget": plan.get("budget"),
        "data_date": data_date, "plan_as_of": plan.get("as_of"),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "universe_pass": int((full["verdict"] == "PASS").sum()), "universe_n": int(len(full)),
        "planned_total": int(pos["amount"].sum()),
        "bought_total": int(pos.loc[pos["status"] == "bought", "amount"].sum()),
    }]).to_csv(out / "run.csv", index=False)
    print(f"report written to {out} (data {data_date})")


if __name__ == "__main__":
    main(*sys.argv[1:6])

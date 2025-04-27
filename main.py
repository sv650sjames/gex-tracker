#!/usr/bin/env python
# main.py ─ yfinance-based gamma tracker with on-the-fly Black-Scholes gamma

import argparse, datetime as dt, json, math
from pathlib import Path
from datetime import datetime, timezone  # Add this import

import numpy as np
import pandas as pd
import yfinance as yf

CONTRACT_SIZE = 100              # OCC equity option contract size
RISK_FREE     = 0.02             # default risk-free rate (2 %)

# ────────────────────────── Black-Scholes helpers ─────────────────────────
def _norm_pdf(x):          # standard-normal pdf
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def bs_gamma(spot, strike, t, iv, r=RISK_FREE):
    """Return Black-Scholes gamma for either put or call."""
    if t <= 0 or iv <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
    return _norm_pdf(d1) / (spot * iv * math.sqrt(t))

# ───────────────────────────── Helpers ────────────────────────────────────
def log_to_csv(ticker: str, total_bn: float):
    csv_path = Path("data/gex_history.csv")
    csv_path.parent.mkdir(exist_ok=True)
    header = not csv_path.exists()
    with csv_path.open("a") as f:
        if header:
            f.write("date,ticker,gex_bn\n")
        f.write(f"{dt.date.today()},{ticker},{total_bn}\n")

def log_top_strikes(ticker: str, gex_top: pd.Series):
    """
    Append 5 strike / gamma pairs to data/top_gamma.csv
    Columns: date, ticker, strike, gex_bn
    """
    csv = Path("data/top_gamma.csv")
    csv.parent.mkdir(exist_ok=True)
    header = not csv.exists()
    today  = dt.date.today()

    with csv.open("a") as f:
        if header:
            f.write("date,ticker,strike,gex_bn\n")
        for strike, gex in gex_top.items():
            f.write(f"{today},{ticker},{strike},{round(gex,4)}\n")

 

def top_gamma_strikes(df: pd.DataFrame, spot: float, n=5):
    gex = (spot**2 * df.gamma * df.open_interest * CONTRACT_SIZE / 1e9)
    gex[df.type == "P"] *= -1
    by_strike = gex.groupby(df.strike).sum()
    order = by_strike.abs().sort_values(ascending=False).index
    return by_strike.loc[order].head(n)

# ───────────────────── Fetch & gamma-compute via yfinance ─────────────────
def fetch_chain_and_spot(ticker: str, r_rate: float):
    yf_sym = ticker if ticker.startswith("^") else ticker
    tk     = yf.Ticker(yf_sym)

    spot = float(tk.info["regularMarketPrice"])
    expiry_str = tk.options[0]                         # ← "YYYY-MM-DD"
    exp_date   = dt.datetime.strptime(expiry_str, "%Y-%m-%d").date()
    t_years    = (exp_date - datetime.now(timezone.utc).date()).days / 365.0

    chain = tk.option_chain(expiry_str)                # explicit expiry

    def _prep(df, cp_flag):
        df = df.copy()
        df["type"] = cp_flag
        iv = df.impliedVolatility.fillna(0).astype(float)
        gamma = [bs_gamma(spot, k, t_years, v, r_rate)
                 for k, v in zip(df.strike, iv)]
        df["gamma"] = gamma
        df.rename(columns={"openInterest": "open_interest"}, inplace=True)
        return df[["strike", "type", "gamma", "open_interest"]]

    calls = _prep(chain.calls, "C")
    puts  = _prep(chain.puts,  "P")
    out   = pd.concat([calls, puts], ignore_index=True)
    return spot, out, chain


# ────────────────────────── Business logic ───────────────────────────────
def compute_total_gex(spot, df, ticker):
    gex = spot**2 * df.gamma * df.open_interest * CONTRACT_SIZE / 1e9
    gex[df.type == "P"] *= -1
    total_bn = round(gex.sum(), 4)

    print(f"\nTotal notional GEX: ${total_bn} Bn")
    log_to_csv(ticker, total_bn)

    gex_top = top_gamma_strikes(df, spot)
    print("\nTop gamma strikes ($ Bn):")
    #print(top_gamma_strikes(df, spot).to_string(float_format="%.2f"))
    print(gex_top.to_string(float_format="%.2f"))
    log_top_strikes(ticker, gex_top)


def run(ticker, save_json, r_rate):
    spot, opt_df, raw = fetch_chain_and_spot(ticker, r_rate)
    # if save_json:
    #     jp = Path("data") / f"{ticker}.json"
    #     jp.write_text(json.dumps(raw))
    #     print(f"raw JSON saved → {jp}")

    spot, opt_df, raw = fetch_chain_and_spot(ticker, r_rate)

    if save_json:
        jp = Path("data") / f"{ticker}.json"
        jp.write_text(json.dumps(raw, default=str))
        print(f"raw JSON saved → {jp}")

    compute_total_gex(spot, opt_df, ticker)

# ────────────────────────────── CLI ───────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", help="symbol (e.g. QQQ, ^SPX)")
    ap.add_argument("--save_json", action="store_true")
    ap.add_argument("--risk_free", type=float, default=RISK_FREE, help="risk-free rate (decimal)")
    args = ap.parse_args()

    ticker = args.ticker.upper() if args.ticker else input("Enter ticker: ").upper()
    if not ticker.replace("^", "").isalpha():
        raise ValueError("Ticker must be letters, optionally starting with ^ for indices.")

    run(ticker, args.save_json, args.risk_free)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ATH Breakout Backtest Engine
Version: Stable Production Build (No UI)
Author: ChatGPT rewrite

Features:
---------
✓ Pure ATH breakout strategy
✓ 2-year backtest
✓ Pullback exit logic
✓ ATR stoploss
✓ Debug logs for triage
✓ Robust yfinance handling
✓ Strict + relaxed modes
"""

import os
import sys
import time
import json
import logging
import traceback
from datetime import datetime, timedelta
from collections import Counter

import yfinance as yf
import pandas as pd
import numpy as np

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

START_CAPITAL = 100000.0
CAPITAL = START_CAPITAL

PERIOD_STRING = "2y"        # Download period
WARMUP_BARS    = 200        # Minimum bars needed
ATR_PERIOD     = 14
ATR_MULT_SL    = 1.0        # Stop loss = Close - 1 ATR
CONFIRM_BARS   = 1          # ATH breakout confirmation by next bar

MIN_RR = 2.0                # Risk-reward filter
MAX_HOLD_DAYS = 20          # Exit after N days
LOGLEVEL = logging.INFO

# -------------------------------------------------------
# LOGGING SETUP
# -------------------------------------------------------

logger = logging.getLogger("ATH")
logger.setLevel(LOGLEVEL)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"
))
logger.addHandler(handler)
log = logger

# -------------------------------------------------------
# TICKER LOADING
# -------------------------------------------------------

def load_tickers_from_csv(path):
    """CSV must contain a 'Symbol' column with tickers like RELIANCE or RELIANCE.NS.
    This function normalizes and appends .NS if missing (so Yahoo symbols are correct).
    """
    if not os.path.exists(path):
        log.error(f"Ticker CSV not found: {path}")
        return []
    try:
        df = pd.read_csv(path)
        syms = []
        for s in df.get("Symbol", []):
            if pd.isna(s):
                continue
            s = str(s).strip()
            if not s:
                continue
            # normalize: add .NS if not present
            if not s.upper().endswith(".NS"):
                s = s + ".NS"
            syms.append(s.upper())
        syms = list(dict.fromkeys(syms))  # preserve order, unique
        log.info(f"Loaded {len(syms)} tickers from CSV")
        return syms
    except Exception as e:
        log.exception("Failed to read ticker CSV: %s", e)
        return []

# -------------------------------------------------------
# SAFE DOWNLOAD
# -------------------------------------------------------

def safe_download(ticker, period=PERIOD_STRING):
    """
    Downloads data and ensures Open/High/Low/Close are pandas Series.
    Returns None on failure.
    """
    try:
        df = yf.download(
            ticker,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=False
        )
    except Exception as e:
        log.warning(f"{ticker}: yfinance error: {e}")
        return None

    # Empty / no data
    if df is None or (hasattr(df, 'empty') and df.empty) or len(df) == 0:
        log.debug(f"{ticker}: no data returned by yfinance")
        return None

    # If df has multi-level columns (happens rarely), try to collapse if single ticker returned
    try:
        # If 'Close' is a DataFrame (multi-columns), pick numeric column if possible
        close_obj = df.get("Close")
        if isinstance(close_obj, pd.DataFrame):
            # choose first numeric subcolumn
            numcols = close_obj.select_dtypes(include=[np.number]).columns
            if len(numcols) > 0:
                df["Close"] = close_obj[numcols[0]]
            else:
                # fallback to first column
                df["Close"] = close_obj.iloc[:, 0]
    except Exception:
        # continue — we'll attempt to coerce below
        pass

    # Ensure index is a DatetimeIndex
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
    except Exception:
        pass

    # Now coerce Open/High/Low/Close into 1-d Series robustly
    for c in ["Open", "High", "Low", "Close"]:
        if c not in df.columns:
            continue
        col = df[c]
        # if scalar (weird), convert to Series with same index
        if not isinstance(col, (pd.Series, np.ndarray, list)):
            try:
                df[c] = pd.Series([col] * len(df), index=df.index)
            except Exception:
                # last resort: drop this ticker
                log.debug(f"{ticker}: column {c} not array-like, skipping ticker")
                return None
        else:
            # if it's a DataFrame-like object (2D), pick numeric column or flatten
            if isinstance(col, pd.DataFrame):
                numcols = col.select_dtypes(include=[np.number]).columns
                if len(numcols) > 0:
                    df[c] = col[numcols[0]].astype(float)
                else:
                    # fallback: first column
                    df[c] = col.iloc[:, 0].astype(float)
            else:
                # safe coercion for Series / ndarray
                try:
                    df[c] = pd.to_numeric(pd.Series(col, index=df.index), errors="coerce")
                except Exception:
                    try:
                        # try converting element-wise
                        df[c] = pd.Series(list(col), index=df.index).astype(float)
                    except Exception:
                        log.debug(f"{ticker}: failed to coerce column {c}")
                        return None

    # Drop rows where Close is NA
    df = df.dropna(subset=["Close"])
    if df is None or len(df) == 0:
        log.debug(f"{ticker}: no numeric close data after coercion")
        return None

    return df

# -------------------------------------------------------
# ATR CALCULATION
# -------------------------------------------------------

def calc_atr(df, period=ATR_PERIOD):
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

# -------------------------------------------------------
# HISTORICAL MAX (ATH BASE)
# -------------------------------------------------------

def highest_close(series):
    """
    Compute max of all previous closes (shifted),
    aligned & numeric-safe.
    """
    if isinstance(series, pd.DataFrame):
        nums = series.select_dtypes(include=[np.number]).columns
        if len(nums):
            s = series[nums[0]]
        else:
            s = series.iloc[:, 0]
    else:
        s = pd.Series(series)

    s = pd.to_numeric(s, errors="coerce")
    shifted = s.shift(1)
    rolling_max = shifted.cummax().fillna(0)

    rolling_max = pd.Series(rolling_max, index=s.index)
    return rolling_max.astype(float)

# -------------------------------------------------------
# INDICATOR PREPARATION
# -------------------------------------------------------

def prepare_indicators(df):
    """
    Adds:
        - ATR
        - RollingMax
        - ATH_break
        - Confirmed_break
        - Quality checks
    Returns:
        fully prepared DF or None (skip)
    """
    try:
        if df is None or len(df) < WARMUP_BARS:
            return None

        df = df.copy()

        # Ensure numeric prices
        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["ATR"] = calc_atr(df)
        df["PrevClose"] = df["Close"].shift(1)

        # Rolling max from previous bars
        rolling_max = highest_close(df["Close"])
        rolling_max.index = df.index
        df["RollingMax"] = rolling_max

        # ATH breakout: today's close > all-time-high until yesterday
        left, right = df["Close"].align(df["RollingMax"], join="left")
        df["ATH_break"] = left.gt(right).fillna(False)

        # Confirmation next bar closes above today's close
        df["Confirmed_break"] = (df["Close"].shift(-1) > df["Close"]) & df["ATH_break"]

        # Quality filter
        if df["ATR"].isna().mean() > 0.20:
            return None

        df = df.dropna(subset=["Close", "ATR", "RollingMax"])
        if len(df) < WARMUP_BARS:
            return None

        return df

    except Exception as e:
        log.exception("prepare_indicators failed: %s", e)
        return None

# -------------------------------------------------------
# PREPARE ALL TICKERS
# -------------------------------------------------------

def prepare_all_ticker_data(tickers):
    """
    Downloads each ticker, prepares indicators, filters bad data.
    Returns a dict: {ticker: prepared_df}
    """
    good = {}
    skipped = []

    log.info(f"Preparing {len(tickers)} tickers...")

    for t in tickers:
        df = safe_download(t)
        if df is None:
            skipped.append((t, "download_fail"))
            continue

        pdf = prepare_indicators(df)
        if pdf is None:
            skipped.append((t, "indicator_fail"))
            continue

        good[t] = pdf

        log.debug(f"{t}: raw_rows={len(df)}, indicator_rows={len(pdf)}")

    log.info(f"Prepared {len(good)} tickers; skipped {len(skipped)}.")
    if skipped:
        reasons = Counter([r for _, r in skipped])
        log.info(f"Skip reasons: {reasons}")

    return good

# -------------------------------------------------------
# TRADE SIGNAL GENERATION (ATH Entry Rules)
# -------------------------------------------------------

def generate_trade_signals(all_data):
    """
    Loop through each prepared ticker and detect ATH signals.

    BUY RULES:
    ----------
    1. Close[0] makes ATH_break == True
    2. Next bar confirms (Confirmed_break == True)
    3. ATR-based stoploss
    4. Risk/Reward ≥ MIN_RR

    Returns list of entries:
        {
            'ticker': 'RELIANCE.NS',
            'date': Timestamp,
            'entry': float,
            'stop': float,
            'target': float,
            'risk': float,
            'rr': float,
        }
    """

    entries = []
    diag = Counter()

    for ticker, df in all_data.items():
        closes = df["Close"]
        atr = df["ATR"]

        for i in range(len(df)-2):
            row = df.iloc[i]

            if not row["ATH_break"]:
                diag["no_ath"] += 1
                continue

            # Confirm breakout on next bar
            if not df.iloc[i+1]["Confirmed_break"]:
                diag["no_confirm"] += 1
                continue

            entry = float(row["Close"])
            stop = entry - (ATR_MULT_SL * float(row["ATR"]))
            risk = entry - stop
            if risk <= 0:
                diag["invalid_risk"] += 1
                continue

            rr_target = entry + (MIN_RR * risk)

            # Must fit R/R requirements
            rr = (rr_target - entry) / risk
            if rr < MIN_RR:
                diag["rr_fail"] += 1
                continue

            diag["accepted"] += 1

            entries.append({
                "ticker": ticker,
                "i": i,
                "date": df.index[i],
                "entry": entry,
                "stop": stop,
                "target": rr_target,
                "risk": risk,
                "rr": rr
            })

    log.info(f"Entry diag: {dict(diag)}")
    log.info(f"Generated {len(entries)} trade entries.")

    return entries

# -------------------------------------------------------
# BACKTEST SIMULATION ENGINE
# -------------------------------------------------------

def simulate_trade(df, start_index, entry, stop, target):
    """
    Simulate trade starting at index=start_index+1 to allow the entry candle
    to close first. We then evaluate subsequent bars:

    Exit Rules:
    -----------
    1. SL hit (Low <= stop → exit at stop or next open)
    2. Target hit (High >= target → exit at target or next open)
    3. Pullback exit:
       - After breakout, if Close < PrevClose OR Close < 0.98 * entry
    4. Time exit (max hold days)

    Returns:
        (exit_price, exit_date, result_string)
    """

    start_bar = start_index + 1
    max_index = min(len(df) - 1, start_bar + MAX_HOLD_DAYS)

    for i in range(start_bar, max_index + 1):
        row = df.iloc[i]
        low, high, close = row["Low"], row["High"], row["Close"]

        # 1. SL hit
        if low <= stop:
            # next open slippage
            next_open = df.iloc[i]["Open"]
            exit_price = min(stop, next_open)
            return exit_price, df.index[i], "LOSS"

        # 2. Target hit
        if high >= target:
            next_open = df.iloc[i]["Open"]
            exit_price = max(target, next_open)
            return exit_price, df.index[i], "WIN"

        # 3. Pullback exit
        prev_close = df["Close"].iloc[i-1]
        if close < prev_close or close < 0.98 * entry:
            return close, df.index[i], "PULLBACK_EXIT"

    # 4. Time-based exit (max hold reached)
    final_close = df["Close"].iloc[max_index]
    result = "WIN" if final_close >= entry else "LOSS"
    return final_close, df.index[max_index], result

# -------------------------------------------------------
# FULL BACKTEST PROCESSOR
# -------------------------------------------------------

def run_backtest(all_data, entries):
    """
    Given:
        all_data : {ticker: dataframe}
        entries  : list of entry dictionaries from generate_trade_signals()

    Returns:
        stats = {
            "start_capital": ...,
            "end_capital": ...,
            "profit": ...,
            "total_trades": ...,
            "wins": ...,
            "win_rate": ...,
            "max_drawdown": ...,
            "curve": [...],
            "history": [...]  # list of executed trade details
        }
    """

    equity = START_CAPITAL
    equity_curve = [equity]
    history = []

    for e in entries:
        tkr = e["ticker"]
        if tkr not in all_data:
            continue

        df = all_data[tkr]

        # simulate trade
        exit_price, exit_date, outcome = simulate_trade(
            df,
            e["i"],
            e["entry"],
            e["stop"],
            e["target"]
        )

        pnl = exit_price - e["entry"]
        qty = int((equity * 0.02) / max(e["risk"], 0.01))  # 2% capital per trade

        trade_pnl = pnl * qty
        equity += trade_pnl
        equity_curve.append(equity)

        history.append({
            "ticker": tkr,
            "entry_date": e["date"],
            "exit_date": exit_date,
            "entry": e["entry"],
            "exit": exit_price,
            "qty": qty,
            "result": outcome,
            "pnl": trade_pnl
        })

    total_trades = len(history)
    wins = sum(1 for h in history if h["pnl"] > 0)
    win_rate = round((wins / total_trades) * 100, 2) if total_trades else 0

    # Max Drawdown
    curve = pd.Series(equity_curve)
    peak = curve.cummax()
    dd = (curve - peak) / peak
    max_dd = round(dd.min() * -100, 2)

    stats = {
        "start_capital": START_CAPITAL,
        "end_capital": equity,
        "profit": round(equity - START_CAPITAL, 2),
        "total_trades": total_trades,
        "wins": wins,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "curve": [round(x, 2) for x in equity_curve],
        "history": history
    }

    log.info(f"=== BACKTEST SUMMARY ===")
    log.info(f"Start capital : ₹{START_CAPITAL}")
    log.info(f"End capital   : ₹{equity}")
    log.info(f"Trades        : {total_trades}")
    log.info(f"Wins          : {wins}")
    log.info(f"Win rate      : {win_rate}%")
    log.info(f"Total PnL     : ₹{stats['profit']}")
    log.info(f"MaxDD         : {stats['max_drawdown']}%")

    return stats

# -------------------------------------------------------
# SAVE JSON
# -------------------------------------------------------

def save_json(path, data):
    """Write JSON with indentation."""
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        log.info(f"Saved results to {path}")
    except Exception as e:
        log.error(f"Failed saving JSON {path}: {e}")

# -------------------------------------------------------
# PRETTY BANNERS
# -------------------------------------------------------

def banner(text):
    bar = "=" * (len(text) + 6)
    log.info("\n" + bar)
    log.info(f"== {text} ==")
    log.info(bar)

# -------------------------------------------------------
# FULL EXECUTION PIPELINE
# -------------------------------------------------------

def run_full_backtest():
    """High-level orchestration: load → prepare → signal → backtest → save."""
    banner("Loading Tickers")
    tickers = load_tickers_from_csv("ind_nifty500list.csv")
    if not tickers:
        log.error("No tickers loaded. Exiting.")
        return

    banner("Preparing Data")
    all_data = prepare_all_ticker_data(tickers)
    if not all_data:
        log.error("No prepared tickers. Exiting.")
        return
    log.info(f"Prepared {len(all_data)} tickers.")

    banner("Generating Signals")
    entries = generate_trade_signals(all_data)
    if not entries:
        log.warning("No signals generated.")
    else:
        log.info(f"{len(entries)} entries generated.")

    banner("Running Backtest")
    stats = run_backtest(all_data, entries)

    banner("Saving Results")
    save_json("backtest_stats.json", stats)

    banner("Completed")
    return stats

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    start = time.time()
    try:
        run_full_backtest()
    except Exception:
        log.error("Fatal error:")
        traceback.print_exc()
    finally:
        elapsed = time.time() - start
        log.info(f"Backtest finished in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()

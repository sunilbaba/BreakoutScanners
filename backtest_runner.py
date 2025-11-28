import os
import json
import math
import time
import logging
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np
import yfinance as yf

# -------------------------------------------------------
# LOGGING CONFIG
# -------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("BacktestATH")

# -------------------------------------------------------
# GLOBAL CONFIGURATION
# -------------------------------------------------------

DATA_YEARS = 2                        # 2 years of data
PERIOD_STRING = f"{DATA_YEARS}y"
WARMUP_BARS = 250                     # required bars for indicators
START_CAPITAL = 100000.0              # starting capital
RISK_PER_TRADE = 0.02                 # risk 2% per trade
BROKERAGE_PCT = 0.001                 # 0.1% brokerage

# ATH Strategy parameters
PULLBACK_EXIT = True                  # exit when close < prev close
ALLOW_MULTI_ENTRY = True              # multiple entries per ticker
CONFIRMATION_REQUIRED = True          # require next-day confirmation

# Output JSON file
OUTPUT_JSON = "backtest_stats.json"

# CSV with tickers (WITH .NS included)
TICKER_CSV = "ind_nifty500list.csv"
CSV_SYMBOL_COL = "Symbol"

# -------------------------------------------------------
# UTILITY FUNCTIONS
# -------------------------------------------------------

def load_tickers_from_csv(path=TICKER_CSV):
    """
    Loads tickers from ind_nifty500list.csv which contains tickers with '.NS'
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ticker file not found: {path}")

    df = pd.read_csv(path)
    if CSV_SYMBOL_COL not in df.columns:
        raise ValueError(f"CSV missing expected column: {CSV_SYMBOL_COL}")

    tickers = df[CSV_SYMBOL_COL].dropna().unique().tolist()
    tickers = [str(t).strip() for t in tickers]
    return tickers


def safe_download(ticker, period=PERIOD_STRING):
    """
    Downloads a single ticker safely with retries.
    """
    for attempt in range(3):
        try:
            df = yf.download(ticker, period=period, progress=False)
            if df is not None and len(df) > 0:
                df = df.copy()
                df.dropna(inplace=True)
                return df
        except Exception as e:
            log.warning(f"{ticker}: download attempt {attempt+1} failed: {e}")
            time.sleep(1 + attempt)
    return None


def highest_close(series):
    """
    Returns rolling all-time-high of closing prices.
    """
    return series.expanding().max()


def calc_atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)

    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1/period, adjust=False).mean()


def max_drawdown(equity_curve):
    """
    Computes max drawdown on an equity curve.
    """
    curve = np.array(equity_curve)
    peak = np.maximum.accumulate(curve)
    dd = (curve - peak) / peak
    return float(dd.min()) * 100.0

# -------------------------------------------------------
# INDICATOR PREPARATION
# -------------------------------------------------------

def prepare_indicators(df):
    """
    Adds ATR, rolling max, ATH flags.
    Requires >= WARMUP_BARS rows for proper signals.
    """
    df = df.copy()
    if len(df) < WARMUP_BARS:
        return None

    df["ATR"] = calc_atr(df)
    df["PrevClose"] = df["Close"].shift(1)

    # ATH breakout flag (raw)
    df["RollingMax"] = highest_close(df["Close"].shift(1).fillna(0))
    df["ATH_break"] = df["Close"] > df["RollingMax"]

    # ensure enough bars
    if df["ATR"].isna().sum() > (len(df) * 0.2):
        return None

    return df


# -------------------------------------------------------
# DATA PREPARATION (ALL TICKERS)
# -------------------------------------------------------

def prepare_all_ticker_data(tickers):
    """
    Download + prepare indicator data for all tickers.
    Returns a dict of {ticker: df} and skip diagnostics.
    """
    prepared = {}
    skipped = defaultdict(int)
    total = len(tickers)

    log.info(f"Downloading & preparing {total} tickers for {DATA_YEARS}y...")

    for t in tickers:
        df = safe_download(t)
        if df is None or len(df) < WARMUP_BARS:
            skipped["too_few_rows"] += 1
            continue

        # prepare indicators
        pdf = prepare_indicators(df)
        if pdf is None or len(pdf) < WARMUP_BARS:
            skipped["not_enough_indicator_rows"] += 1
            continue

        prepared[t] = pdf

        log.info(f"{t}: raw_rows={len(df)}, indicator_rows={len(pdf)}")

    log.info(f"Prepared {len(prepared)} tickers; skipped {total - len(prepared)} tickers.")

    # Summaries
    top_skip = [(f"{k}", v) for k, v in skipped.items()]
    log.info(f"Top skip reasons: {top_skip}")

    return prepared

# -------------------------------------------------------
# STRATEGY CORE: ATH BREAKOUT + NEXT-DAY CONFIRMATION
# -------------------------------------------------------

def generate_trade_signals(prepped):
    """
    Loop through each ticker’s DF and find ATH breakout entries.
    Entry rule:
        Day D:  Close > previous all-time-high (ATH_break = True)
        Day D+1: Confirmation close > D close (must exist)
    Stoploss (Option C):
        SL = previous ATH (RollingMax[D])
    Exit:
        - Pullback hit: Close < EMA5
        - Or SL hit intraday
        - Or Take Profit: 3 × ATR
    """

    entries = []          # Raw entries
    rejected_stats = Counter()

    for t, df in prepped.items():

        for i in range(WARMUP_BARS, len(df) - 2):
            row = df.iloc[i]

            # 1) ATH breakout occurred today
            if not row["ATH_break"]:
                rejected_stats["no_ath_break"] += 1
                continue

            # ATH breakout but closing weak?
            if row["Close"] <= row["RollingMax"]:
                rejected_stats["not_true_break"] += 1
                continue

            # 2) Confirmation must happen next day
            conf = df.iloc[i + 1]
            if conf["Close"] <= row["Close"]:
                rejected_stats["no_confirmation"] += 1
                continue

            # 3) Stoploss (Option C) — SL = previous ATH
            sl = row["RollingMax"]
            if sl <= 0 or sl >= row["Close"]:
                rejected_stats["invalid_sl"] += 1
                continue

            # 4) ATR must exist
            atr = float(row["ATR"])
            if atr <= 0:
                rejected_stats["no_atr"] += 1
                continue

            # 5) Take-profit = 3×ATR
            tgt = row["Close"] + 3 * atr

            entries.append({
                "symbol": t,
                "entry_date": df.index[i + 1],     # next day
                "entry_price": float(conf["Close"]),
                "sl": float(sl),
                "tgt": float(tgt),
                "atr": float(atr),
            })

    log.info(f"Entry diagnostics: {rejected_stats}")
    log.info(f"Total accepted entries: {len(entries)}")

    return entries

# -------------------------------------------------------
# Position sizing
# -------------------------------------------------------

def compute_position_size(entry_price, sl_price):
    """Risk-based sizing: 1% of capital per trade."""
    risk_per_trade = CAPITAL * 0.01
    stop_distance = entry_price - sl_price

    if stop_distance <= 0:
        return 0

    qty = int(risk_per_trade / stop_distance)
    if qty <= 0:
        return 0

    return qty


# -------------------------------------------------------
# Simulate each trade through future bars
# -------------------------------------------------------

def simulate_trade(df, entry_idx, entry_price, sl, tgt):
    """
    Simulates a trade starting at entry_idx.
    Exits:
      - Intraday SL
      - Intraday Target
      - Pullback Exit (Close < EMA5)
    """
    df = df.copy()
    df["EMA5"] = df["Close"].ewm(span=5).mean()

    for j in range(entry_idx, len(df)):
        day = df.iloc[j]
        low = day["Low"]
        high = day["High"]
        close = day["Close"]

        # 1) SL hit intraday
        if low <= sl:
            exit_price = sl
            return exit_price, "SL"

        # 2) TGT hit intraday
        if high >= tgt:
            exit_price = tgt
            return exit_price, "TP"

        # 3) Pullback exit at close
        if close < day["EMA5"]:
            exit_price = close
            return exit_price, "PULLBACK"

    # If never exited, last close
    return df.iloc[-1]["Close"], "END"


# -------------------------------------------------------
# Backtest execution
# -------------------------------------------------------

def run_backtest(prepped, entries):
    equity = CAPITAL
    curve = [equity]
    history = []

    grouped = defaultdict(list)
    for e in entries:
        grouped[e["symbol"]].append(e)

    for sym, trades in grouped.items():
        df = prepped[sym]
        for e in trades:
            entry_date = e["entry_date"]
            if entry_date not in df.index:
                continue

            entry_idx = df.index.get_loc(entry_date)
            entry_price = e["entry_price"]
            sl = e["sl"]
            tgt = e["tgt"]

            qty = compute_position_size(entry_price, sl)
            if qty <= 0:
                continue

            exit_price, exit_type = simulate_trade(df, entry_idx, entry_price, sl, tgt)

            pnl = (exit_price - entry_price) * qty

            history.append({
                "symbol": sym,
                "entry_date": str(entry_date),
                "exit_type": exit_type,
                "entry": entry_price,
                "exit": exit_price,
                "qty": qty,
                "pnl": pnl
            })

            equity += pnl
            curve.append(equity)

    wins = len([t for t in history if t["pnl"] > 0])
    total = len(history)
    win_rate = round((wins / total * 100), 2) if total else 0
    profit = round(equity - CAPITAL, 2)

    log.info(f"Backtest complete. Trades={total}, Wins={wins}, Win rate={win_rate}%, Profit={profit}")

    return {
        "curve": curve,
        "profit": profit,
        "total_trades": total,
        "win_rate": win_rate,
        "history": history
    }

# -------------------------------------------------------
# JSON Helpers
# -------------------------------------------------------

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log.info(f"Saved results to {path}")


# -------------------------------------------------------
# DIAGNOSTICS
# -------------------------------------------------------

def print_entry_summary(entries):
    sym_count = Counter([e["symbol"] for e in entries])
    log.info(f"Top entry Tickers: {sym_count.most_common(10)}")
    log.info(f"Total entries found: {len(entries)}")


def print_backtest_summary(stats):
    log.info("=== BACKTEST SUMMARY ===")
    log.info(f"Start capital : ₹{CAPITAL}")
    log.info(f"End capital   : ₹{CAPITAL + stats['profit']}")
    log.info(f"Trades        : {stats['total_trades']}")
    log.info(f"Wins          : {sum(1 for t in stats['history'] if t['pnl']>0)}")
    log.info(f"Win rate      : {stats['win_rate']}%")
    log.info(f"Total PnL     : ₹{stats['profit']}")


# -------------------------------------------------------
# MASTER RUNNER
# -------------------------------------------------------

def run_full_backtest():
    # 1) Load ticker list
    tickers = load_tickers_from_csv("ind_nifty500list.csv")
    if not tickers:
        log.error("Ticker list empty. Aborting.")
        return

    # 2) Download + prepare data
    prepped = prepare_all_ticker_data(tickers)
    if not prepped:
        log.error("No data prepared. Exiting.")
        return

    # 3) Find entries
    entries = generate_trade_signals(prepped)
    print_entry_summary(entries)

    if len(entries) == 0:
        log.warning("No entries found. Check rules or relax constraints.")
        stats = {
            "curve": [CAPITAL],
            "profit": 0,
            "total_trades": 0,
            "win_rate": 0,
            "history": []
        }
        save_json("backtest_stats.json", stats)
        return

    # 4) Run backtest simulation
    stats = run_backtest(prepped, entries)

    # 5) Save final result
    save_json("backtest_stats.json", stats)

    # 6) Print summary
    print_backtest_summary(stats)

# -------------------------------------------------------
# MISC IMPORTS USED LATER (ensure Counter is available)
# -------------------------------------------------------
from collections import Counter

# -------------------------------------------------------
# MAIN / CLI
# -------------------------------------------------------

def main():
    start = datetime.utcnow()
    log.info("Starting ATH Backtest Runner")
    try:
        run_full_backtest()
    except Exception as e:
        log.exception("Fatal error during backtest: %s", e)
    finally:
        end = datetime.utcnow()
        elapsed = (end - start).total_seconds()
        log.info("Backtest finished in %.2f seconds", elapsed)


if __name__ == "__main__":
    main()

# -------------------------------------------------------
# ADVANCED DIAGNOSTICS — optional but extremely useful
# -------------------------------------------------------

def diagnostic_count_ath(prepped):
    """
    Counts ATH breakouts per ticker for debugging.
    """
    results = {}
    for sym, df in prepped.items():
        if "ATH_break" in df:
            results[sym] = int(df["ATH_break"].sum())
    top = sorted(results.items(), key=lambda x: x[1], reverse=True)[:15]
    log.info("Top 15 tickers by ATH-break count:")
    for s, c in top:
        log.info(f"  {s}: {c}")
    return results


def diagnostic_entry_flow(entries):
    """
    Show distribution of entries across all tickers.
    """
    sym_count = Counter([e["symbol"] for e in entries])
    log.info("Entry distribution (top 20):")
    for sym, cnt in sym_count.most_common(20):
        log.info(f"  {sym}: {cnt} entries")


def diagnostic_trade_results(history):
    """
    Summaries for trade outcomes by type.
    """
    types = Counter([h["exit_type"] for h in history])
    log.info("Exit reason distribution:")
    for t, c in types.items():
        log.info(f"  {t}: {c}")

    sym_wins = Counter([h["symbol"] for h in history if h["pnl"] > 0])
    log.info("Top winning symbols:")
    for s, c in sym_wins.most_common(10):
        log.info(f"  {s}: {c} wins")


def dump_debug_csv(prepped, path="debug_tickers.csv"):
    """
    Save quick summary of each ticker to CSV for investigation.
    """
    rows = []
    for sym, df in prepped.items():
        rows.append({
            "symbol": sym,
            "rows": len(df),
            "ath_count": int(df["ATH_break"].sum()),
            "atr_min": float(df["ATR"].min()),
            "atr_max": float(df["ATR"].max()),
            "close_min": float(df["Close"].min()),
            "close_max": float(df["Close"].max())
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    log.info(f"Saved debug ticker summary → {path}")

# -------------------------------------------------------
# PRETTY LOGGING UTILITIES
# -------------------------------------------------------

COLOR = {
    "RESET": "\033[0m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "CYAN": "\033[96m",
    "YELLOW": "\033[93m",
    "MAGENTA": "\033[95m",
    "BLUE": "\033[94m",
    "GRAY": "\033[90m",
}

def banner(msg):
    """Nice readable section header."""
    bar = "=" * (len(msg) + 4)
    log.info(COLOR["CYAN"] + bar)
    log.info(f"| {msg} |")
    log.info(bar + COLOR["RESET"])


def pretty_kv(title, dct):
    log.info(COLOR["MAGENTA"] + f"--- {title} ---" + COLOR["RESET"])
    for k, v in dct.items():
        log.info(f"{k:20s}: {v}")


def enable_verbose():
    """Increase logging detail."""
    log.setLevel(logging.DEBUG)
    for handler in log.handlers:
        handler.setLevel(logging.DEBUG)
    log.debug("Verbose logging enabled.")


# -------------------------------------------------------
# OPTIONAL: CLI ARG PARSER (to toggle verbose mode)
# -------------------------------------------------------

import argparse

def parse_cli():
    parser = argparse.ArgumentParser(description="ATH Backtester Runner")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable detailed debug logging.")
    parser.add_argument("--dump", action="store_true",
                        help="Dump debug CSV with ticker stats.")
    return parser.parse_args()


# -------------------------------------------------------
# UPDATED MAIN TO SUPPORT VERBOSE MODE
# -------------------------------------------------------

def main():
    args = parse_cli()
    if args.verbose:
        enable_verbose()

    start = datetime.utcnow()
    banner("Starting ATH Breakout Backtest")

    try:
        tickers = load_tickers_from_csv("ind_nifty500list.csv")
        if not tickers:
            log.error("Ticker list empty — aborting.")
            return

        banner("Preparing Data")
        prepped = prepare_all_ticker_data(tickers)

        banner("Scanning for ATH Entries")
        entries = generate_trade_signals(prepped)
        print_entry_summary(entries)

        banner("Running Trade Simulation")
        stats = run_backtest(prepped, entries)

        banner("Backtest Summary")
        print_backtest_summary(stats)

        save_json("backtest_stats.json", stats)

        if args.dump:
            banner("Dumping Debug CSV")
            dump_debug_csv(prepped)

    except Exception as e:
        log.exception("Fatal error during run: %s", e)

    end = datetime.utcnow()
    log.info(f"Completed in {(end - start).total_seconds():.2f}s")
    banner("Done")

#!/usr/bin/env python3
"""
backtest_ath_pullback.py

Strategy: Enter after new ALL-TIME HIGH (ATH). Entry at next day's OPEN.
Protective stop = previous ATH level (breakout level) — used only if hit intraday.
Exit rule (your choice E): exit when any pullback occurs:
    Close < previous day's Close  -> exit at that day's Close

Backtest period: 2 years (DATA_PERIOD)
Saves results JSON to: ath_backtest_stats.json

Requirements:
    pip install yfinance pandas numpy
"""

import os
import json
import math
import logging
from datetime import datetime
from collections import defaultdict

import yfinance as yf
import pandas as pd
import numpy as np

# -------------------------
# CONFIG
# -------------------------
DATA_PERIOD = "2y"
CAPITAL = 100_000.0
RISK_PER_TRADE = 0.02      # fraction of equity risked per trade
BROKERAGE_PCT = 0.001      # round-trip as fraction of value (applied simply)
MIN_ROWS = 200             # skip tickers with too little history
CACHE_FILE = "ath_backtest_stats.json"
CSV_FILE = "ind_nifty500list.csv"

DEFAULT_TICKERS = ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "SBIN.NS"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BacktestATH")

# -------------------------
# HELPERS
# -------------------------
def load_tickers():
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            if 'Symbol' in df.columns:
                syms = [s.strip().upper() + ("" if s.strip().upper().endswith('.NS') else ".NS") for s in df['Symbol'].dropna().astype(str)]
                syms = [s for s in syms if s]
                logger.info("Loaded %d tickers from %s", len(syms), CSV_FILE)
                return sorted(list(dict.fromkeys(syms)))  # unique preserve order
        except Exception as e:
            logger.warning("Failed to read %s: %s", CSV_FILE, e)
    logger.info("Using default tickers (%d)", len(DEFAULT_TICKERS))
    return DEFAULT_TICKERS.copy()

def safe_download(ticker, period=DATA_PERIOD):
    """
    Download daily data for a single ticker and return prepared DataFrame or None.
    Use auto_adjust=True (yfinance changes).
    """
    try:
        df = yf.download(ticker, period=period, interval='1d', progress=False, auto_adjust=True, threads=False, ignore_tz=True)
    except Exception as e:
        logger.warning("Download failed for %s: %s", ticker, e)
        return None
    if df is None or df.empty:
        return None
    # Ensure required columns in expected names
    # yfinance with auto_adjust returns columns: Open, High, Low, Close, Volume
    needed = {'Open','High','Low','Close','Volume'}
    if not needed.issubset(df.columns):
        return None
    # drop rows with NaN in Close
    df = df.dropna(subset=['Close'])
    if len(df) < MIN_ROWS:
        logger.info("%s: too few rows (%d) — skipping", ticker, len(df))
        return None
    return df

def calc_max_drawdown(equity_curve):
    arr = np.array(equity_curve, dtype=float)
    if arr.size == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / peak
    max_dd = float(dd.min()) if dd.size>0 else 0.0
    return round(max_dd * 100, 2)  # percent negative

def risk_qty(equity, entry_price, stop_price):
    """
    Returns integer qty for risk-based sizing.
    If stop_price >= entry_price -> returns 0 (can't compute)
    """
    if stop_price >= entry_price:
        return 0
    risk_amount = equity * RISK_PER_TRADE
    per_share_risk = entry_price - stop_price
    if per_share_risk <= 0:
        return 0
    raw_qty = int(risk_amount / per_share_risk)
    if raw_qty <= 0:
        return 0
    # Prevent oversized positions: max 25% of equity exposure
    max_shares = int((equity * 0.25) / entry_price)
    if max_shares <= 0:
        return 0
    qty = min(raw_qty, max_shares)
    return max(qty, 0)

# -------------------------
# BACKTEST IMPLEMENTATION
# -------------------------
def run_backtest(tickers):
    equity = CAPITAL
    equity_curve = [equity]
    trades = []
    per_ticker_summary = defaultdict(lambda: {'trades':0, 'wins':0, 'pnl':0.0})

    for t_i, ticker in enumerate(tickers, start=1):
        logger.info("Downloading %s (%d/%d)", ticker, t_i, len(tickers))
        df = safe_download(ticker)
        if df is None:
            continue

        # compute running historical ATH (highest close prior to current day)
        # For day i (index i), the ATH up to previous day is max Close of df.iloc[:i]
        closes = df['Close']
        dates = df.index

        position = None  # dict with keys: entry_price, entry_date, sl, qty
        # sweep rows
        for i in range(1, len(df)):  # start at 1 because we may enter on day i using previous days info
            # compute historical ATH up to previous day (0..i-1)
            hist_max = float(closes.iloc[:i].max())  # ATH prior to day i
            prev_close = float(closes.iloc[i-1])
            today_open = float(df['Open'].iloc[i])
            today_low = float(df['Low'].iloc[i])
            today_close = float(closes.iloc[i])

            # If we have a position, check exits first (we evaluate market open -> intraday -> close)
            if position:
                # 1) intraday stop (protective stop = breakout level). If today's low <= SL -> exit at SL
                if today_low <= position['sl']:
                    exit_price = position['sl']
                    pnl = (exit_price - position['entry']) * position['qty']
                    fees = (position['entry'] * position['qty'] + exit_price * position['qty']) * BROKERAGE_PCT
                    net_pnl = pnl - fees
                    trades.append({
                        'symbol': ticker,
                        'entry_date': position['entry_date'].strftime("%Y-%m-%d"),
                        'entry': round(position['entry'],2),
                        'exit_date': dates[i].strftime("%Y-%m-%d"),
                        'exit': round(exit_price,2),
                        'qty': position['qty'],
                        'pnl': round(net_pnl,2),
                        'reason': 'stop_hit'
                    })
                    per_ticker_summary[ticker]['trades'] += 1
                    per_ticker_summary[ticker]['pnl'] += net_pnl
                    if net_pnl > 0: per_ticker_summary[ticker]['wins'] += 1
                    # release cash (simulated): equity increases by exit proceeds
                    equity += (exit_price * position['qty'] - fees)
                    equity_curve.append(round(equity,2))
                    position = None
                    # continue to next day after exit
                    continue

                # 2) exit at close if today's close is a pullback: Close < previous day's Close
                if today_close < prev_close:
                    exit_price = today_close
                    pnl = (exit_price - position['entry']) * position['qty']
                    fees = (position['entry'] * position['qty'] + exit_price * position['qty']) * BROKERAGE_PCT
                    net_pnl = pnl - fees
                    trades.append({
                        'symbol': ticker,
                        'entry_date': position['entry_date'].strftime("%Y-%m-%d"),
                        'entry': round(position['entry'],2),
                        'exit_date': dates[i].strftime("%Y-%m-%d"),
                        'exit': round(exit_price,2),
                        'qty': position['qty'],
                        'pnl': round(net_pnl,2),
                        'reason': 'pullback_close'
                    })
                    per_ticker_summary[ticker]['trades'] += 1
                    per_ticker_summary[ticker]['pnl'] += net_pnl
                    if net_pnl > 0: per_ticker_summary[ticker]['wins'] += 1
                    equity += (exit_price * position['qty'] - fees)
                    equity_curve.append(round(equity,2))
                    position = None
                    continue

                # else, keep holding (no exit today); equity not updated for this ticker until exit (we'll compute m2m later)
                # Continue scanning subsequent days
            else:
                # No position — check entry condition: was yesterday a NEW ATH? (i-1 day close > previous ATH)
                # Yesterday's close is closes.iloc[i-1] and hist_max (computed above) was max up to i-1 (which includes day 0..i-1)
                # We need to check if there was a NEW ATH formed at day i-1 -> close at day i-1 was greater than all prior closes
                prev_hist_max = float(closes.iloc[:i-1].max()) if i-1 > 0 else float(closes.iloc[:0].max()) if i-1==0 else -math.inf
                # Simpler: check if closes.iloc[i-1] > max of closes[:i-1]
                previous_ath_before_yesterday = float(closes.iloc[:i-1].max()) if i-1>0 else -math.inf
                yesterday_close = float(closes.iloc[i-1])
                # new ATH at yesterday if yesterday_close > previous_ath_before_yesterday
                if yesterday_close > previous_ath_before_yesterday:
                    # entry on today's OPEN (we already have today_open)
                    entry_price = today_open
                    breakout_level = yesterday_close  # previous ATH as SL
                    sl = breakout_level
                    qty = risk_qty(equity, entry_price, sl)
                    if qty <= 0:
                        # skip sizing-zero trades
                        continue
                    fees = (entry_price * qty) * BROKERAGE_PCT
                    # reduce equity by cost (simulate cash outflow)
                    equity -= (entry_price * qty + fees)
                    position = {
                        'symbol': ticker,
                        'entry': entry_price,
                        'entry_date': dates[i],
                        'sl': sl,
                        'qty': qty
                    }
                    # note: we don't immediately record a trade until exit
                    # add a small snapshot to equity curve to reflect position mark to market (we'll add actual m2m below at day's end)
                    # but for simplicity, append current equity
                    equity_curve.append(round(equity,2))
                    continue
                # else: no entry
                continue

        # End of ticker scanning: if position still open at end of data, close at last Close
        if position:
            last_close = float(closes.iloc[-1])
            exit_price = last_close
            pnl = (exit_price - position['entry']) * position['qty']
            fees = (position['entry'] * position['qty'] + exit_price * position['qty']) * BROKERAGE_PCT
            net_pnl = pnl - fees
            trades.append({
                'symbol': ticker,
                'entry_date': position['entry_date'].strftime("%Y-%m-%d"),
                'entry': round(position['entry'],2),
                'exit_date': dates[-1].strftime("%Y-%m-%d"),
                'exit': round(exit_price,2),
                'qty': position['qty'],
                'pnl': round(net_pnl,2),
                'reason': 'eod_close'
            })
            per_ticker_summary[ticker]['trades'] += 1
            per_ticker_summary[ticker]['pnl'] += net_pnl
            if net_pnl > 0: per_ticker_summary[ticker]['wins'] += 1
            equity += (exit_price * position['qty'] - fees)
            equity_curve.append(round(equity,2))
            position = None

    # End all tickers
    total_trades = len(trades)
    wins = len([t for t in trades if t['pnl'] > 0])
    losses = total_trades - wins
    total_pnl = round(sum(t['pnl'] for t in trades), 2)
    avg_pnl = round((total_pnl / total_trades), 2) if total_trades else 0.0
    win_rate = round(100.0 * wins / total_trades, 2) if total_trades else 0.0
    max_dd = calc_max_drawdown(equity_curve)

    stats = {
        "updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "portfolio": {
            "start_capital": CAPITAL,
            "end_capital": round(equity,2),
            "curve": equity_curve,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl_per_trade": avg_pnl,
            "max_drawdown_pct": max_dd
        },
        "trades": trades,
        "per_ticker": {k: v for k, v in per_ticker_summary.items()}
    }

    # Save JSON
    with open(CACHE_FILE, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info("Backtest complete.")
    logger.info("Trades: %d | Wins: %d | Win rate: %s%% | Total PnL: %s | MaxDD: %s%%",
                total_trades, wins, win_rate, total_pnl, max_dd)
    return stats

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    tickers = load_tickers()
    logger.info("Starting ATH backtest on %d tickers (period=%s)", len(tickers), DATA_PERIOD)
    stats = run_backtest(tickers)
    # Print summary
    p = stats['portfolio']
    print("=== BACKTEST SUMMARY ===")
    print(f"Start capital : ₹{p['start_capital']}")
    print(f"End capital   : ₹{p['end_capital']}")
    print(f"Trades        : {p['total_trades']}")
    print(f"Wins          : {p['wins']}")
    print(f"Win rate      : {p['win_rate']}%")
    print(f"Total PnL     : ₹{p['total_pnl']}")
    print(f"Avg PnL/trade : ₹{p['avg_pnl_per_trade']}")
    print(f"Max Drawdown  : {p['max_drawdown_pct']}%")
    print(f"Saved results to {CACHE_FILE}")

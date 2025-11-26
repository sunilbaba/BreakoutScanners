#!/usr/bin/env python3
"""
backtest_runner.py
- Runs ONCE daily (e.g., 8 PM IST).
- Downloads 2 Years data for Nifty 500 (or CSV list).
- Simulation: Runs strategy backtest.
- Output: Saves 'backtest_stats.json' with Win Rates, Equity Curve, and Trade Ledger.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime
from time import sleep

# --- CONFIG ---
DATA_PERIOD = "2y"                  # <- changed to 2 years as requested
CACHE_FILE = "backtest_stats.json"
CAPITAL = 100_000.0
RISK_PER_TRADE = 0.02
BROKERAGE_PCT = 0.001
MAX_POSITION_PERC = 0.25           # max capital per position (percent of total capital)
MIN_ROWS_REQUIRED = 220            # minimal rows for indicator windows (approx 1 year)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Backtester")

SECTOR_INDICES = {
    "NIFTY 50": "^NSEI", "BANK": "^NSEBANK", "AUTO": "^CNXAUTO",
    "IT": "^CNXIT", "METAL": "^CNXMETAL", "PHARMA": "^CNXPHARMA",
    "FMCG": "^CNXFMCG", "ENERGY": "^CNXENERGY", "REALTY": "^CNXREALTY",
    "PSU BANK": "^CNXPSUBANK"
}

# --- 1. DATA HELPERS ---
def get_tickers():
    if os.path.exists("ind_nifty500list.csv"):
        try:
            df = pd.read_csv("ind_nifty500list.csv")
            syms = [f"{x}.NS" for x in df['Symbol'].dropna().unique()]
            logger.info(f"Loaded {len(syms)} tickers from CSV")
            return syms
        except Exception as e:
            logger.info("Failed to read CSV list, using defaults: %s", e)
    # fallback sample set
    return ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "SBIN.NS"]

def robust_download(tickers, period=DATA_PERIOD, batch_size=20):
    logger.info(f"📥 Downloading {len(tickers)} symbols ({period}) in batches of {batch_size}...")
    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        tries = 0
        while tries < 3:
            try:
                data = yf.download(batch, period=period, group_by='ticker', threads=True, progress=False, ignore_tz=True)
                # If single ticker, yfinance sometimes returns single-level columns.
                frames.append(data)
                break
            except Exception as e:
                tries += 1
                logger.info("Download batch failed (attempt %d) for %s: %s", tries, batch, e)
                sleep(1.0 * tries)
        else:
            logger.info("Skipping batch after retries: %s", batch)
    if not frames:
        logger.info("No data downloaded.")
        return pd.DataFrame()
    try:
        combined = pd.concat(frames, axis=1)
        return combined
    except Exception as e:
        logger.info("Concat error: %s", e)
        # if concatenation fails, return first non-empty frame
        for f in frames:
            if isinstance(f, pd.DataFrame) and not f.empty:
                return f
    return pd.DataFrame()

def _tz_safe_index(df):
    df = df.copy()
    try:
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
    except Exception:
        try:
            df.index = pd.to_datetime(df.index).tz_localize(None)
        except Exception:
            pass
    return df

def extract_df(bulk, ticker):
    """
    Extract per-ticker dataframe from yfinance 'bulk' output.
    Handles MultiIndex and single-DataFrame cases.
    Returns None when data insufficient or missing.
    """
    try:
        if bulk is None or bulk.empty:
            return None
        # MultiIndex (many tickers)
        if isinstance(bulk.columns, pd.MultiIndex):
            if ticker in bulk.columns.get_level_values(0):
                df = bulk[ticker].copy().dropna()
                df = _tz_safe_index(df)
                if len(df) < MIN_ROWS_REQUIRED:
                    logger.info("Skipping %s (rows=%d < %d)", ticker, len(df), MIN_ROWS_REQUIRED)
                    return None
                return df
            else:
                return None
        else:
            # bulk likely is already a per-ticker DataFrame
            if all(c in bulk.columns for c in ['Open', 'High', 'Low', 'Close']):
                df = bulk.copy().dropna()
                df = _tz_safe_index(df)
                if len(df) < MIN_ROWS_REQUIRED:
                    logger.info("Single-frame data has insufficient rows: %d", len(df))
                    return None
                return df
    except Exception as e:
        logger.debug("extract_df error for %s: %s", ticker, e)
    return None

# --- 2. INDICATORS & PREP ---
def true_range(df):
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df, period=14):
    return true_range(df).ewm(alpha=1/period, adjust=False).mean()

def wilder_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss)).fillna(50)

def adx(df, period=14):
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), -minus_dm, 0.0)
    tr = true_range(df).ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / tr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / tr)
    out = (abs(plus_di - minus_di) / (plus_di + minus_di) * 100)
    return out.ewm(alpha=1/period, adjust=False).mean().fillna(0)

def prepare_df(df):
    """
    Adds indicators and returns a new DataFrame. Assumes df has at least MIN_ROWS_REQUIRED rows.
    """
    df = df.copy()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['ATR'] = atr(df)
    df['RSI'] = wilder_rsi(df['Close'])
    df['ADX'] = adx(df)
    return df

# --- 3. WIN RATE on single stock (for UI display) ---
def calc_stock_win_rate(df):
    if df is None or len(df) < 150:
        return 0
    df = prepare_df(df)
    wins, total = 0, 0
    # Check last ~6 months (approx)
    start = max(0, len(df) - 130)
    for i in range(start, len(df)-10):
        row = df.iloc[i]
        if row['Close'] > row['EMA20'] and row['RSI'] > 60 and row['ADX'] > 25:
            stop = row['Close'] - row['ATR']
            target = row['Close'] + (3 * row['ATR'])
            outcome = "OPEN"
            for j in range(1, 15):
                if i+j >= len(df): break
                fut = df.iloc[i+j]
                if fut['Low'] <= stop:
                    outcome = "LOSS"; break
                if fut['High'] >= target:
                    outcome = "WIN"; break
            if outcome != "OPEN":
                total += 1
                if outcome == "WIN": wins += 1
    return round((wins/total*100), 0) if total > 0 else 0

# --- 4. PORTFOLIO SIMULATION ---
def run_portfolio_sim(bulk_data, tickers):
    processed = {}
    for t in tickers:
        df = extract_df(bulk_data, t)
        if df is not None and len(df) > MIN_ROWS_REQUIRED:
            processed[t] = prepare_df(df)
        else:
            logger.debug("Not processed: %s", t)

    if not processed:
        logger.info("No tickers processed - exiting sim.")
        return [], [], 0, 0, 0

    # unify dates
    dates = sorted(list(set().union(*[d.index for d in processed.values()])))
    sim_dates = dates[150:]  # skip initial warmup

    cash = CAPITAL
    equity_curve = [CAPITAL]
    portfolio = []
    history = []

    for date in sim_dates:
        # --- EXITS ---
        active = []
        for trade in portfolio:
            sym = trade['symbol']
            if date not in processed[sym].index:
                active.append(trade); continue
            row = processed[sym].loc[date]
            exit_p = None

            # priority: gap open triggers, then intra-day low/high
            if row['Open'] < trade['sl']:
                exit_p = row['Open']
            elif row['Low'] <= trade['sl']:
                exit_p = trade['sl']
            elif row['Open'] > trade['tgt']:
                exit_p = row['Open']
            elif row['High'] >= trade['tgt']:
                exit_p = trade['tgt']

            if exit_p:
                pnl = (exit_p - trade['entry']) * trade['qty']
                cost = (exit_p * trade['qty'] * BROKERAGE_PCT)
                real_pnl = pnl - cost - trade['entry_cost']
                cash += (exit_p * trade['qty'] - cost)
                history.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "symbol": sym,
                    "entry": round(trade['entry'], 2),
                    "exit": round(exit_p, 2),
                    "pnl": round(real_pnl, 2),
                    "status": "WIN" if real_pnl > 0 else "LOSS"
                })
            else:
                active.append(trade)
        portfolio = active

        # --- ENTRIES ---
        if len(portfolio) < 5:
            for sym, df in processed.items():
                if date not in df.index: continue
                row = df.loc[date]
                # entry filter
                if row['Close'] > row['EMA20'] and row['RSI'] > 60 and row['ADX'] > 25 and row['Close'] > row['SMA200']:
                    if any(t['symbol'] == sym for t in portfolio): continue
                    risk = row['ATR']
                    if not risk or risk <= 0: continue
                    # position sizing
                    qty = int((equity_curve[-1] * RISK_PER_TRADE) / risk)
                    # cap by max position percentage
                    if (qty * row['Close']) > (CAPITAL * MAX_POSITION_PERC):
                        qty = int((CAPITAL * MAX_POSITION_PERC) / row['Close'])
                    cost = qty * row['Close']
                    if qty > 0 and cash > (cost + cost * BROKERAGE_PCT):
                        fees = cost * BROKERAGE_PCT
                        cash -= (cost + fees)
                        portfolio.append({
                            "symbol": sym, "entry": row['Close'], "qty": qty,
                            "sl": row['Close'] - risk, "tgt": row['Close'] + (3*risk),
                            "entry_cost": fees
                        })
                        logger.debug("Entered %s on %s qty=%d entry=%.2f", sym, date.strftime("%Y-%m-%d"), qty, row['Close'])
                        if len(portfolio) >= 5: break

        # --- MARK-TO-MARKET EQUITY ---
        m2m = 0
        for t in portfolio:
            sym = t['symbol']
            price = processed[sym].loc[date]['Close'] if date in processed[sym].index else t['entry']
            m2m += (price * t['qty'])
        equity_curve.append(round(cash + m2m, 2))

    wins = len([h for h in history if h['pnl'] > 0])
    win_rate = round(wins / len(history) * 100, 1) if history else 0
    profit = round(equity_curve[-1] - CAPITAL, 2)
    return equity_curve, history, win_rate, len(history), profit

# --- MAIN ---
if __name__ == "__main__":
    tickers = get_tickers()
    all_syms = tickers + list(SECTOR_INDICES.values())

    bulk = robust_download(all_syms, period=DATA_PERIOD, batch_size=20)

    # 1. Individual Win Rates
    logger.info("📊 Calculating Win Rates for individual tickers...")
    ticker_stats = {}
    for t in tickers:
        df = extract_df(bulk, t)
        if df is not None:
            try:
                ticker_stats[t.replace('.NS','')] = calc_stock_win_rate(df)
            except Exception as e:
                logger.debug("Win rate calc failed for %s: %s", t, e)
                ticker_stats[t.replace('.NS','')] = 0
        else:
            ticker_stats[t.replace('.NS','')] = 0

    # 2. Portfolio Sim
    logger.info("📈 Running Portfolio Simulation...")
    curve, ledger, win_rate, trades, profit = run_portfolio_sim(bulk, tickers)

    output = {
        "updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "portfolio": {
            "curve": curve,
            "ledger": ledger[-50:],  # keep last 50 trades to keep file light
            "win_rate": win_rate,
            "total_trades": trades,
            "profit": profit
        },
        "tickers": ticker_stats,
        "last_run": datetime.utcnow().strftime("%Y-%m-%d")
    }

    with open(CACHE_FILE, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"✅ Saved stats to {CACHE_FILE} (profit={profit}, trades={trades}, win_rate={win_rate}%)")

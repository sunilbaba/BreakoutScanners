#!/usr/bin/env python3
"""
backtest_runner.py — deeper diagnostic edition
- Downloads 2y data, robust extraction.
- Runs SIGNAL DIAGNOSTIC with very relaxed thresholds.
- If still zero signals, prints indicator samples (last 10 rows) for a few tickers.
- Produces backtest_stats.json as before.
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
DATA_PERIOD = "2y"
CACHE_FILE = "backtest_stats.json"
CAPITAL = 100_000.0
RISK_PER_TRADE = 0.02
BROKERAGE_PCT = 0.001
MAX_POSITION_PERC = 0.25

# Debug / tuning
MIN_ROWS_REQUIRED = 150
BATCH_SIZE = 5
DIAGNOSTIC = True

# VERY relaxed thresholds for deep debug
DEBUG_RSI_THRESHOLD = 50    # allow neutral RSI
DEBUG_ADX_THRESHOLD = 5     # allow very low ADX

# Toggle to temporarily ignore SMA200 filter
DEBUG_ALLOW_NO_SMA200 = True

# How many tickers to show indicator samples for when signals==0
INDICATOR_SAMPLE_TICKERS = 8

# Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Backtester")

SECTOR_INDICES = {
    "NIFTY 50": "^NSEI", "BANK": "^NSEBANK", "AUTO": "^CNXAUTO",
    "IT": "^CNXIT", "METAL": "^CNXMETAL", "PHARMA": "^CNXPHARMA",
    "FMCG": "^CNXFMCG", "ENERGY": "^CNXENERGY", "REALTY": "^CNXREALTY",
    "PSU BANK": "^CNXPSUBANK"
}

# -------------------------
# Data helpers
# -------------------------
def get_tickers():
    if os.path.exists("ind_nifty500list.csv"):
        try:
            df = pd.read_csv("ind_nifty500list.csv")
            syms = [f"{x}.NS" for x in df['Symbol'].dropna().unique()]
            logger.info(f"Loaded {len(syms)} tickers from CSV")
            return syms
        except Exception as e:
            logger.info("Failed to read CSV list, using defaults: %s", e)
    return [
        "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "SBIN.NS",
        "HINDUNILVR.NS", "ICICIBANK.NS", "LT.NS", "AXISBANK.NS", "BHARTIARTL.NS"
    ]

def _tz_safe_index(df):
    df = df.copy()
    try:
        df.index = pd.to_datetime(df.index)
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_convert(None)
    except Exception:
        try:
            df.index = pd.to_datetime(df.index).tz_localize(None)
        except Exception:
            pass
    return df

def robust_download(tickers, period=DATA_PERIOD, batch_size=BATCH_SIZE):
    logger.info(f"📥 Downloading {len(tickers)} symbols ({period}) in batches of {batch_size}...")
    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        success = False
        for attempt in range(3):
            try:
                data = yf.download(batch, period=period, group_by='ticker', threads=False, progress=False, ignore_tz=True)
                if isinstance(data, pd.DataFrame) and not data.empty:
                    frames.append(data)
                    success = True
                    break
            except Exception as e:
                logger.info("Batch download failed attempt %d for %s: %s", attempt+1, batch, e)
                sleep(1.0 * (attempt+1))
        if not success:
            logger.info("Falling back to per-symbol download for batch: %s", batch)
            for sym in batch:
                tries = 0
                while tries < 2:
                    try:
                        df = yf.download(sym, period=period, threads=False, progress=False, ignore_tz=True)
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            frames.append(df)
                            break
                    except Exception as e:
                        logger.debug("Single download fail for %s: %s", sym, e)
                        sleep(0.5)
                    tries += 1
    if not frames:
        logger.info("No data frames downloaded.")
        return pd.DataFrame()
    try:
        combined = pd.concat(frames, axis=1)
        return combined
    except Exception as e:
        logger.info("Concat failed: %s - returning first non-empty frame", e)
        for f in frames:
            if isinstance(f, pd.DataFrame) and not f.empty:
                return f
    return pd.DataFrame()

def extract_df(bulk, ticker):
    try:
        if bulk is None or bulk.empty:
            return None
        if isinstance(bulk.columns, pd.MultiIndex):
            level0 = list(bulk.columns.get_level_values(0).unique())
            if ticker in level0:
                df = bulk[ticker].copy().dropna()
            else:
                tbase = ticker.replace('.NS', '')
                if tbase in level0:
                    df = bulk[tbase].copy().dropna()
                else:
                    found = None
                    for cand in level0:
                        if cand.upper().startswith(tbase.upper()):
                            found = cand; break
                    if found:
                        df = bulk[found].copy().dropna()
                    else:
                        return None
            df = _tz_safe_index(df)
            if len(df) < MIN_ROWS_REQUIRED:
                logger.debug("Ticker %s has %d rows (<%d)", ticker, len(df), MIN_ROWS_REQUIRED)
                return None
            return df
        else:
            if all(c in bulk.columns for c in ['Open','High','Low','Close']):
                df = bulk.copy().dropna()
                df = _tz_safe_index(df)
                if len(df) < MIN_ROWS_REQUIRED:
                    logger.debug("Single-frame data has insufficient rows: %d", len(df))
                    return None
                return df
    except Exception as e:
        logger.debug("extract_df error for %s: %s", ticker, e)
    return None

# -------------------------
# Indicators
# -------------------------
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
    # avoid division by zero - safe compute
    denom = (plus_di + minus_di).replace(0, np.nan)
    adx_raw = (abs(plus_di - minus_di) / denom * 100).fillna(0)
    return adx_raw.ewm(alpha=1/period, adjust=False).mean().fillna(0)

def prepare_df(df):
    df = df.copy()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['ATR'] = atr(df)
    df['RSI'] = wilder_rsi(df['Close'])
    df['ADX'] = adx(df)
    return df

# -------------------------
# Signal counting & samples
# -------------------------
def count_signals_for_df(df):
    count = 0
    df = prepare_df(df)
    for i in range(0, len(df)-1):
        row = df.iloc[i]
        sma_ok = True if DEBUG_ALLOW_NO_SMA200 else (row['Close'] > row['SMA200'])
        # check values exist and are numbers
        if pd.isna(row.get('EMA20')) or pd.isna(row.get('RSI')) or pd.isna(row.get('ADX')): 
            continue
        if row['Close'] > row['EMA20'] and row['RSI'] >= DEBUG_RSI_THRESHOLD and row['ADX'] >= DEBUG_ADX_THRESHOLD and sma_ok:
            if pd.notna(row.get('ATR')) and row['ATR'] > 0:
                count += 1
    return count

def print_indicator_sample(df, ticker, n=10):
    df = prepare_df(df)
    tail = df.tail(n)[['Open','High','Low','Close','SMA200','EMA20','ATR','RSI','ADX']].copy()
    print(f"=== INDICATOR SAMPLE FOR {ticker} (last {n} rows) ===")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 120):
        print(tail.to_string())
    print("=== END SAMPLE ===")

# -------------------------
# Win-rate & sim (use relaxed thresholds)
# -------------------------
def calc_stock_win_rate(df):
    if df is None or len(df) < 150:
        return 0
    df = prepare_df(df)
    wins, total = 0, 0
    start = max(0, len(df) - 130)
    for i in range(start, len(df)-10):
        row = df.iloc[i]
        if pd.isna(row.get('EMA20')) or pd.isna(row.get('RSI')) or pd.isna(row.get('ADX')): continue
        if row['Close'] > row['EMA20'] and row['RSI'] >= DEBUG_RSI_THRESHOLD and row['ADX'] >= DEBUG_ADX_THRESHOLD:
            stop = row['Close'] - row['ATR']
            tgt = row['Close'] + (3 * row['ATR'])
            outcome = "OPEN"
            for j in range(1, 15):
                if i + j >= len(df): break
                fut = df.iloc[i + j]
                if fut['Low'] <= stop:
                    outcome = "LOSS"; break
                if fut['High'] >= tgt:
                    outcome = "WIN"; break
            if outcome != "OPEN":
                total += 1
                if outcome == "WIN": wins += 1
    return round((wins / total * 100), 0) if total > 0 else 0

def run_portfolio_sim(bulk_data, tickers):
    processed = {}
    for t in tickers:
        df = extract_df(bulk_data, t)
        if df is not None and len(df) >= MIN_ROWS_REQUIRED:
            processed[t] = prepare_df(df)
    logger.info("Processed tickers for sim: %d/%d", len(processed), len(tickers))
    if len(processed) < 10:
        logger.info("Processed list (sample): %s", sorted(list(processed.keys()))[:20])
    if not processed:
        logger.info("No tickers processed - exiting sim.")
        return [], [], 0, 0, 0
    dates = sorted(list(set().union(*[d.index for d in processed.values()])))
    sim_dates = dates[150:]
    cash = CAPITAL
    equity_curve = [CAPITAL]
    portfolio = []
    history = []
    for date in sim_dates:
        active = []
        for trade in portfolio:
            sym = trade['symbol']
            if date not in processed[sym].index:
                active.append(trade); continue
            row = processed[sym].loc[date]
            exit_p = None
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
                    "entry": round(trade['entry'],2),
                    "exit": round(exit_p,2),
                    "pnl": round(real_pnl,2),
                    "status": "WIN" if real_pnl>0 else "LOSS"
                })
            else:
                active.append(trade)
        portfolio = active
        if len(portfolio) < 5:
            for sym, df in processed.items():
                if date not in df.index: continue
                row = df.loc[date]
                if pd.isna(row.get('EMA20')) or pd.isna(row.get('RSI')) or pd.isna(row.get('ADX')): continue
                sma_ok = True if DEBUG_ALLOW_NO_SMA200 else (row['Close'] > row['SMA200'])
                if row['Close'] > row['EMA20'] and row['RSI'] >= DEBUG_RSI_THRESHOLD and row['ADX'] >= DEBUG_ADX_THRESHOLD and sma_ok:
                    if any(t['symbol'] == sym for t in portfolio): continue
                    risk = row['ATR']
                    if not risk or risk <= 0: continue
                    qty = int((equity_curve[-1] * RISK_PER_TRADE) / risk)
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
                        if len(portfolio) >= 5: break
        m2m = 0
        for t in portfolio:
            sym = t['symbol']
            price = processed[sym].loc[date]['Close'] if date in processed[sym].index else t['entry']
            m2m += (price * t['qty'])
        equity_curve.append(round(cash + m2m,2))
    wins = len([h for h in history if h['pnl']>0])
    win_rate = round(wins / len(history) * 100, 1) if history else 0
    profit = round(equity_curve[-1] - CAPITAL, 2)
    return equity_curve, history, win_rate, len(history), profit

# -------------------------
# Diagnostic: count signals across tickers & show samples if needed
# -------------------------
def signal_diagnostic_and_samples(bulk, tickers):
    print("=== SIGNAL DIAGNOST

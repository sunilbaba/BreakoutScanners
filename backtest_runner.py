#!/usr/bin/env python3
"""
backtest_runner.py
Robust 2-year backtest for NIFTY500:
- Batched downloads with per-ticker fallback.
- Defensive prepare_df (doesn't drop entire DF).
- Diagnostic skipped-tickers report.
- Produces backtest_stats.json
"""

import os
import time
import json
import logging
from datetime import datetime
from collections import Counter

import yfinance as yf
import pandas as pd
import numpy as np

# -------------------------
# CONFIG
# -------------------------
DATA_PERIOD = "2y"                # <- 2 years as requested
CAPITAL = 100_000.0
RISK_PER_TRADE = 0.02
BROKERAGE_PCT = 0.001

# Strategy / thresholds (tuneable)
MIN_HISTORY_ROWS = 200           # safe lower bound for 2y ≈ 480 days
MIN_INDICATOR_ROWS = 120         # minimum rows with computed indicators to accept a ticker
BATCH_SIZE = 40
BATCH_RETRIES = 2

ADX_MIN = 18
RSI_MIN = 55
RR_MIN = 1.8
VOLUME_SPIKE = 1.0
BREAKOUT_LOOKBACK = 20

# TSL rules
TSL_BE_R = 1.0
TSL_EMA20_R = 2.0
TSL_SWING_R = 3.0

CACHE_FILE = "backtest_stats.json"

DEFAULT_TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "SBIN.NS",
    "ICICIBANK.NS", "LT.NS", "AXISBANK.NS", "ITC.NS", "HINDUNILVR.NS"
]

# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BacktestRunner")

# -------------------------
# INDICATOR HELPERS
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
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_adx(df, period=14):
    up = df['High'].diff()
    dn = df['Low'].diff() * -1
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(df)
    tr_smoothed = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / tr_smoothed)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / tr_smoothed)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.ewm(alpha=1/period, adjust=False).mean().fillna(0)

def compute_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iat[i] > df['Close'].iat[i-1]:
            obv.append(obv[-1] + int(df['Volume'].iat[i]))
        elif df['Close'].iat[i] < df['Close'].iat[i-1]:
            obv.append(obv[-1] - int(df['Volume'].iat[i]))
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

# -------------------------
# PREPARE DF (defensive)
# -------------------------
def prepare_df(df, require_full=False):
    """
    Compute indicators. Do NOT drop rows aggressively.
    If require_full=True, returns only rows that have all key indicators.
    Otherwise keep all rows and set _has_all_inds flag.
    """
    df = df.copy()

    # Normalize required columns (case-insensitive)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            for c in df.columns:
                if c.lower() == col.lower():
                    df[col] = df[c]
                    break

    if not set(["Open", "High", "Low", "Close", "Volume"]).issubset(df.columns):
        return pd.DataFrame()

    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA20_SLOPE'] = df['EMA20'].diff()
    df['ATR'] = atr(df)
    df['RSI'] = wilder_rsi(df['Close'])
    df['ADX'] = compute_adx(df)
    df['VOL_SMA20'] = df['Volume'].rolling(20).mean()
    df['OBV'] = compute_obv(df)
    df['OBV_SMA'] = df['OBV'].rolling(20).mean()

    key_inds = ["SMA50", "SMA200", "EMA20", "ATR", "RSI", "ADX", "VOL_SMA20", "OBV", "OBV_SMA"]
    df['_has_all_inds'] = df[key_inds].notna().all(axis=1)

    if require_full:
        df2 = df[df['_has_all_inds']].copy()
        df2.drop(columns=['_has_all_inds'], inplace=True, errors='ignore')
        return df2
    return df

# -------------------------
# WEEKLY AGGREGATION (used as an additional filter)
# -------------------------
def make_weekly(df):
    if df.empty:
        return df
    weekly = df.resample('W').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    if weekly.empty:
        return weekly
    weekly['EMA20'] = weekly['Close'].ewm(span=20, adjust=False).mean()
    weekly['EMA20_SLOPE'] = weekly['EMA20'].diff()
    return weekly

# -------------------------
# Download + prepare (batched + fallback)
# -------------------------
def download_and_prepare_tickers(tickers, period=DATA_PERIOD, batch_size=BATCH_SIZE, batch_retries=BATCH_RETRIES):
    prepared_map = {}
    skipped = []

    logger.info(f"Starting batched download for {len(tickers)} tickers (period={period})")

    # split into batches
    for start in range(0, len(tickers), batch_size):
        batch = tickers[start:start+batch_size]
        success = False
        for attempt in range(batch_retries+1):
            try:
                logger.info(f"Batch download: {len(batch)} tickers (attempt {attempt+1})")
                raw = yf.download(batch, period=period, interval='1d', group_by='ticker',
                                  progress=False, threads=True, ignore_tz=True, auto_adjust=True)
                success = True
                break
            except Exception as e:
                logger.warning(f"Batch download failure (attempt {attempt+1}): {e}")
                time.sleep(1.5 * (attempt+1))
                raw = None

        # If batch failed, fallback to per-ticker download
        if raw is None or raw.empty:
            logger.info(f"Falling back to individual downloads for batch starting at {batch[0]}")
            for t in batch:
                pdf, reason = try_single_download_and_prepare(t, period)
                if pdf is not None:
                    prepared_map[t] = pdf
                else:
                    skipped.append((t, reason))
            continue

        # If batch succeeded, raw may be a MultiIndex
        if isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
            tickers_in_raw = raw.columns.get_level_values(0).unique()
            for t in tickers_in_raw:
                try:
                    sub = raw[t].dropna(how='all')
                except Exception:
                    sub = None
                pdf, reason = prepare_candidate_from_raw(sub, t)
                if pdf is not None:
                    prepared_map[t] = pdf
                else:
                    skipped.append((t, reason))
        else:
            # Single DF returned for this batch — treat as individual downloads
            for t in batch:
                pdf, reason = try_single_download_and_prepare(t, period)
                if pdf is not None:
                    prepared_map[t] = pdf
                else:
                    skipped.append((t, reason))

    if not prepared_map:
        logger.error("No tickers prepared successfully. Aborting.")
        return None, {}, skipped

    # Build bulk MultiIndex DataFrame
    try:
        bulk = pd.concat(prepared_map, axis=1)
    except Exception as e:
        logger.error(f"Concatenate failed: {e}")
        return None, prepared_map, skipped

    logger.info(f"Prepared {len(prepared_map)} tickers; skipped {len(skipped)} tickers.")
    # show top skip reasons
    counter = Counter([r for (_, r) in skipped])
    logger.info(f"Top skip reasons: {counter.most_common(8)}")
    return bulk, prepared_map, skipped

def try_single_download_and_prepare(ticker, period):
    """Download single ticker and prepare; return (pdf, reason)"""
    try:
        raw = yf.download(ticker, period=period, interval='1d', progress=False, threads=True,
                          ignore_tz=True, auto_adjust=True)
    except Exception as e:
        return None, f"exception_download:{e}"

    return prepare_candidate_from_raw(raw, ticker)

def prepare_candidate_from_raw(raw, ticker):
    """Given a raw DF (or None), validate and prepare; return (pdf, reason)"""
    if raw is None or raw.empty:
        return None, "no_data"
    # check required columns
    cols = set(raw.columns)
    req = {"Open", "High", "Low", "Close", "Volume"}
    if not req.issubset(cols):
        return None, f"missing_cols:{list(cols)}"
    raw_rows = len(raw)
    if raw_rows < MIN_HISTORY_ROWS:
        return None, f"too_few_rows:{raw_rows}"
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    pdf = prepare_df(df, require_full=False)
    if pdf is None or pdf.empty:
        return None, "prepare_empty"
    full_rows = int(pdf["_has_all_inds"].sum()) if "_has_all_inds" in pdf.columns else 0
    if full_rows < MIN_INDICATOR_ROWS:
        return None, f"not_enough_indicator_rows:{full_rows}"
    logger.info(f"{ticker}: raw_rows={raw_rows}, indicator_rows={full_rows}")
    return pdf, None

# -------------------------
# Backtest engine
# -------------------------
class BacktestEngine:
    def __init__(self, bulk_data, tickers):
        """
        bulk_data: MultiIndex DataFrame returned from pd.concat({ticker: df})
        tickers: list of tickers keys (e.g., "RELIANCE.NS")
        """
        self.bulk = bulk_data
        self.tickers = [t for t in tickers if not str(t).startswith('^')]
        self.cash = CAPITAL
        self.equity_curve = [CAPITAL]
        self.portfolio = []  # open positions
        self.history = []    # closed trades

    def calc_qty(self, entry, sl, equity):
        risk_amt = equity * RISK_PER_TRADE
        per_share_risk = abs(entry - sl)
        if per_share_risk <= 0: return 0
        qty = int(risk_amt / per_share_risk)
        max_qty = int((equity * 0.25) / entry) if entry > 0 else 0
        if qty * entry > equity * 0.25:
            qty = max_qty
        return max(qty, 0)

    def process_exit(self, trade, row, date):
        exit_price = None
        # stop/gap
        if row['Open'] < trade['sl']:
            exit_price = row['Open']
        elif row['Low'] <= trade['sl']:
            exit_price = trade['sl']
        # target
        if row['Open'] > trade['target']:
            exit_price = row['Open']
        elif row['High'] >= trade['target']:
            exit_price = trade['target']
        if exit_price is None:
            return False

        qty = trade['qty']
        revenue = exit_price * qty
        cost = revenue * BROKERAGE_PCT
        entry_cost = trade['entry'] * qty * BROKERAGE_PCT
        pnl = revenue - cost - (trade['entry'] * qty + entry_cost)

        self.history.append({
            "symbol": trade['symbol'],
            "entry": round(trade['entry'], 2),
            "exit": round(exit_price, 2),
            "qty": qty,
            "entry_date": trade['entry_date'].strftime("%Y-%m-%d"),
            "exit_date": date.strftime("%Y-%m-%d"),
            "sl": round(trade['sl'], 2),
            "target": round(trade['target'], 2),
            "rr": round((trade['target'] - trade['entry']) / (trade['entry'] - trade['sl']) if (trade['entry'] - trade['sl'])!=0 else 0, 2),
            "pnl": round(pnl, 2)
        })

        self.cash += (revenue - cost)
        return True

    def apply_tsl(self, trade, row, df, idx):
        # Move SL to BE at +1R
        atr_v = row['ATR']
        entry = trade['entry']
        if row['High'] >= entry + (TSL_BE_R * atr_v):
            trade['sl'] = max(trade['sl'], entry)
        # Move SL to EMA20 at +2R
        if row['High'] >= entry + (TSL_EMA20_R * atr_v):
            if not pd.isna(row.get('EMA20', np.nan)):
                trade['sl'] = max(trade['sl'], row['EMA20'])
        # Move SL to swing low at +3R
        if row['High'] >= entry + (TSL_SWING_R * atr_v):
            if idx >= 5:
                swing_low = df['Low'].iloc[idx-5:idx].min()
                trade['sl'] = max(trade['sl'], swing_low)

    def calc_recent_high(self, df, date, lookback=BREAKOUT_LOOKBACK):
        if date not in df.index:
            return None
        i = df.index.get_loc(date)
        if i - lookback < 0:
            return df['High'].iloc[:i].max() if i>0 else None
        return df['High'].iloc[i-lookback:i].max()

    def run(self):
        logger.info("Backtest engine: building processed map...")
        processed = {}
        weekly_map = {}
        # Build processed dict from bulk (bulk is MultiIndex with top level tickers)
        for t in self.tickers:
            try:
                if isinstance(self.bulk, dict):
                    df = self.bulk.get(t)
                elif isinstance(self.bulk.columns, pd.MultiIndex):
                    if t in self.bulk.columns.get_level_values(0):
                        df = self.bulk[t].copy()
                    else:
                        df = None
                else:
                    df = None
            except Exception:
                df = None

            if df is None or len(df) < MIN_HISTORY_ROWS:
                continue

            if 'ATR' not in df.columns or 'EMA20' not in df.columns:
                df = prepare_df(df, require_full=False)

            if df is None or df.empty or len(df) < MIN_HISTORY_ROWS:
                continue

            processed[t] = df
            weekly_map[t] = make_weekly(df)

        if not processed:
            logger.error("No processed tickers available for backtest.")
            return {
                "curve": [CAPITAL],
                "profit": 0,
                "total_trades": 0,
                "win_rate": 0,
                "history": [],
                "last_run": datetime.utcnow().strftime("%Y-%m-%d")
            }

        all_dates = sorted(set().union(*[df.index for df in processed.values()]))
        for date in all_dates:
            # exits
            new_port = []
            for trade in self.portfolio:
                df = processed.get(trade['symbol'])
                if df is None or date not in df.index:
                    new_port.append(trade); continue
                row = df.loc[date]
                idx = df.index.get_loc(date)
                self.apply_tsl(trade, row, df, idx)
                if not self.process_exit(trade, row, date):
                    new_port.append(trade)
            self.portfolio = new_port

            # entries
            if len(self.portfolio) < 5:
                for sym, df in processed.items():
                    if date not in df.index: continue
                    if len(self.portfolio) >= 5: break
                    row = df.loc[date]

                    # weekly quick checks
                    wdf = weekly_map.get(sym)
                    if wdf is None or len(wdf) < 8: continue
                    last_week = wdf.iloc[-1]
                    if last_week['Close'] < last_week['EMA20'] or last_week['EMA20_SLOPE'] <= 0: continue

                    # daily checks
                    if not (row['Close'] > row['EMA20'] and row['EMA20_SLOPE'] > 0 and row['Close'] > row['SMA200']):
                        continue
                    if row['RSI'] < RSI_MIN or row['ADX'] < ADX_MIN: continue
                    if pd.isna(row.get('VOL_SMA20', None)) or row['Volume'] < VOLUME_SPIKE * row['VOL_SMA20']: continue

                    recent_high = self.calc_recent_high(df, date)
                    if recent_high is None: continue
                    if not (row['Close'] >= recent_high * 0.995): continue

                    if pd.isna(row.get('OBV', None)) or pd.isna(row.get('OBV_SMA', None)): continue
                    if row['OBV'] < row['OBV_SMA'] * 0.95: continue

                    sl = row['Close'] - row['ATR']
                    target = row['Close'] + 2.5 * row['ATR']
                    if sl >= row['Close']: continue
                    rr = (target - row['Close']) / (row['Close'] - sl) if (row['Close'] - sl)!=0 else 0
                    if rr < RR_MIN: continue

                    qty = self.calc_qty(row['Close'], sl, self.equity_curve[-1])
                    if qty <= 0 or qty * row['Close'] > self.equity_curve[-1]: continue

                    entry_cost = row['Close'] * qty * BROKERAGE_PCT
                    self.cash -= (row['Close'] * qty + entry_cost)
                    self.portfolio.append({
                        "symbol": sym, "entry": row['Close'], "sl": sl, "target": target,
                        "qty": qty, "entry_date": date
                    })

            # mark-to-market equity
            m2m = self.cash
            for t in self.portfolio:
                df = processed.get(t['symbol'])
                if df is None:
                    m2m += t['entry'] * t['qty']
                else:
                    if date in df.index:
                        m2m += df.loc[date]['Close'] * t['qty']
                    else:
                        m2m += t['entry'] * t['qty']
            self.equity_curve.append(round(m2m, 2))

        # summary
        if not self.history:
            return {
                "curve": [round(x, 2) for x in self.equity_curve],
                "profit": round(self.equity_curve[-1] - CAPITAL, 2),
                "total_trades": 0,
                "win_rate": 0,
                "history": [],
                "last_run": datetime.utcnow().strftime("%Y-%m-%d")
            }

        wins = [h for h in self.history if h['pnl'] > 0]
        losses = [h for h in self.history if h['pnl'] <= 0]
        win_rate = round(100 * len(wins) / len(self.history), 2) if self.history else 0
        avg_win = round(np.mean([h['pnl'] for h in wins]) if wins else 0, 2)
        avg_loss = round(np.mean([h['pnl'] for h in losses]) if losses else 0, 2)
        avg_rr = round(np.mean([h['rr'] for h in self.history]) if self.history else 0, 2)

        curve_series = pd.Series(self.equity_curve)
        roll_max = curve_series.cummax()
        drawdown = (curve_series - roll_max) / roll_max
        max_dd = float(drawdown.min())

        ret = curve_series.pct_change().dropna()
        sharpe = float(round((ret.mean() / ret.std() * (252 ** 0.5)) if ret.std() != 0 else 0, 3))

        return {
            "curve": [round(x, 2) for x in self.equity_curve],
            "profit": round(self.equity_curve[-1] - CAPITAL, 2),
            "total_trades": len(self.history),
            "win_rate": win_rate,
            "history": self.history,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_rr": avg_rr,
            "max_drawdown": round(max_dd, 4),
            "sharpe_ratio": sharpe,
            "last_run": datetime.utcnow().strftime("%Y-%m-%d")
        }

# -------------------------
# MAIN runner
# -------------------------
def run_backtest_main():
    # load tickers from csv if available
    if os.path.exists("ind_nifty500list.csv"):
        try:
            symdf = pd.read_csv("ind_nifty500list.csv")
            tickers = [f"{s}.NS" for s in symdf['Symbol'].dropna().unique()]
        except Exception as e:
            logger.warning(f"Failed to read CSV: {e}; using defaults")
            tickers = DEFAULT_TICKERS.copy()
    else:
        tickers = DEFAULT_TICKERS.copy()

    # sanitize out placeholders (TMP, DUMMY, etc)
    cleaned = []
    for t in tickers:
        base = t.replace('.NS', '').upper()
        if base.startswith(('TMP','DUMMY','ZZ','TEST')):
            logger.info(f"Filtering placeholder symbol: {t}")
            continue
        cleaned.append(t)
    tickers = cleaned

    # Download + prepare
    bulk, prepared_map, skipped = download_and_prepare_tickers(tickers, period=DATA_PERIOD)

    if bulk is None:
        logger.error("No data prepared. Exiting.")
        out = {
            "updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "prepared": 0,
            "skipped": skipped,
            "last_run": datetime.utcnow().strftime("%Y-%m-%d")
        }
        with open(CACHE_FILE, 'w') as f:
            json.dump(out, f, indent=2)
        return out

    # run engine
    engine = BacktestEngine(bulk, list(prepared_map.keys()))
    report = engine.run()

    out = {
        "updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "portfolio": {
            "curve": report.get("curve", [CAPITAL]),
            "profit": report.get("profit", 0),
            "total_trades": report.get("total_trades", 0),
            "win_rate": report.get("win_rate", 0),
            "history": report.get("history", [])[-1000:]
        },
        "tickers": {k.replace('.NS',''): 0 for k in prepared_map.keys()},
        "skipped": skipped,
        "avg_win": report.get("avg_win", 0),
        "avg_loss": report.get("avg_loss", 0),
        "avg_rr": report.get("avg_rr", 0),
        "max_drawdown": report.get("max_drawdown", 0),
        "sharpe_ratio": report.get("sharpe_ratio", 0),
        "last_run": report.get("last_run", datetime.utcnow().strftime("%Y-%m-%d"))
    }

    with open(CACHE_FILE, 'w') as f:
        json.dump(out, f, indent=2)

    logger.info("Backtest finished.")
    logger.info(f"Prepared tickers: {len(prepared_map)} | Skipped: {len(skipped)}")
    logger.info(f"Trades: {out['portfolio']['total_trades']} | Win Rate: {out['portfolio']['win_rate']}% | Profit: {out['portfolio']['profit']}")
    logger.info(f"Saved results to {CACHE_FILE}")
    return out

if __name__ == '__main__':
    run_backtest_main()

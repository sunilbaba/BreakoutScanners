#!/usr/bin/env python3
"""
backtest_runner.py

- Reads tickers from ind_nifty500list.csv (Symbol column) or uses DEFAULT_TICKERS.
- Downloads 2 years of daily data from yfinance (batched with per-ticker fallback).
- Prepares indicators and runs backtest engine.
- Saves backtest_stats.json.

Install: pip install yfinance pandas numpy
Run: python backtest_runner.py
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
DATA_PERIOD = "2y"                # 2 years data
CAPITAL = 100_000.0
RISK_PER_TRADE = 0.02
BROKERAGE_PCT = 0.001

# Strategy thresholds (production)
ADX_MIN = 18
RSI_MIN = 55
RR_MIN = 1.8
BREAKOUT_LOOKBACK = 20

# Relaxed thresholds for debug pass
RELAXED_ADX_MIN = 14
RELAXED_RSI_MIN = 50
RELAXED_RR_MIN = 1.4

MIN_HISTORY_ROWS = 200
MIN_INDICATOR_ROWS = 120

BATCH_SIZE = 40
BATCH_RETRIES = 2
CACHE_FILE = "backtest_stats.json"

DEFAULT_TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "SBIN.NS"
]

# TSL / trailing logic params
TSL_BE_R = 1.0
TSL_EMA20_R = 2.0
TSL_SWING_R = 3.0

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BacktestA_YF")

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
    return 100 - (100 / (1 + gain / loss)).fillna(50)

def compute_adx(df, period=14):
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), -minus_dm, 0.0)
    tr = true_range(df).ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / tr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / tr)
    adx = (abs(plus_di - minus_di) / (plus_di + minus_di) * 100).ewm(alpha=1/period, adjust=False).mean().fillna(0)
    return adx

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
# PREPARE DF
# -------------------------
def prepare_df(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    # normalize column names (some yf downloads include lowercase)
    mapping = {}
    for c in df.columns:
        lc = c.lower()
        if lc == 'adj close':
            mapping[c] = 'Close'
        elif lc in ('open','high','low','close','volume'):
            mapping[c] = lc.capitalize()
    if mapping:
        df = df.rename(columns=mapping)
    # required columns
    if not set(["Open","High","Low","Close","Volume"]).issubset(df.columns):
        return pd.DataFrame()
    # indicators
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
    # signal-ready rows count:
    return df

# -------------------------
# DOWNLOAD HELPERS (batched with fallback)
# -------------------------
def normalize_ticker(sym):
    s = str(sym).strip().upper()
    if not s.endswith('.NS') and not s.startswith('^'):
        s = s + '.NS'
    return s

def try_single_download_and_prepare(ticker, period=DATA_PERIOD):
    try:
        df = yf.download(ticker, period=period, interval='1d', progress=False, threads=True, ignore_tz=True, auto_adjust=True)
    except Exception as e:
        return None, f"exception:{e}"
    if df is None or df.empty:
        return None, "no_data"
    pdf = prepare_df(df)
    if pdf.empty:
        return None, "prepare_empty"
    raw_rows = len(df)
    valid_rows = int(pdf.dropna(subset=['SMA50','SMA200','EMA20','ATR','RSI','ADX']).shape[0])
    if raw_rows < MIN_HISTORY_ROWS:
        return None, f"too_few_rows:{raw_rows}"
    if valid_rows < MIN_INDICATOR_ROWS:
        return None, f"not_enough_indicator_rows:{valid_rows}"
    logger.info(f"{ticker}: raw_rows={raw_rows}, indicator_rows={valid_rows}")
    return pdf, None

def download_and_prepare_tickers(tickers, period=DATA_PERIOD, batch_size=BATCH_SIZE, batch_retries=BATCH_RETRIES):
    prepared_map = {}
    skipped = []
    logger.info(f"Downloading {len(tickers)} tickers for {period}...")
    for start in range(0, len(tickers), batch_size):
        batch = tickers[start:start+batch_size]
        raw = None
        for attempt in range(batch_retries+1):
            try:
                logger.info(f"Batch download attempt {attempt+1} for {len(batch)} tickers...")
                raw = yf.download(batch, period=period, interval='1d', group_by='ticker', progress=False, threads=True, ignore_tz=True, auto_adjust=True)
                break
            except Exception as e:
                logger.warning(f"Batch download failed (attempt {attempt+1}): {e}")
                time.sleep(1.5 * (attempt+1))
                raw = None
        if raw is None or raw.empty:
            logger.info(f"Batch fallback: downloading individually for {len(batch)} tickers...")
            for t in batch:
                pdf, reason = try_single_download_and_prepare(t, period)
                if pdf is not None:
                    prepared_map[t] = pdf
                else:
                    skipped.append((t, reason))
            continue

        # raw may be multiindex or single dataframe
        if isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
            cols0 = raw.columns.get_level_values(0).unique()
            for t in cols0:
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
            # single df case (unlikely for batch), fallback to per-ticker
            for t in batch:
                pdf, reason = try_single_download_and_prepare(t, period)
                if pdf is not None:
                    prepared_map[t] = pdf
                else:
                    skipped.append((t, reason))

    if not prepared_map:
        logger.error("No tickers prepared successfully. Aborting.")
        return None, {}, skipped
    counter = Counter([r for (_, r) in skipped])
    logger.info(f"Prepared {len(prepared_map)} tickers; skipped {len(skipped)} tickers.")
    logger.info(f"Top skip reasons: {counter.most_common(8)}")
    return prepared_map, prepared_map, skipped

def prepare_candidate_from_raw(raw, ticker):
    if raw is None or raw.empty:
        return None, "no_data"
    # raw may have correct columns already
    pdf = prepare_df(raw)
    if pdf.empty:
        return None, "prepare_empty"
    raw_rows = len(raw)
    valid_rows = int(pdf.dropna(subset=['SMA50','SMA200','EMA20','ATR','RSI','ADX']).shape[0])
    if raw_rows < MIN_HISTORY_ROWS:
        return None, f"too_few_rows:{raw_rows}"
    if valid_rows < MIN_INDICATOR_ROWS:
        return None, f"not_enough_indicator_rows:{valid_rows}"
    logger.info(f"{ticker}: raw_rows={raw_rows}, indicator_rows={valid_rows}")
    return pdf, None

# -------------------------
# BACKTEST ENGINE (keeps previous logic)
# -------------------------
class BacktestEngine:
    def __init__(self, prepared_map):
        self.map = prepared_map
        self.tickers = [t for t in prepared_map.keys() if not str(t).startswith('^')]
        self.cash = CAPITAL
        self.equity_curve = [CAPITAL]
        self.portfolio = []
        self.history = []
        self._entry_diag = Counter()

    def calc_qty(self, entry, sl, equity):
        risk_amt = equity * RISK_PER_TRADE
        risk_share = abs(entry - sl)
        if risk_share <= 0: return 0
        qty = int(risk_amt / risk_share)
        max_qty = int((equity * 0.25) / entry) if entry > 0 else 0
        if qty * entry > equity * 0.25:
            qty = max_qty
        return max(qty, 0)

    def calc_recent_high(self, df, date, lookback=BREAKOUT_LOOKBACK):
        if date not in df.index:
            return None
        i = df.index.get_loc(date)
        if i - lookback < 0:
            return df['High'].iloc[:i].max() if i>0 else None
        return df['High'].iloc[i-lookback:i].max()

    def apply_tsl(self, trade, row, df, idx):
        atr_v = row['ATR']
        entry = trade['entry']
        if row['High'] >= entry + (TSL_BE_R * atr_v):
            trade['sl'] = max(trade['sl'], entry)
        if row['High'] >= entry + (TSL_EMA20_R * atr_v):
            if not pd.isna(row.get('EMA20', np.nan)):
                trade['sl'] = max(trade['sl'], row['EMA20'])
        if row['High'] >= entry + (TSL_SWING_R * atr_v):
            if idx >= 5:
                swing_low = df['Low'].iloc[idx-5:idx].min()
                trade['sl'] = max(trade['sl'], swing_low)

    def process_exit(self, trade, row, date):
        exit_price = None
        if row['Open'] < trade['sl']:
            exit_price = row['Open']
        elif row['Low'] <= trade['sl']:
            exit_price = trade['sl']
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

    def _run_with_thresholds(self, adx_min, rsi_min, rr_min, breakout_lookback):
        self.cash = CAPITAL
        self.equity_curve = [CAPITAL]
        self.portfolio = []
        self.history = []
        self._entry_diag = Counter()

        processed = self.map
        weekly_map = {t: df.resample('W').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna() for t, df in processed.items()}
        for t, w in weekly_map.items():
            if not w.empty:
                w['EMA20'] = w['Close'].ewm(span=20, adjust=False).mean()
                w['EMA20_SLOPE'] = w['EMA20'].diff()

        # union dates
        dates = sorted(list(set().union(*[df.index for df in processed.values()])))
        for date in dates:
            # Exits
            active = []
            for trade in self.portfolio:
                sym = trade['symbol']
                df = processed.get(sym)
                if df is None or date not in df.index:
                    active.append(trade); continue
                row = df.loc[date]
                idx = df.index.get_loc(date)
                self.apply_tsl(trade, row, df, idx)
                if not self.process_exit(trade, row, date):
                    active.append(trade)
            self.portfolio = active

            # Entries
            if len(self.portfolio) < 5:
                filter_counts = Counter()
                for sym, df in processed.items():
                    if date not in df.index: continue
                    if len(self.portfolio) >= 5: break
                    row = df.loc[date]

                    # weekly quick checks (relaxed)
                    wdf = weekly_map.get(sym)
                    if wdf is None or len(wdf) < 3:
                        pass
                    else:
                        last_week = wdf.iloc[-1]
                        ema_slope = last_week.get('EMA20_SLOPE', 0)
                        two_week_avg = wdf['Close'].iloc[-2:].mean() if len(wdf) >= 2 else last_week['Close']
                        if (last_week['Close'] < last_week['EMA20']) and (ema_slope <= -0.5) and (two_week_avg < last_week['EMA20']):
                            filter_counts["weekly_down"] += 1
                            continue

                    # daily basic
                    if not (row['Close'] > row['EMA20'] and row['Close'] > row['SMA200']):
                        filter_counts["daily_basic"] += 1
                        continue

                    # RSI/ADX
                    if row['RSI'] < rsi_min:
                        filter_counts["rsi"] += 1
                        continue
                    if row['ADX'] < adx_min:
                        filter_counts["adx"] += 1
                        continue

                    # volume / obv (lenient)
                    if pd.isna(row.get('VOL_SMA20', None)) or row['Volume'] < 0.8 * row['VOL_SMA20']:
                        filter_counts["vol"] += 1
                        continue
                    if pd.isna(row.get('OBV', None)) or pd.isna(row.get('OBV_SMA', None)):
                        filter_counts["obv"] += 1
                        continue
                    if row['OBV'] < 0.8 * row['OBV_SMA']:
                        filter_counts["obv"] += 1
                        continue

                    # recent high breakout (lenient)
                    i = df.index.get_loc(date)
                    recent_high = df['High'].iloc[max(0, i-breakout_lookback):i].max() if i>0 else None
                    if recent_high is None or not (row['Close'] >= recent_high * 0.99):
                        filter_counts["recent_high"] += 1
                        continue

                    # rr & qty
                    sl = row['Close'] - row['ATR']
                    target = row['Close'] + (2.5 * row['ATR'])
                    if sl >= row['Close']:
                        filter_counts["rr"] += 1
                        continue
                    rr = (target - row['Close']) / (row['Close'] - sl) if (row['Close'] - sl) != 0 else 0
                    if rr < rr_min:
                        filter_counts["rr"] += 1
                        continue
                    qty = int((self.equity_curve[-1] * RISK_PER_TRADE) / (row['Close'] - sl)) if (row['Close'] - sl) != 0 else 0
                    if qty <= 0 or qty * row['Close'] > self.cash:
                        filter_counts["qty"] += 1
                        continue

                    # accept
                    fees = row['Close'] * qty * BROKERAGE_PCT
                    self.cash -= (row['Close'] * qty + fees)
                    self.portfolio.append({
                        "symbol": sym, "entry": row['Close'], "qty": qty,
                        "sl": sl, "target": target, "entry_cost": fees, "entry_date": date
                    })
                    filter_counts["accepted"] += 1

                self._entry_diag.update(filter_counts)

            # Equity
            m2m = self.cash
            for t in self.portfolio:
                sym = t['symbol']
                df = processed.get(sym)
                if df is None:
                    m2m += t['entry'] * t['qty']
                else:
                    if date in df.index:
                        m2m += df.loc[date]['Close'] * t['qty']
                    else:
                        m2m += t['entry'] * t['qty']
            self.equity_curve.append(round(m2m, 2))

        # results
        if not self.history:
            return {
                "curve": [round(x,2) for x in self.equity_curve],
                "profit": round(self.equity_curve[-1] - CAPITAL, 2),
                "total_trades": 0,
                "win_rate": 0,
                "history": [],
                "entry_diag": dict(self._entry_diag),
                "last_run": datetime.utcnow().strftime("%Y-%m-%d")
            }

        wins = [h for h in self.history if h['pnl'] > 0]
        win_rate = round(100 * len(wins) / len(self.history), 2) if self.history else 0
        return {
            "curve": [round(x,2) for x in self.equity_curve],
            "profit": round(self.equity_curve[-1] - CAPITAL, 2),
            "total_trades": len(self.history),
            "win_rate": win_rate,
            "history": self.history,
            "entry_diag": dict(self._entry_diag),
            "last_run": datetime.utcnow().strftime("%Y-%m-%d")
        }

    def run(self):
        logger.info("Backtest: running strict pass (production thresholds)")
        res = self._run_with_thresholds(ADX_MIN, RSI_MIN, RR_MIN, BREAKOUT_LOOKBACK)
        logger.info(f"Strict pass entry breakdown: {res.get('entry_diag', {})}")
        if res.get("total_trades", 0) > 0:
            return res
        logger.info("Strict pass produced 0 trades — running one RELAXED pass to surface signals")
        res_relaxed = self._run_with_thresholds(RELAXED_ADX_MIN, RELAXED_RSI_MIN, RELAXED_RR_MIN, max(BREAKOUT_LOOKBACK, 40))
        logger.info(f"Relaxed pass entry breakdown: {res_relaxed.get('entry_diag', {})}")
        return res_relaxed

# -------------------------
# MAIN
# -------------------------
def get_tickers_from_csv_or_default():
    if os.path.exists("ind_nifty500list.csv"):
        try:
            df = pd.read_csv("ind_nifty500list.csv")
            if 'Symbol' in df.columns:
                syms = [normalize_ticker(x) for x in df['Symbol'].dropna().unique()]
                return syms
        except Exception as e:
            logger.warning(f"Failed to read CSV: {e}")
    return DEFAULT_TICKERS.copy()

def main():
    tickers = get_tickers_from_csv_or_default()
    tickers = [normalize_ticker(t) for t in tickers]
    # remove obvious placeholders
    tickers = [t for t in tickers if not any(p in t for p in ('DUMMY','TMP','ZZ'))]
    logger.info(f"Tickers count to download: {len(tickers)}")
    bulk, prepared_map, skipped = download_and_prepare_tickers(tickers, period=DATA_PERIOD)
    if not prepared_map:
        logger.error("No data prepared. Exiting.")
        with open(CACHE_FILE, 'w') as f:
            json.dump({"updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                       "prepared": 0, "skipped": skipped, "last_run": datetime.utcnow().strftime("%Y-%m-%d")}, f, indent=2)
        return

    engine = BacktestEngine(prepared_map)
    report = engine.run()
    out = {
        "updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "portfolio": {
            "curve": report.get("curve", [CAPITAL]),
            "ledger": report.get("history", [])[-50:],
            "win_rate": report.get("win_rate", 0),
            "total_trades": report.get("total_trades", 0),
            "profit": report.get("profit", 0)
        },
        "tickers": {k.replace('.NS',''): 0 for k in prepared_map.keys()},
        "skipped": skipped,
        "entry_diag": report.get("entry_diag", {}),
        "last_run": report.get("last_run", datetime.utcnow().strftime("%Y-%m-%d"))
    }
    with open(CACHE_FILE, 'w') as f:
        json.dump(out, f, indent=2)
    logger.info("Backtest finished.")
    logger.info(f"Prepared tickers: {len(prepared_map)} | Skipped: {len(skipped)}")
    logger.info(f"Trades: {out['portfolio']['total_trades']} | Win Rate: {out['portfolio']['win_rate']}% | Profit: {out['portfolio']['profit']}")
    logger.info(f"Saved results to {CACHE_FILE}")

if __name__ == "__main__":
    main()

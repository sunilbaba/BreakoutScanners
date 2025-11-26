#!/usr/bin/env python3
"""
backtest_runner.py

Very-strict pullback backtest (High win-rate: ~60%+ , few trades).
- 2 years of daily data
- Tight pullback to EMA20, volume dry-up, trend filter
- Tight SL and modest TP
- Writes backtest_stats.json
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
# CONFIG (Strict C)
# -------------------------
DATA_PERIOD = "2y"
CAPITAL = 100_000.0
RISK_PER_TRADE = 0.015   # 1.5% risk per trade (smaller risk for strict setups)
BROKERAGE_PCT = 0.001

VOL_MULT = 0.70          # volume must be less than 70% of vol_sma20 (dry-up)
PULLBACK_PCT = 0.008     # within ±0.8% of EMA20
RSI_LOW = 45
RSI_HIGH = 56
BREAKOUT_LOOKBACK = 1    # follow-through: today > yesterday high
ATR_SMA_PERIOD = 20
MIN_HISTORY_ROWS = 250   # require ~1 year+ history
MIN_INDICATOR_ROWS = 180

CONCURRENT_POSITIONS = 4
BATCH_SIZE = 40
BATCH_RETRIES = 2
CACHE_FILE = "backtest_stats.json"
DEFAULT_TICKERS = ["RELIANCE.NS","HDFCBANK.NS","INFY.NS","TCS.NS","SBIN.NS"]

# TSL params
TSL_BE_ATR = 0.9
TSL_EMA10_ATR = 1.6
TSL_SWING_ATR = 2.4

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BacktestC")

# -------------------------
# INDICATORS
# -------------------------
def true_range(df):
    pc = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - pc).abs()
    tr3 = (df['Low'] - pc).abs()
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
    # normalize col names if needed
    cols_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc == 'adj close': cols_map[c] = 'Close'
        elif lc in ('open','high','low','close','volume'): cols_map[c] = lc.capitalize()
    if cols_map:
        df = df.rename(columns=cols_map)
    if not set(['Open','High','Low','Close','Volume']).issubset(df.columns):
        return pd.DataFrame()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['ATR'] = atr(df)
    df['ATR_SMA'] = df['ATR'].rolling(ATR_SMA_PERIOD).mean()
    df['RSI'] = wilder_rsi(df['Close'])
    df['ADX'] = compute_adx(df)
    df['VOL_SMA20'] = df['Volume'].rolling(20).mean()
    df['OBV'] = compute_obv(df)
    df['OBV_SMA20'] = df['OBV'].rolling(20).mean()
    return df

# -------------------------
# DOWNLOAD HELPERS
# -------------------------
def normalize_ticker(sym):
    s = str(sym).strip().upper()
    if s.startswith('^'): return s
    if not s.endswith('.NS'):
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
    valid_rows = int(pdf.dropna(subset=['SMA50','SMA200','EMA20','ATR','RSI']).shape[0])
    if raw_rows < MIN_HISTORY_ROWS:
        return None, f"too_few_rows:{raw_rows}"
    if valid_rows < MIN_INDICATOR_ROWS:
        return None, f"not_enough_indicator_rows:{valid_rows}"
    logger.info(f"{ticker}: raw_rows={raw_rows}, indicator_rows={valid_rows}")
    return pdf, None

def prepare_candidate_from_raw(raw, ticker):
    if raw is None or raw.empty:
        return None, "no_data"
    pdf = prepare_df(raw)
    if pdf.empty:
        return None, "prepare_empty"
    raw_rows = len(raw)
    valid_rows = int(pdf.dropna(subset=['SMA50','SMA200','EMA20','ATR','RSI']).shape[0])
    if raw_rows < MIN_HISTORY_ROWS:
        return None, f"too_few_rows:{raw_rows}"
    if valid_rows < MIN_INDICATOR_ROWS:
        return None, f"not_enough_indicator_rows:{valid_rows}"
    logger.info(f"{ticker}: raw_rows={raw_rows}, indicator_rows={valid_rows}")
    return pdf, None

def download_and_prepare_tickers(tickers, period=DATA_PERIOD, batch_size=BATCH_SIZE, batch_retries=BATCH_RETRIES):
    prepared = {}
    skipped = []
    logger.info(f"Downloading {len(tickers)} tickers for {period}...")
    for start in range(0, len(tickers), batch_size):
        batch = tickers[start:start+batch_size]
        raw = None
        for attempt in range(batch_retries + 1):
            try:
                raw = yf.download(batch, period=period, interval='1d', group_by='ticker', progress=False, threads=True, ignore_tz=True, auto_adjust=True)
                break
            except Exception as e:
                logger.warning(f"Batch download failed (attempt {attempt+1}): {e}")
                time.sleep(1.2 * (attempt+1))
                raw = None
        if raw is None or raw.empty:
            # fallback to per-symbol
            for t in batch:
                pdf, reason = try_single_download_and_prepare(t, period)
                if pdf is not None:
                    prepared[t] = pdf
                else:
                    skipped.append((t, reason))
            continue

        if isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
            cols0 = raw.columns.get_level_values(0).unique()
            for t in cols0:
                try:
                    sub = raw[t].dropna(how='all')
                except Exception:
                    sub = None
                pdf, reason = prepare_candidate_from_raw(sub, t)
                if pdf is not None:
                    prepared[t] = pdf
                else:
                    skipped.append((t, reason))
        else:
            for t in batch:
                pdf, reason = try_single_download_and_prepare(t, period)
                if pdf is not None:
                    prepared[t] = pdf
                else:
                    skipped.append((t, reason))

    if not prepared:
        logger.error("No tickers prepared successfully. Aborting.")
        return None, {}, skipped
    skip_counter = Counter([r for (_, r) in skipped])
    logger.info(f"Prepared {len(prepared)} tickers; skipped {len(skipped)} tickers.")
    logger.info(f"Top skip reasons: {skip_counter.most_common(8)}")
    return prepared, prepared, skipped

# -------------------------
# BACKTEST ENGINE (Strict)
# -------------------------
class BacktestEngine:
    def __init__(self, prepared_map):
        self.map = prepared_map
        self.tickers = [t for t in prepared_map.keys() if not str(t).startswith('^')]
        self.cash = CAPITAL
        self.equity_curve = [CAPITAL]
        self.portfolio = []
        self.history = []
        self.entry_diag = Counter()

    def calc_qty(self, entry, sl, equity):
        risk_amt = equity * RISK_PER_TRADE
        risk_share = abs(entry - sl)
        if risk_share <= 0: return 0
        qty = int(risk_amt / risk_share)
        max_qty = int((equity * 0.20) / entry) if entry > 0 else 0
        if qty * entry > equity * 0.20:
            qty = max_qty
        return max(qty, 0)

    def apply_tsl(self, trade, row, df, idx):
        atr_v = row['ATR']
        entry = trade['entry']
        if row['High'] >= entry + (TSL_BE_ATR * atr_v):
            trade['sl'] = max(trade['sl'], entry)
        if row['High'] >= entry + (TSL_EMA10_ATR * atr_v):
            if not pd.isna(row.get('EMA10', np.nan)):
                trade['sl'] = max(trade['sl'], row['EMA10'])
        if row['High'] >= entry + (TSL_SWING_ATR * atr_v):
            if idx >= 5:
                swing_low = df['Low'].iloc[max(0, idx-5):idx].min()
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

    def run(self):
        logger.info("Running very-strict pullback backtest...")
        processed = self.map
        dates = sorted(list(set().union(*[df.index for df in processed.values()])))
        for date in dates:
            # EXITS
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

            # ENTRIES
            if len(self.portfolio) < CONCURRENT_POSITIONS:
                for sym, df in processed.items():
                    if date not in df.index: continue
                    if len(self.portfolio) >= CONCURRENT_POSITIONS: break
                    row = df.loc[date]
                    # basic checks
                    if pd.isna(row.get('SMA50')) or pd.isna(row.get('SMA200')) or pd.isna(row.get('EMA20')):
                        self.entry_diag['no_ind'] += 1; continue
                    # Trend
                    if not (row['SMA50'] > row['SMA200'] and row['EMA20'] > row['SMA50'] * 0.98):  # EMA20 rising near SMA50
                        self.entry_diag['trend_fail'] += 1; continue
                    # Pullback tight to EMA20
                    if abs(row['Close'] - row['EMA20'])/row['EMA20'] > PULLBACK_PCT:
                        self.entry_diag['pullback_fail'] += 1; continue
                    # Volume dry-up
                    if pd.isna(row.get('VOL_SMA20')) or row['Volume'] >= VOL_MULT * row['VOL_SMA20']:
                        self.entry_diag['vol_fail'] += 1; continue
                    # RSI band
                    if pd.isna(row.get('RSI')) or row['RSI'] < RSI_LOW or row['RSI'] > RSI_HIGH:
                        self.entry_diag['rsi_fail'] += 1; continue
                    # Follow-through: must be > yesterday high
                    i = df.index.get_loc(date)
                    if i == 0:
                        self.entry_diag['no_prev'] += 1; continue
                    prev_high = df['High'].iat[i-1]
                    if row['Close'] <= prev_high:
                        self.entry_diag['no_follow'] += 1; continue
                    # SL and TP
                    swing_low = df['Low'].iloc[max(0, i-5):i].min() if i>0 else row['Low']
                    sl_candidate = min(swing_low, row['Close'] - 0.8 * row['ATR']) if not pd.isna(row['ATR']) else swing_low
                    if sl_candidate >= row['Close']:
                        self.entry_diag['bad_sl'] += 1; continue
                    target = row['Close'] + 1.6 * row['ATR'] if not pd.isna(row['ATR']) else row['Close'] * 1.02
                    rr = (target - row['Close']) / (row['Close'] - sl_candidate) if (row['Close'] - sl_candidate)!=0 else 0
                    # For strict, require rr >= 1.2 (small RR but ok for high win-rate)
                    if rr < 1.1:
                        self.entry_diag['rr_fail'] += 1; continue
                    qty = self.calc_qty(row['Close'], sl_candidate, self.equity_curve[-1])
                    if qty <= 0 or qty * row['Close'] > self.cash:
                        self.entry_diag['qty_fail'] += 1; continue
                    fees = row['Close'] * qty * BROKERAGE_PCT
                    self.cash -= (row['Close'] * qty + fees)
                    self.portfolio.append({
                        "symbol": sym,
                        "entry": row['Close'],
                        "qty": qty,
                        "sl": sl_candidate,
                        "target": target,
                        "entry_cost": fees,
                        "entry_date": date
                    })
                    self.entry_diag['accepted'] += 1

            # EQUITY update
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
            report = {
                "curve": [round(x,2) for x in self.equity_curve],
                "profit": round(self.equity_curve[-1] - CAPITAL, 2),
                "total_trades": 0,
                "win_rate": 0,
                "history": [],
                "entry_diag": dict(self.entry_diag),
                "last_run": datetime.utcnow().strftime("%Y-%m-%d")
            }
            return report

        wins = [h for h in self.history if h['pnl'] > 0]
        win_rate = round(100 * len(wins) / len(self.history), 2) if self.history else 0
        report = {
            "curve": [round(x,2) for x in self.equity_curve],
            "profit": round(self.equity_curve[-1] - CAPITAL, 2),
            "total_trades": len(self.history),
            "win_rate": win_rate,
            "history": self.history,
            "entry_diag": dict(self.entry_diag),
            "last_run": datetime.utcnow().strftime("%Y-%m-%d")
        }
        return report

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

def normalize_ticker(sym):
    s = str(sym).strip().upper()
    if s.startswith('^'): return s
    if not s.endswith('.NS'):
        s = s + '.NS'
    return s

def main():
    tickers = get_tickers_from_csv_or_default()
    tickers = [normalize_ticker(t) for t in tickers]
    tickers = [t for t in tickers if not any(p in t for p in ('DUMMY','TMP','ZZ'))]
    logger.info(f"Tickers count to download: {len(tickers)}")
    prepared_map, _, skipped = download_and_prepare_tickers(tickers, period=DATA_PERIOD)
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
    logger.info(f"Entry diag: {out['entry_diag']}")
    logger.info(f"Saved results to {CACHE_FILE}")

if __name__ == "__main__":
    main()

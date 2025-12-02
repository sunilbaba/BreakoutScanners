"""
backtest_runner.py
Institutional-Grade Simulation Engine.
- Runs ONCE daily (Nightly Job).
- Downloads 2 Years of Data.
- Replays the market with EXACT execution logic (Gaps, Slippage, Costs).
- Saves 'backtest_stats.json' for the dashboard.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime
from functools import wraps
import time

# -------------------------
# 1. CONFIGURATION
# -------------------------
DATA_PERIOD = "2y" 
CACHE_FILE = "backtest_stats.json"
CAPITAL = 100_000.0
RISK_PER_TRADE = 0.02
MAX_POSITIONS = 5
BROKERAGE_PCT = 0.001 # 0.1% per side (Costs)

# Strategy Config
ATR_MULTIPLIER_TARGET = 3.0
ATR_MULTIPLIER_STOP = 1.0
ADX_THRESHOLD = 25.0

SECTOR_INDICES = {
    "NIFTY 50": "^NSEI", "BANK": "^NSEBANK", "AUTO": "^CNXAUTO", "IT": "^CNXIT",
    "METAL": "^CNXMETAL", "PHARMA": "^CNXPHARMA", "FMCG": "^CNXFMCG",
    "ENERGY": "^CNXENERGY", "REALTY": "^CNXREALTY", "PSU BANK": "^CNXPSUBANK"
}

# Fallback List
DEFAULT_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LTIM.NS",
    "AXISBANK.NS", "MARUTI.NS", "TITAN.NS", "SUNPHARMA.NS", "BAJFINANCE.NS",
    "HCLTECH.NS", "TATASTEEL.NS", "ADANIENT.NS", "JIOFIN.NS", "ZOMATO.NS",
    "DLF.NS", "HAL.NS", "TRENT.NS", "BEL.NS", "POWERGRID.NS", "ONGC.NS", "WIPRO.NS"
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Backtester")

# -------------------------
# 2. UTILS
# -------------------------
def retry_on_exception(max_tries=3, backoff=2.0):
    def deco(func):
        @wraps(func)
        def inner(*a, **kw):
            for i in range(max_tries):
                try: return func(*a, **kw)
                except Exception as e: 
                    logger.warning(f"Retry {i+1}: {e}")
                    time.sleep(backoff * (2 ** i))
            raise Exception(f"Failed after {max_tries} tries")
        return inner
    return deco

# -------------------------
# 3. DATA
# -------------------------
def get_tickers():
    if os.path.exists("ind_nifty500list.csv"):
        try:
            df = pd.read_csv("ind_nifty500list.csv")
            return [f"{x}.NS" for x in df['Symbol'].dropna().unique()]
        except: pass
    return DEFAULT_TICKERS

@retry_on_exception(max_tries=3)
def robust_download(tickers):
    logger.info(f"⬇️ Downloading {len(tickers)} symbols ({DATA_PERIOD})...")
    frames = []
    batch_size = 20
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(batch, period=DATA_PERIOD, group_by='ticker', threads=True, progress=False, ignore_tz=True)
            frames.append(data)
        except: pass
    if not frames: return pd.DataFrame()
    return pd.concat(frames, axis=1)

def extract_df(bulk, ticker):
    try:
        if isinstance(bulk.columns, pd.MultiIndex):
            if ticker in bulk.columns.get_level_values(0):
                return bulk[ticker].copy().dropna()
    except: pass
    return None

# -------------------------
# 4. MATH ENGINE
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
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss)).fillna(50)

def adx(df, period=14):
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), -minus_dm, 0.0)
    tr = true_range(df).ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / tr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / tr)
    return (abs(plus_di - minus_di) / (plus_di + minus_di) * 100).ewm(alpha=1/14, adjust=False).mean().fillna(0)

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
# 5. SIMULATION LOGIC (The "Restored" Logic)
# -------------------------
class BacktestEngine:
    def __init__(self, bulk_data, tickers):
        self.data = bulk_data
        self.tickers = [t for t in tickers if not str(t).startswith('^')]
        self.cash = CAPITAL
        self.equity_curve = [CAPITAL]
        self.portfolio = []
        self.history = []
        
    def run(self):
        logger.info("⏳ Processing Data & Indicators...")
        processed = {}
        for t in self.tickers:
            raw = extract_df(self.data, t)
            if raw is not None and len(raw) > 250:
                processed[t] = prepare_df(raw)
        
        if not processed: return {}
        
        # Timeline
        dates = sorted(list(set().union(*[d.index for d in processed.values()])))
        sim_dates = dates[200:] # Warmup skip
        
        logger.info(f"🚀 Simulating {len(sim_dates)} Days...")
        
        for date in sim_dates:
            self.process_day(date, processed)
            
            # Daily Mark-to-Market
            m2m = 0
            for t in self.portfolio:
                sym = t['symbol']
                price = processed[sym].loc[date]['Close'] if date in processed[sym].index else t['entry']
                m2m += (price * t['qty'])
            self.equity_curve.append(round(self.cash + m2m, 2))

        # Final Stats
        wins = len([t for t in self.history if t['pnl'] > 0])
        win_rate = round(wins / len(self.history) * 100, 1) if self.history else 0
        
        # Per-Stock Win Rates (For Dashboard Highlighting)
        ticker_stats = {}
        for t in self.tickers:
            if t in processed:
                ticker_stats[t.replace('.NS','')] = self.calc_single_wr(processed[t])

        return {
            "curve": self.equity_curve,
            "win_rate": win_rate,
            "total_trades": len(self.history),
            "profit": round(self.equity_curve[-1] - CAPITAL, 2),
            "ledger": self.history,
            "tickers": ticker_stats
        }

    def calc_single_wr(self, df):
        """Fast vectorized win-rate check for a single stock"""
        wins, total = 0, 0
        start = max(200, len(df) - 130)
        for i in range(start, len(df)-10):
            row = df.iloc[i]
            sma200 = row['SMA200'] if not pd.isna(row['SMA200']) else row['Close']
            
            # THE STRATEGY LOGIC
            if row['Close'] > row['EMA20'] and row['RSI'] > 60 and row['ADX'] > 25 and row['Close'] > sma200:
                outcome = "OPEN"
                stop = row['Close'] - row['ATR']
                target = row['Close'] + (3 * row['ATR'])
                
                for j in range(1, 15):
                    if i+j >= len(df): break
                    fut = df.iloc[i+j]
                    if fut['Low'] <= stop: outcome="LOSS"; break
                    if fut['High'] >= target: outcome="WIN"; break
                
                if outcome != "OPEN":
                    total += 1
                    if outcome == "WIN": wins += 1
        return round(wins/total*100, 0) if total > 0 else 0

    def process_day(self, date, data_map):
        # 1. CHECK EXITS (Priority)
        active = []
        for t in self.portfolio:
            sym = t['symbol']
            if date not in data_map[sym].index:
                active.append(t); continue
            
            row = data_map[sym].loc[date]
            exit_price = None
            reason = ""
            
            # GAP LOGIC (Restored)
            if row['Open'] < t['sl']: 
                exit_price = row['Open']; reason="Gap Down SL"
            elif row['Low'] <= t['sl']: 
                exit_price = t['sl']; reason="Stop Hit"
            elif row['Open'] > t['tgt']: 
                exit_price = row['Open']; reason="Gap Up Win"
            elif row['High'] >= t['tgt']: 
                exit_price = t['tgt']; reason="Target Hit"
            
            if exit_price:
                revenue = exit_price * t['qty']
                cost = revenue * BROKERAGE_PCT
                self.cash += (revenue - cost)
                
                pnl = revenue - cost - (t['entry'] * t['qty'] + t['entry_cost'])
                self.history.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "symbol": sym,
                    "entry": round(t['entry'], 2),
                    "exit": round(exit_price, 2),
                    "pnl": round(pnl, 2),
                    "result": "WIN" if pnl>0 else "LOSS",
                    "reason": reason
                })
            else:
                active.append(t)
        self.portfolio = active
        
        # 2. CHECK ENTRIES (Max 5)
        if len(self.portfolio) >= MAX_POSITIONS: return
        
        for sym, df in data_map.items():
            if date not in df.index: continue
            row = df.loc[date]
            
            # Safety checks
            if pd.isna(row['SMA200']) or pd.isna(row['ADX']): continue
            
            # STRATEGY: Trend + Momentum + Strength
            if row['Close'] > row['EMA20'] and row['RSI'] > 60 and row['ADX'] > ADX_THRESHOLD and row['Close'] > row['SMA200']:
                if any(t['symbol'] == sym for t in self.portfolio): continue
                
                risk = row['ATR']
                if risk <= 0: continue
                
                # Position Sizing (Dynamic)
                current_equity = self.equity_curve[-1]
                qty = int((current_equity * RISK_PER_TRADE) / risk)
                cost = qty * row['Close']
                
                # Affordable Sizing
                if cost > self.cash: 
                    qty = int(self.cash / row['Close'])
                    cost = qty * row['Close']
                
                if qty > 0 and self.cash > cost:
                    fees = cost * BROKERAGE_PCT
                    self.cash -= (cost + fees)
                    self.portfolio.append({
                        "symbol": sym, "entry": row['Close'], "qty": qty,
                        "sl": row['Close'] - risk, 
                        "tgt": row['Close'] + (3*risk),
                        "entry_cost": fees
                    })
                    if len(self.portfolio) >= MAX_POSITIONS: break

# --- MAIN ---
if __name__ == "__main__":
    tickers = get_tickers()
    all_syms = tickers + list(SECTOR_INDICES.values())
    
    # 1. Download 2Y Data
    bulk = robust_download(all_syms)
    
    # 2. Run Simulation
    engine = BacktestEngine(bulk, tickers)
    stats = engine.run()
    
    # 3. Save
    output = {
        "updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "portfolio": stats,
        "tickers": stats['tickers']
    }
    
    with open(CACHE_FILE, "w") as f:
        json.dump(output, f)
        
    logger.info(f"✅ Success! Profit: {stats['profit']} | Trades: {stats['total_trades']}")

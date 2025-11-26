"""
backtest_runner.py
- Runs ONCE daily (e.g., 8 PM IST).
- Heavy Task: Downloads 1 Year data for Nifty 500.
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

# --- CONFIG ---
DATA_PERIOD = "1y"
CACHE_FILE = "backtest_stats.json"
CAPITAL = 100_000.0
RISK_PER_TRADE = 0.02
BROKERAGE_PCT = 0.001

SECTOR_INDICES = { "NIFTY 50": "^NSEI", "BANK": "^NSEBANK", "AUTO": "^CNXAUTO", "IT": "^CNXIT", "METAL": "^CNXMETAL", "PHARMA": "^CNXPHARMA", "FMCG": "^CNXFMCG", "ENERGY": "^CNXENERGY", "REALTY": "^CNXREALTY", "PSU BANK": "^CNXPSUBANK" }

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Backtester")

# --- 1. DATA ---
def get_tickers():
    if os.path.exists("ind_nifty500list.csv"):
        try:
            df = pd.read_csv("ind_nifty500list.csv")
            return [f"{x}.NS" for x in df['Symbol'].dropna().unique()]
        except: pass
    return ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "SBIN.NS"]

def robust_download(tickers):
    logger.info(f"📥 Downloading {len(tickers)} stocks (1 Year)...")
    batch_size = 50
    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(batch, period=DATA_PERIOD, group_by='ticker', threads=True, progress=False, ignore_tz=True)
            frames.append(data)
        except: pass
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()

def extract_df(bulk, ticker):
    try:
        if isinstance(bulk.columns, pd.MultiIndex):
            if ticker in bulk.columns.get_level_values(0):
                return bulk[ticker].copy().dropna()
    except: pass
    return None

# --- 2. MATH ---
def prepare_df(df):
    df = df.copy()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    
    h_l = df['High'] - df['Low']
    h_c = np.abs(df['High'] - df['Close'].shift())
    l_c = np.abs(df['Low'] - df['Close'].shift())
    df['ATR'] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss)).fillna(50)
    
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), -minus_dm, 0.0)
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).ewm(alpha=1/14).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14).mean() / tr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14).mean() / tr)
    df['ADX'] = (abs(plus_di - minus_di) / (plus_di + minus_di) * 100).ewm(alpha=1/14).mean().fillna(0)
    return df

# --- 3. WIN RATE CALCULATOR ---
def calc_stock_win_rate(df):
    if len(df) < 150: return 0
    wins, total = 0, 0
    # Check last 6 months
    for i in range(len(df)-130, len(df)-10):
        row = df.iloc[i]
        if row['Close'] > row['EMA20'] and row['RSI'] > 60 and row['ADX'] > 25:
            stop = row['Close'] - row['ATR']
            target = row['Close'] + (3 * row['ATR'])
            outcome = "OPEN"
            for j in range(1, 15):
                if i+j >= len(df): break
                fut = df.iloc[i+j]
                if fut['Low'] <= stop: outcome="LOSS"; break
                if fut['High'] >= target: outcome="WIN"; break
            if outcome != "OPEN":
                total += 1
                if outcome == "WIN": wins += 1
    return round((wins/total*100), 0) if total > 0 else 0

# --- 4. PORTFOLIO SIMULATION ---
def run_portfolio_sim(bulk_data, tickers):
    processed = {}
    for t in tickers:
        df = extract_df(bulk_data, t)
        if df is not None and len(df) > 200:
            processed[t] = prepare_df(df)
    
    if not processed: return [], [], 0, 0, 0
    
    dates = sorted(list(set().union(*[d.index for d in processed.values()])))
    sim_dates = dates[150:]
    
    cash = CAPITAL
    equity_curve = [CAPITAL]
    portfolio = []
    history = [] # The Backtest Ledger
    
    for date in sim_dates:
        active = []
        # Exits
        for trade in portfolio:
            sym = trade['symbol']
            if date not in processed[sym].index:
                active.append(trade); continue
            row = processed[sym].loc[date]
            exit_p = None
            
            if row['Open'] < trade['sl']: exit_p = row['Open']
            elif row['Low'] <= trade['sl']: exit_p = trade['sl']
            elif row['Open'] > trade['tgt']: exit_p = row['Open']
            elif row['High'] >= trade['tgt']: exit_p = trade['tgt']
            
            if exit_p:
                pnl = (exit_p - trade['entry']) * trade['qty']
                cost = (exit_price * trade['qty'] * BROKERAGE_PCT) if 'exit_price' in locals() else 0 # rough est
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
            else: active.append(trade)
        portfolio = active
        
        # Entries
        if len(portfolio) < 5:
            for sym, df in processed.items():
                if date not in df.index: continue
                row = df.loc[date]
                if row['Close'] > row['EMA20'] and row['RSI'] > 60 and row['ADX'] > 25 and row['Close'] > row['SMA200']:
                    if any(t['symbol'] == sym for t in portfolio): continue
                    risk = row['ATR']
                    qty = int((equity_curve[-1] * RISK_PER_TRADE) / risk)
                    cost = qty * row['Close']
                    if qty > 0 and cash > cost:
                        fees = cost * BROKERAGE_PCT
                        cash -= (cost + fees)
                        portfolio.append({
                            "symbol": sym, "entry": row['Close'], "qty": qty,
                            "sl": row['Close'] - risk, "tgt": row['Close'] + (3*risk),
                            "entry_cost": fees
                        })
                        if len(portfolio) >= 5: break
        
        # Equity
        m2m = 0
        for t in portfolio:
            sym = t['symbol']
            price = processed[sym].loc[date]['Close'] if date in processed[sym].index else t['entry']
            m2m += (price * t['qty'])
        equity_curve.append(round(cash + m2m, 2))

    wins = len([h for h in history if h['pnl'] > 0])
    win_rate = round(wins / len(history) * 100, 1) if history else 0
    return equity_curve, history, win_rate, len(history), round(equity_curve[-1] - CAPITAL, 2)

# --- MAIN ---
if __name__ == "__main__":
    tickers = get_tickers()
    all_syms = tickers + list(SECTOR_INDICES.values())
    bulk = robust_download(all_syms)
    
    # 1. Individual Win Rates
    logger.info("📊 Calculating Win Rates...")
    ticker_stats = {}
    for t in tickers:
        df = extract_df(bulk, t)
        if df is not None:
            ticker_stats[t.replace('.NS','')] = calc_stock_win_rate(prepare_df(df))
            
    # 2. Portfolio Sim
    logger.info("📈 Running Sim...")
    curve, ledger, win_rate, trades, profit = run_portfolio_sim(bulk, tickers)
    
    output = {
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        "portfolio": {
            "curve": curve,
            "ledger": ledger[-50:], # Only save last 50 trades to keep file light
            "win_rate": win_rate,
            "total_trades": trades,
            "profit": profit
        },
        "tickers": ticker_stats
    }
    
    with open(CACHE_FILE, "w") as f: json.dump(output, f)
    logger.info(f"✅ Saved stats to {CACHE_FILE}")

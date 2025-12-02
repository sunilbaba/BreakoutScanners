"""
backtest_runner.py (Tuned Edition)
- Relaxed constraints to find more trades.
- ADX > 15 (was 20).
- RSI > 50 (was 55).
- Fixed hardcoded values in Win Rate calculator.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import logging
from collections import Counter
from datetime import datetime

# --- CONFIG ---
DATA_PERIOD = "2y" 
CACHE_FILE = "backtest_stats.json"
CAPITAL = 100_000.0
RISK_PER_TRADE = 0.02
BROKERAGE_PCT = 0.001

# --- RELAXED SETTINGS FOR MORE TRADES ---
ADX_THRESHOLD = 15.0  # Lowered from 20 to catch earlier trends
RSI_THRESHOLD = 50.0  # Lowered from 55 to catch standard momentum

SECTOR_INDICES = {
    "NIFTY 50": "^NSEI", "BANK": "^NSEBANK", "AUTO": "^CNXAUTO", "IT": "^CNXIT",
    "METAL": "^CNXMETAL", "PHARMA": "^CNXPHARMA", "FMCG": "^CNXFMCG",
    "ENERGY": "^CNXENERGY", "REALTY": "^CNXREALTY", "PSU BANK": "^CNXPSUBANK"
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Backtest")

# --- 1. DATA ---
def get_tickers():
    if os.path.exists("ind_nifty500list.csv"):
        try:
            df = pd.read_csv("ind_nifty500list.csv")
            return [f"{x}.NS" for x in df['Symbol'].dropna().unique()]
        except: pass
    return ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "ICICIBANK.NS", 
            "SBIN.NS", "TATAMOTORS.NS", "ITC.NS", "SUNPHARMA.NS", "MARUTI.NS"]

def robust_download(tickers):
    logger.info(f"⬇️ Downloading {len(tickers)} symbols...")
    frames = []
    batch_size = 20
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
    
    # Volatility
    h_l = df['High'] - df['Low']
    h_c = (df['High'] - df['Close'].shift()).abs()
    l_c = (df['Low'] - df['Close'].shift()).abs()
    df['ATR'] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    
    # Momentum
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss)).fillna(50)
    
    # Trend Strength (ADX)
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), -minus_dm, 0.0)
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).ewm(alpha=1/14).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14).mean() / tr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14).mean() / tr)
    df['ADX'] = (abs(plus_di - minus_di) / (plus_di + minus_di) * 100).ewm(alpha=1/14).mean().fillna(0)
    
    return df

# --- 3. WIN RATE CALC ---
def calc_single_wr(df):
    if len(df) < 250: return 0
    wins, total = 0, 0
    start = max(200, len(df) - 130)
    for i in range(start, len(df)-10):
        row = df.iloc[i]
        sma200 = row['SMA200'] if not pd.isna(row['SMA200']) else row['Close']
        
        # Use GLOBAL thresholds to match simulation
        if row['Close'] > row['EMA20'] and row['RSI'] > RSI_THRESHOLD and row['ADX'] > ADX_THRESHOLD and row['Close'] > sma200:
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

# --- 4. SIMULATION ---
def run_simulation(bulk_data, tickers):
    processed = {}
    for t in tickers:
        raw = extract_df(bulk_data, t)
        if raw is not None and len(raw) > 250:
            processed[t] = prepare_df(raw)
            
    if not processed: return [], [], 0, 0, 0
    
    dates = sorted(list(set().union(*[d.index for d in processed.values()])))
    sim_dates = dates[200:] # Warmup
    
    cash = CAPITAL
    curve = [CAPITAL]
    portfolio = []
    history = []
    
    # DIAGNOSTICS
    reject_reasons = Counter()
    potential_signals = 0
    
    logger.info(f"🚀 Simulating {len(sim_dates)} days across {len(processed)} stocks...")
    
    for date in sim_dates:
        # A. Exits
        active = []
        for t in portfolio:
            sym = t['symbol']
            if date not in processed[sym].index:
                active.append(t); continue
            
            row = processed[sym].loc[date]
            exit_p = None
            
            if row['Open'] < t['sl']: exit_p = row['Open']
            elif row['Low'] <= t['sl']: exit_p = t['sl']
            elif row['High'] >= t['tgt']: exit_p = t['tgt']
            
            if exit_p:
                pnl = (exit_p - t['entry']) * t['qty']
                cash += (exit_p * t['qty'])
                history.append({"date": date.strftime("%Y-%m-%d"), "symbol": sym, "pnl": round(pnl, 2), "result": "WIN" if pnl>0 else "LOSS"})
            else: active.append(t)
        portfolio = active
        
        # B. Entries
        if len(portfolio) < 5:
            for sym, df in processed.items():
                if date not in df.index: continue
                row = df.loc[date]
                
                # RULES
                sma200 = row['SMA200'] if not pd.isna(row['SMA200']) else row['Close']
                
                passed_trend = row['Close'] > sma200
                passed_mom = row['Close'] > row['EMA20']
                passed_rsi = row['RSI'] > RSI_THRESHOLD
                passed_adx = row['ADX'] > ADX_THRESHOLD
                
                if not passed_trend: reject_reasons['Downtrend'] += 1
                elif not passed_mom: reject_reasons['Weak Momentum'] += 1
                elif not passed_rsi: reject_reasons['Low RSI'] += 1
                elif not passed_adx: reject_reasons['Low ADX'] += 1
                else:
                    potential_signals += 1
                    if any(t['symbol'] == sym for t in portfolio): 
                        reject_reasons['Already Owned'] += 1
                        continue
                        
                    risk = row['ATR']
                    if risk <= 0: continue
                    
                    qty = int((curve[-1] * RISK_PER_TRADE) / risk)
                    cost = qty * row['Close']
                    
                    # Affordable sizing
                    if cost > cash: 
                        qty = int(cash / row['Close'])
                        cost = qty * row['Close']
                    
                    if qty > 0 and cash > cost:
                        cash -= cost
                        portfolio.append({
                            "symbol": sym, "entry": row['Close'], "qty": qty,
                            "sl": row['Close'] - risk, "tgt": row['Close'] + (3*risk)
                        })
                        if len(portfolio) >= 5: break
                    else:
                        reject_reasons['Insufficient Cash'] += 1
        
        # C. Equity
        m2m = 0
        for t in portfolio:
            sym = t['symbol']
            price = processed[sym].loc[date]['Close'] if date in processed[sym].index else t['entry']
            m2m += (price * t['qty'])
        curve.append(round(cash + m2m, 2))

    # REPORT DIAGNOSTICS
    logger.info("\n--- DIAGNOSTIC REPORT ---")
    logger.info(f"Total Potential Signals: {potential_signals}")
    logger.info("Rejection Reasons:")
    for reason, count in reject_reasons.most_common():
        logger.info(f"  - {reason}: {count}")
    logger.info(f"Actual Trades Taken: {len(history)}")
    logger.info("-------------------------\n")

    wins = len([h for h in history if h['pnl'] > 0])
    win_rate = round(wins / len(history) * 100, 1) if history else 0
    return curve, history, win_rate, len(history), round(curve[-1] - CAPITAL, 2)

# --- MAIN ---
if __name__ == "__main__":
    tickers = get_tickers()
    bulk = robust_download(tickers + list(SECTOR_INDICES.values()))
    
    ticker_stats = {}
    for t in tickers:
        df = extract_df(bulk, t)
        if df is not None:
            ticker_stats[t.replace('.NS','')] = calc_single_wr(prepare_df(df))
            
    curve, ledger, win_rate, trades, profit = run_simulation(bulk, tickers)
    
    output = {
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        "portfolio": {
            "curve": curve, "ledger": ledger, 
            "win_rate": win_rate, "total_trades": trades, "profit": profit
        },
        "tickers": ticker_stats
    }
    
    with open(CACHE_FILE, "w") as f: json.dump(output, f)
    logger.info(f"✅ Saved stats. Profit: {profit} | Trades: {trades}")

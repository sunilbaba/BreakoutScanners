"""
strategy_optimizer.py
The "Generative AI" Engine for Trading.
---------------------------------------
1. Downloads 2 Years of Data.
2. "Breeds" 500 random trading strategies.
3. Tests each one against history.
4. PRINTS the top 3 detailed strategies to the console.
5. SAVES the best strategy to 'strategy_config.json' for the trading bot.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import random
import json
import logging
import os
from datetime import datetime

# --- CONFIGURATION ---
ITERATIONS = 500          # Population size (Higher = Better results, Slower)
MIN_TRADES = 30           # Statistical significance threshold
DATA_PERIOD = "2y"

# Fallback Universe
DEFAULT_TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "ICICIBANK.NS", "SBIN.NS",
    "ITC.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LTIM.NS", "AXISBANK.NS", "MARUTI.NS",
    "TITAN.NS", "SUNPHARMA.NS", "BAJFINANCE.NS", "HCLTECH.NS", "TATASTEEL.NS",
    "ADANIENT.NS", "JIOFIN.NS", "ZOMATO.NS", "DLF.NS", "HAL.NS", "TRENT.NS"
]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Optimizer")

# -------------------------
# 1. DATA ENGINE
# -------------------------
def get_tickers():
    if os.path.exists("ind_nifty500list.csv"):
        try:
            df = pd.read_csv("ind_nifty500list.csv")
            return [f"{x}.NS" for x in df['Symbol'].dropna().unique()]
        except: pass
    return DEFAULT_TICKERS

def robust_download(tickers):
    logger.info(f"📥 Downloading {len(tickers)} stocks ({DATA_PERIOD})...")
    frames = []
    batch_size = 50
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

# -------------------------
# 2. FEATURE ENGINEERING
# -------------------------
def prepare_features(df):
    df = df.copy()
    # Moving Averages
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['EMA10'] = df['Close'].ewm(span=10).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    
    # ATR
    h_l = df['High'] - df['Low']
    h_c = (df['High'] - df['Close'].shift()).abs()
    l_c = (df['Low'] - df['Close'].shift()).abs()
    df['ATR'] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss)).fillna(50)
    
    # ADX
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), -minus_dm, 0.0)
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).ewm(alpha=1/14).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14).mean() / tr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14).mean() / tr)
    df['ADX'] = (abs(plus_di - minus_di) / (plus_di + minus_di) * 100).ewm(alpha=1/14).mean().fillna(0)
    
    return df.dropna()

# -------------------------
# 3. STRATEGY GENERATOR
# -------------------------
def generate_random_strategy():
    return {
        "name": f"Gen-AI-{random.randint(1000,9999)}",
        # Entry Logic
        "trend_filter": random.choice(["SMA200", "SMA50", "NONE"]),
        "rsi_logic": random.choice(["OVERSOLD", "OVERBOUGHT", "NEUTRAL"]),
        "rsi_threshold": random.choice([30, 35, 40, 45, 50, 55, 60, 65, 70]),
        "adx_min": random.choice([0, 15, 20, 25]),
        # Exit Logic
        "sl_atr_mult": random.choice([1.0, 1.5, 2.0, 2.5]),
        "tgt_atr_mult": random.choice([2.0, 2.5, 3.0, 4.0, 5.0])
    }

def evaluate_strategy(df, genome):
    wins, losses = 0, 0
    
    # Pre-fetch columns
    close = df['Close']
    sma200 = df['SMA200']
    sma50 = df['SMA50']
    rsi = df['RSI']
    adx = df['ADX']
    atr = df['ATR']
    
    # Simulation (Last 1 Year)
    start_idx = max(200, len(df) - 250)
    
    for i in range(start_idx, len(df) - 20):
        row_close = close.iloc[i]
        
        # 1. Trend
        trend_ok = True
        if genome["trend_filter"] == "SMA200" and row_close < sma200.iloc[i]: trend_ok = False
        if genome["trend_filter"] == "SMA50" and row_close < sma50.iloc[i]: trend_ok = False
        
        # 2. RSI
        row_rsi = rsi.iloc[i]
        rsi_ok = False
        if genome["rsi_logic"] == "OVERSOLD" and row_rsi < genome["rsi_threshold"]: rsi_ok = True
        elif genome["rsi_logic"] == "OVERBOUGHT" and row_rsi > genome["rsi_threshold"]: rsi_ok = True
        elif genome["rsi_logic"] == "NEUTRAL" and row_rsi > 45 and row_rsi < 65: rsi_ok = True
        
        # 3. ADX
        adx_ok = adx.iloc[i] > genome["adx_min"]
        
        if trend_ok and rsi_ok and adx_ok:
            entry = row_close
            sl = entry - (genome["sl_atr_mult"] * atr.iloc[i])
            tgt = entry + (genome["tgt_atr_mult"] * atr.iloc[i])
            
            outcome = "OPEN"
            for j in range(1, 20):
                if i+j >= len(df): break
                curr_low = df['Low'].iloc[i+j]
                curr_high = df['High'].iloc[i+j]
                
                if curr_low <= sl: outcome = "LOSS"; break
                if curr_high >= tgt: outcome = "WIN"; break
            
            if outcome == "WIN": wins += 1
            elif outcome == "LOSS": losses += 1
            
            i += j # Skip overlap

    return wins, losses

# -------------------------
# 4. EVOLUTION ENGINE
# -------------------------
def save_best_strategy(best_genome):
    config = {
        "updated": datetime.utcnow().strftime("%Y-%m-%d"),
        "strategy_name": best_genome['name'],
        "parameters": {
            "trend_filter": best_genome['trend_filter'],
            "rsi_threshold": best_genome['rsi_threshold'],
            "rsi_logic": best_genome['rsi_logic'],
            "adx_threshold": best_genome['adx_min'],
            "atr_stop_mult": best_genome['sl_atr_mult'],
            "atr_target_mult": best_genome['tgt_atr_mult']
        }
    }
    with open("strategy_config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info("✅ Strategy Logic saved to 'strategy_config.json'")

def run_optimization():
    tickers = get_tickers()
    bulk = robust_download(tickers)
    
    processed_data = []
    for t in tickers:
        raw = extract_df(bulk, t)
        if raw is not None and len(raw) > 250:
            processed_data.append(prepare_features(raw))
    
    if not processed_data:
        logger.error("No valid data found.")
        return

    logger.info(f"🧬 Evolving {ITERATIONS} Strategies on {len(processed_data)} stocks...")
    
    results = []
    for _ in range(ITERATIONS):
        genome = generate_random_strategy()
        total_wins, total_losses = 0, 0
        
        for df in processed_data:
            w, l = evaluate_strategy(df, genome)
            total_wins += w
            total_losses += l
            
        total = total_wins + total_losses
        win_rate = (total_wins / total * 100) if total > 0 else 0
        
        if total > MIN_TRADES:
            results.append({
                "genome": genome,
                "win_rate": round(win_rate, 2),
                "trades": total
            })
    
    results.sort(key=lambda x: x['win_rate'], reverse=True)
    
    # --- PRINTING THE RESULTS (RESTORED) ---
    logger.info("\n" + "="*40)
    logger.info(f"🏆 TOP 3 STRATEGIES (Surviving out of {ITERATIONS})")
    logger.info("="*40)
    
    for i, res in enumerate(results[:3]):
        g = res['genome']
        logger.info(f"\n🥉 RANK #{i+1} | Win Rate: {res['win_rate']}% | Trades: {res['trades']}")
        logger.info(f"   • Trend Filter : {g['trend_filter']}")
        logger.info(f"   • Entry Signal : RSI {g['rsi_logic']} ({g['rsi_threshold']})")
        logger.info(f"   • Trend Strength: ADX > {g['adx_min']}")
        logger.info(f"   • Exit Rules   : Stop {g['sl_atr_mult']}x ATR | Target {g['tgt_atr_mult']}x ATR")
    
    logger.info("\n" + "="*40)

    if results:
        save_best_strategy(results[0]['genome'])
    else:
        logger.warning("No viable strategies found (Try relaxing data constraints).")

if __name__ == "__main__":
    run_optimization()

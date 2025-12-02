"""
strategy_optimizer.py (Fixed Logic)
-----------------------------------
1. FIX: Removed "Double Trimming" of data.
2. FIX: Switched to 'while' loop for correct trade skipping.
3. LOGIC: Evolving 500 strategies to find the highest Win Rate.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import random
import json
import logging
import os
from datetime import datetime

# --- CONFIG ---
ITERATIONS = 500          # How many strategies to breed
MIN_TRADES = 10           # Minimum trades to be considered valid
DATA_PERIOD = "2y"        # 2 Years required for SMA200

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Optimizer")

# -------------------------
# 1. DATA LOADING
# -------------------------
def get_tickers():
    if os.path.exists("ind_nifty500list.csv"):
        try:
            df = pd.read_csv("ind_nifty500list.csv")
            return [f"{x}.NS" for x in df['Symbol'].dropna().unique()]
        except: pass
    return ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "SBIN.NS", "ITC.NS", "TATAMOTORS.NS"]

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
                return bulk[ticker].copy() # Don't dropna yet
    except: pass
    return None

# -------------------------
# 2. MATH & INDICATORS
# -------------------------
def prepare_features(df):
    df = df.copy()
    # Calculate Indicators
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
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
    
    # CRITICAL FIX: Only drop rows AFTER calculations are done
    return df.dropna()

# -------------------------
# 3. GENERATOR
# -------------------------
def generate_random_strategy():
    return {
        "name": f"AI-Strat-{random.randint(1000,9999)}",
        # Diverse Entry Rules
        "trend_filter": random.choice(["SMA200", "SMA50", "NONE"]),
        "rsi_logic": random.choice(["OVERSOLD", "OVERBOUGHT", "NEUTRAL"]),
        "rsi_threshold": random.choice([30, 35, 40, 50, 60, 65, 70]),
        "adx_min": random.choice([0, 15, 20, 25]),
        # Diverse Risk Rules
        "sl_atr_mult": random.choice([1.0, 1.5, 2.0, 3.0]),
        "tgt_atr_mult": random.choice([1.5, 2.0, 3.0, 4.0])
    }

# -------------------------
# 4. EVALUATION ENGINE (FIXED)
# -------------------------
def evaluate_strategy(df, genome):
    wins, losses = 0, 0
    
    # FIX 1: Since df is already trimmed by dropna(), index 0 is valid.
    # We don't need to skip another 200 days.
    i = 0 
    max_idx = len(df) - 20
    
    # FIX 2: Use WHILE loop to correctly skip days after a trade
    while i < max_idx:
        row = df.iloc[i]
        close = row['Close']
        
        # 1. Trend
        trend_ok = True
        if genome["trend_filter"] == "SMA200" and close < row['SMA200']: trend_ok = False
        if genome["trend_filter"] == "SMA50" and close < row['SMA50']: trend_ok = False
        
        # 2. RSI
        rsi_val = row['RSI']
        rsi_ok = False
        thresh = genome["rsi_threshold"]
        
        if genome["rsi_logic"] == "OVERSOLD" and rsi_val < thresh: rsi_ok = True
        elif genome["rsi_logic"] == "OVERBOUGHT" and rsi_val > thresh: rsi_ok = True
        elif genome["rsi_logic"] == "NEUTRAL" and (40 < rsi_val < 60): rsi_ok = True
        
        # 3. ADX
        adx_ok = row['ADX'] > genome["adx_min"]
        
        if trend_ok and rsi_ok and adx_ok:
            # ENTRY!
            atr = row['ATR']
            sl = close - (genome["sl_atr_mult"] * atr)
            tgt = close + (genome["tgt_atr_mult"] * atr)
            
            # Simulate future days
            outcome = "OPEN"
            days_held = 0
            
            for j in range(1, 20): # Max hold 20 days
                days_held = j
                if (i + j) >= len(df): break
                
                fut = df.iloc[i + j]
                # Check Low for SL first (Conservative)
                if fut['Low'] <= sl: 
                    outcome = "LOSS"
                    break
                # Check High for Target
                if fut['High'] >= tgt: 
                    outcome = "WIN"
                    break
            
            if outcome == "WIN": wins += 1
            elif outcome == "LOSS": losses += 1
            
            # FIX 3: Skip the days we were in the trade
            i += days_held
        else:
            # No trade, move to next day
            i += 1
            
    return wins, losses

# -------------------------
# 5. MAIN LOOP
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
    logger.info(f"✅ Saved best strategy: {best_genome['name']}")

def run_optimization():
    tickers = get_tickers()
    bulk = robust_download(tickers)
    
    processed_data = []
    logger.info("📊 Preparing Data...")
    for t in tickers:
        raw = extract_df(bulk, t)
        if raw is not None and len(raw) > 250:
            processed_data.append(prepare_features(raw))
    
    if not processed_data:
        logger.error("❌ No valid data found.")
        return

    logger.info(f"🧬 Evolving {ITERATIONS} Strategies on {len(processed_data)} stocks...")
    
    results = []
    
    for idx in range(ITERATIONS):
        genome = generate_random_strategy()
        total_wins, total_losses = 0, 0
        
        # Test on random sample of 30 stocks for speed
        sample = random.sample(processed_data, min(len(processed_data), 30))
        
        for df in sample:
            w, l = evaluate_strategy(df, genome)
            total_wins += w
            total_losses += l
            
        total = total_wins + total_losses
        win_rate = (total_wins / total * 100) if total > 0 else 0
        
        if total >= MIN_TRADES:
            results.append({
                "genome": genome,
                "win_rate": round(win_rate, 1),
                "trades": total
            })
            
        if idx % 100 == 0:
            best_wr = max([r['win_rate'] for r in results]) if results else 0
            logger.info(f"... Gen {idx}: Best Win Rate {best_wr}%")

    # Final Sort
    results.sort(key=lambda x: x['win_rate'], reverse=True)
    
    logger.info("\n" + "="*40)
    if results:
        top = results[0]
        g = top['genome']
        logger.info(f"🏆 WINNER: {top['win_rate']}% Win Rate ({top['trades']} Trades)")
        logger.info(f"   • Trend: {g['trend_filter']}")
        logger.info(f"   • RSI: {g['rsi_logic']} ({g['rsi_threshold']})")
        logger.info(f"   • ADX > {g['adx_min']}")
        logger.info(f"   • Risk: Stop {g['sl_atr_mult']}x / Target {g['tgt_atr_mult']}x")
        
        save_best_strategy(g)
    else:
        logger.warning("⚠️ No viable strategy found. Creating default safe strategy.")
        default_strat = {
            "name": "Default-Safe", "trend_filter": "SMA200", "rsi_logic": "OVERSOLD", 
            "rsi_threshold": 30, "adx_min": 20, "sl_atr_mult": 1.5, "tgt_atr_mult": 3.0
        }
        save_best_strategy(default_strat)
    logger.info("="*40)

if __name__ == "__main__":
    run_optimization()

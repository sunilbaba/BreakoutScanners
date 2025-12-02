"""
strategy_optimizer.py (Fixed Loop & Deep Scan)
----------------------------------------------
1. FIX: Used 'while' loop to correctly handle trade duration skipping.
2. FIX: Increased sample size from 20 to 50 stocks per strategy.
3. LOGIC: Injects a "Baseline" strategy to ensure the engine works.
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
ITERATIONS = 500          
MIN_TRADES = 5            # Lowered to 5 to catch rare but good setups
DATA_PERIOD = "2y"
SAMPLE_SIZE = 50          # Test on 50 stocks (10% of Nifty 500)

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
    return ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "SBIN.NS", "ITC.NS", "TATAMOTORS.NS", "MARUTI.NS"]

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
# 2. MATH & INDICATORS
# -------------------------
def prepare_features(df):
    df = df.copy()
    # Trend
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
    
    return df.dropna()

# -------------------------
# 3. STRATEGY GENERATOR
# -------------------------
def generate_random_strategy():
    return {
        "name": f"AI-{random.randint(100,999)}",
        "trend_filter": random.choice(["SMA200", "NONE"]), # Simplified
        "rsi_logic": random.choice(["OVERSOLD", "MOMENTUM"]), # Removed NEUTRAL to force direction
        "rsi_threshold": random.choice([30, 35, 40, 60, 65, 70]),
        "adx_min": random.choice([0, 15, 20]),
        "sl_atr_mult": random.choice([1.0, 2.0]),
        "tgt_atr_mult": random.choice([2.0, 3.0, 4.0])
    }

# -------------------------
# 4. EVALUATION ENGINE (WHILE LOOP FIX)
# -------------------------
def evaluate_strategy(df, genome):
    wins, losses = 0, 0
    
    # Vector Access
    close = df['Close'].values
    low = df['Low'].values
    high = df['High'].values
    sma200 = df['SMA200'].values
    rsi = df['RSI'].values
    adx = df['ADX'].values
    atr = df['ATR'].values
    
    if len(close) < 250: return 0, 0

    # Loop Logic
    i = 200 # Start after warmup
    end_idx = len(df) - 20
    
    while i < end_idx:
        # Check Entry
        is_entry = False
        
        # 1. Trend
        if genome["trend_filter"] == "SMA200":
            if close[i] < sma200[i]: 
                i += 1; continue
        
        # 2. ADX
        if adx[i] <= genome["adx_min"]:
            i += 1; continue

        # 3. RSI
        rsi_val = rsi[i]
        if genome["rsi_logic"] == "OVERSOLD":
            if rsi_val < genome["rsi_threshold"]: is_entry = True
        elif genome["rsi_logic"] == "MOMENTUM":
            if rsi_val > genome["rsi_threshold"]: is_entry = True
            
        if is_entry:
            entry_price = close[i]
            sl = entry_price - (genome["sl_atr_mult"] * atr[i])
            tgt = entry_price + (genome["tgt_atr_mult"] * atr[i])
            
            # Simulate Forward
            outcome = "OPEN"
            days_held = 0
            
            for j in range(1, 15): # Max 15 days hold
                days_held = j
                curr_idx = i + j
                if curr_idx >= len(close): break
                
                # Check Low first (Conservative)
                if low[curr_idx] <= sl:
                    outcome = "LOSS"
                    break
                if high[curr_idx] >= tgt:
                    outcome = "WIN"
                    break
            
            if outcome == "WIN": wins += 1
            elif outcome == "LOSS": losses += 1
            
            # Skip forward by holding period
            i += days_held
        else:
            i += 1

    return wins, losses

# -------------------------
# 5. MAIN OPTIMIZER
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
    logger.info(f"✅ Strategy Saved: {best_genome['name']}")

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
        logger.error("❌ No valid data.")
        return

    logger.info(f"🧬 Testing {ITERATIONS} Strategies on sample of {SAMPLE_SIZE} stocks...")
    
    # INJECT BASELINE (Known Good Strategy)
    baseline = {
        "name": "BASELINE-DIP", "trend_filter": "SMA200", "rsi_logic": "OVERSOLD",
        "rsi_threshold": 40, "adx_min": 15, "sl_atr_mult": 1.5, "tgt_atr_mult": 3.0
    }
    
    results = []
    
    # Evolution Loop
    for idx in range(ITERATIONS + 1):
        if idx == 0: genome = baseline
        else: genome = generate_random_strategy()
        
        total_wins, total_losses = 0, 0
        
        # Test on random sample
        sample = random.sample(processed_data, min(len(processed_data), SAMPLE_SIZE))
        
        for df in sample:
            w, l = evaluate_strategy(df, genome)
            total_wins += w
            total_losses += l
            
        total = total_wins + total_losses
        win_rate = (total_wins / total * 100) if total > 0 else 0
        
        if total >= MIN_TRADES:
            results.append({ "genome": genome, "win_rate": round(win_rate, 1), "trades": total })
            
        # Verbose log for first 5 to prove it works
        if idx < 5:
            logger.info(f"Test {idx}: WR {win_rate:.1f}% | Trades {total}")

    results.sort(key=lambda x: x['win_rate'], reverse=True)
    
    logger.info("\n" + "="*40)
    if results:
        top = results[0]
        g = top['genome']
        logger.info(f"🏆 WINNER: {top['win_rate']}% Win Rate ({top['trades']} Trades)")
        logger.info(f"   • Signal: RSI {g['rsi_logic']} < {g['rsi_threshold']} (Trend: {g['trend_filter']})")
        logger.info(f"   • Risk: Stop {g['sl_atr_mult']} ATR | Target {g['tgt_atr_mult']} ATR")
        save_best_strategy(g)
    else:
        logger.warning("⚠️ Still no viable strategy. Using Baseline.")
        save_best_strategy(baseline)
    logger.info("="*40)

if __name__ == "__main__":
    run_optimization()

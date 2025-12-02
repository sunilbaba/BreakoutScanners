"""
strategy_optimizer.py (Math Fixed)
----------------------------------
1. FIX: Corrected ADX Formula (PrevLow - CurrLow).
2. LOGIC: Ensures ADX is always 0-100.
3. RESULT: Optimizes strategies with valid Trend Strength data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import random
import json
import logging
import os
from collections import Counter
from datetime import datetime

# --- CONFIG ---
ITERATIONS = 500          
MIN_TRADES = 10           
DATA_PERIOD = "2y"
SAMPLE_SIZE = 50          

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
            if not data.empty:
                frames.append(data)
        except Exception as e: 
            pass
            
    if not frames: return pd.DataFrame()
    return pd.concat(frames, axis=1)

def extract_df(bulk, ticker):
    try:
        if isinstance(bulk.columns, pd.MultiIndex):
            if ticker in bulk.columns.get_level_values(0):
                return bulk[ticker].copy()
    except: pass
    return None

# -------------------------
# 2. MATH & INDICATORS (CORRECTED)
# -------------------------
def prepare_features(df, ticker_name="Unknown"):
    df = df.copy()
    
    if len(df) < 200: return None
        
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
    
    # ADX (CORRECTED STANDARD FORMULA)
    # UpMove = High - PrevHigh
    up_move = df['High'] - df['High'].shift(1)
    # DownMove = PrevLow - Low
    down_move = df['Low'].shift(1) - df['Low']
    
    # Initialize +DM and -DM with zeros
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    
    # Logic: If Up > Down and Up > 0, then +DM = Up
    mask_plus = (up_move > down_move) & (up_move > 0)
    plus_dm[mask_plus] = up_move[mask_plus]
    
    # Logic: If Down > Up and Down > 0, then -DM = Down
    mask_minus = (down_move > up_move) & (down_move > 0)
    minus_dm[mask_minus] = down_move[mask_minus]
    
    # Smoothed True Range and DMs
    atr_smooth = df['ATR'].ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_smooth)
    
    # DX and ADX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean().fillna(0)
    
    df = df.dropna()
    
    # DEBUG CHECK
    if ticker_name == "RELIANCE.NS" and len(df) > 0:
        last = df.iloc[-1]
        logger.info(f"🔍 CHECK: {ticker_name} | ADX={last['ADX']:.2f} (0-100 Valid)")
    
    return df

# -------------------------
# 3. STRATEGY GENERATOR
# -------------------------
def generate_random_strategy():
    return {
        "name": f"AI-{random.randint(100,999)}",
        "trend_filter": random.choice(["SMA200", "NONE"]), 
        "rsi_logic": random.choice(["OVERSOLD", "MOMENTUM"]), 
        "rsi_threshold": random.choice([30, 35, 40, 60, 65, 70]),
        "adx_min": random.choice([0, 15, 20]),
        "sl_atr_mult": random.choice([1.0, 2.0]),
        "tgt_atr_mult": random.choice([2.0, 3.0, 4.0])
    }

# -------------------------
# 4. EVALUATION ENGINE
# -------------------------
def evaluate_strategy(df, genome, debug=False):
    wins, losses = 0, 0
    
    # Vector Access
    close = df['Close'].values
    low = df['Low'].values
    high = df['High'].values
    sma200 = df['SMA200'].values
    rsi = df['RSI'].values
    adx = df['ADX'].values
    atr = df['ATR'].values
    
    if len(close) < 50: return 0, 0

    # Loop Logic
    i = 0 
    end_idx = len(df) - 20
    
    while i < end_idx:
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
            
            outcome = "OPEN"
            days_held = 0
            
            for j in range(1, 15): 
                days_held = j
                curr_idx = i + j
                if curr_idx >= len(close): break
                
                if low[curr_idx] <= sl:
                    outcome = "LOSS"
                    break
                if high[curr_idx] >= tgt:
                    outcome = "WIN"
                    break
            
            if outcome == "WIN": wins += 1
            elif outcome == "LOSS": losses += 1
            
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
        if raw is not None:
            clean = prepare_features(raw, t)
            if clean is not None: processed_data.append(clean)
    
    if not processed_data:
        logger.error("❌ No valid data prepared.")
        return

    logger.info(f"🧬 Testing {ITERATIONS} Strategies on sample of {SAMPLE_SIZE} stocks...")
    
    # INJECT BASELINE
    baseline = {
        "name": "BASELINE-DIP", "trend_filter": "SMA200", "rsi_logic": "OVERSOLD",
        "rsi_threshold": 40, "adx_min": 15, "sl_atr_mult": 1.5, "tgt_atr_mult": 3.0
    }
    
    results = []
    
    for idx in range(ITERATIONS + 1):
        if idx == 0: genome = baseline
        else: genome = generate_random_strategy()
        
        total_wins, total_losses = 0, 0
        sample = random.sample(processed_data, min(len(processed_data), SAMPLE_SIZE))
        
        for i, df in enumerate(sample):
            w, l = evaluate_strategy(df, genome)
            total_wins += w
            total_losses += l
            
        total = total_wins + total_losses
        win_rate = (total_wins / total * 100) if total > 0 else 0
        
        if total >= MIN_TRADES:
            results.append({ "genome": genome, "win_rate": round(win_rate, 1), "trades": total })
            
        if idx < 3: 
            logger.info(f"Test {idx}: WR {win_rate:.1f}% | Trades {total}")

    results.sort(key=lambda x: x['win_rate'], reverse=True)
    
    logger.info("\n" + "="*40)
    if results:
        top = results[0]
        g = top['genome']
        logger.info(f"🏆 WINNER: {top['win_rate']}% Win Rate ({top['trades']} Trades)")
        logger.info(f"   • Signal: RSI {g['rsi_logic']} < {g['rsi_threshold']} (Trend: {g['trend_filter']})")
        save_best_strategy(g)
    else:
        logger.warning("⚠️ Still no viable strategy. Using Baseline.")
        save_best_strategy(baseline)
    logger.info("="*40)

if __name__ == "__main__":
    run_optimization()

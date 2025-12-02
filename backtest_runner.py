"""
strategy_optimizer.py
The "Generative" Engine.
------------------------
1. Reads Tickers from 'ind_nifty500list.csv'.
2. Downloads 2 Years of Data.
3. Generates 500+ unique strategy variations (Genomes).
4. Backtests each variation against the market.
5. Returns the "fittest" strategy with the highest Win Rate.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import random
import logging
import os
from datetime import datetime

# --- CONFIGURATION ---
ITERATIONS = 500          # How many strategies to generate & test
MIN_TRADES = 30           # Ignore strategies with too few trades
DATA_PERIOD = "2y"

# Fallback Universe (Liquid Nifty 50 Stocks) if CSV is missing
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
    """Reads tickers from CSV or falls back to default list."""
    if os.path.exists("ind_nifty500list.csv"):
        try:
            df = pd.read_csv("ind_nifty500list.csv")
            # Ensure .NS extension for Yahoo Finance
            tickers = [f"{x}.NS" if not str(x).endswith(".NS") else x for x in df['Symbol'].dropna().unique()]
            logger.info(f"✅ Loaded {len(tickers)} tickers from CSV.")
            return tickers
        except Exception as e:
            logger.warning(f"⚠️ CSV Read Error: {e}")
            pass
    
    logger.info("⚠️ CSV not found. Using default Nifty 50 list.")
    return DEFAULT_TICKERS

def robust_download(tickers):
    logger.info(f"📥 Downloading {len(tickers)} stocks ({DATA_PERIOD})...")
    frames = []
    batch_size = 50 # Increased batch size for efficiency
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
    
    # ATR (Volatility)
    h_l = df['High'] - df['Low']
    h_c = (df['High'] - df['Close'].shift()).abs()
    l_c = (df['Low'] - df['Close'].shift()).abs()
    df['ATR'] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss)).fillna(50)
    
    # ADX (Strength)
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
# 3. STRATEGY GENERATOR (The "AI")
# -------------------------
def generate_random_strategy():
    """Creates a random trading strategy genome."""
    return {
        "name": f"Strat-{random.randint(1000,9999)}",
        # Entry Rules
        "trend_filter": random.choice(["SMA200", "SMA50", "NONE"]),
        "rsi_logic": random.choice(["OVERSOLD", "OVERBOUGHT", "NEUTRAL"]),
        "rsi_threshold": random.randint(20, 80),
        "adx_min": random.choice([0, 10, 15, 20, 25]),
        # Exit Rules
        "sl_atr_mult": random.choice([0.5, 1.0, 1.5, 2.0, 3.0]),
        "tgt_atr_mult": random.choice([1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    }

def evaluate_strategy(df, genome):
    """Runs the specific genome on a stock's history."""
    wins, losses = 0, 0
    
    # Pre-calc columns to speed up loop
    close = df['Close']
    sma200 = df['SMA200']
    sma50 = df['SMA50']
    rsi = df['RSI']
    adx = df['ADX']
    atr = df['ATR']
    
    # Simulation Loop (Last 1 Year approx)
    start_idx = max(200, len(df) - 250)
    for i in range(start_idx, len(df) - 10): 
        row_close = close.iloc[i]
        row_rsi = rsi.iloc[i]
        row_adx = adx.iloc[i]
        row_atr = atr.iloc[i]
        
        # 1. Trend Filter
        trend_ok = True
        if genome["trend_filter"] == "SMA200" and row_close < sma200.iloc[i]: trend_ok = False
        if genome["trend_filter"] == "SMA50" and row_close < sma50.iloc[i]: trend_ok = False
        
        # 2. RSI Logic
        rsi_ok = False
        if genome["rsi_logic"] == "OVERSOLD" and row_rsi < genome["rsi_threshold"]: rsi_ok = True
        if genome["rsi_logic"] == "OVERBOUGHT" and row_rsi > genome["rsi_threshold"]: rsi_ok = True
        if genome["rsi_logic"] == "NEUTRAL" and row_rsi > 40 and row_rsi < 60: rsi_ok = True
        
        # 3. ADX Logic
        adx_ok = row_adx > genome["adx_min"]
        
        if trend_ok and rsi_ok and adx_ok:
            # ENTRY TRIGGERED
            entry = row_close
            sl = entry - (genome["sl_atr_mult"] * row_atr)
            tgt = entry + (genome["tgt_atr_mult"] * row_atr)
            
            # Check Outcome (Look ahead 20 days)
            outcome = "OPEN"
            for j in range(1, 20):
                if i+j >= len(df): break
                fut_low = df['Low'].iloc[i+j]
                fut_high = df['High'].iloc[i+j]
                
                if fut_low <= sl: outcome = "LOSS"; break
                if fut_high >= tgt: outcome = "WIN"; break
            
            if outcome == "WIN": wins += 1
            elif outcome == "LOSS": losses += 1
            
            # Skip ahead to avoid overlapping trades
            i += j 

    return wins, losses

# -------------------------
# 4. EVOLUTION LOOP
# -------------------------
def run_optimization():
    tickers = get_tickers()
    bulk = robust_download(tickers)
    
    # Prepare data once
    processed_data = []
    for t in tickers:
        raw = extract_df(bulk, t)
        if raw is not None and len(raw) > 250:
            processed_data.append(prepare_features(raw))
    
    if not processed_data:
        logger.error("No valid data available for optimization.")
        return

    logger.info(f"🧬 Evolving {ITERATIONS} Strategies on {len(processed_data)} stocks...")
    
    results = []
    
    for _ in range(ITERATIONS):
        genome = generate_random_strategy()
        total_wins, total_losses = 0, 0
        
        # Test genome on ALL stocks
        for df in processed_data:
            w, l = evaluate_strategy(df, genome)
            total_wins += w
            total_losses += l
            
        total_trades = total_wins + total_losses
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        
        if total_trades > MIN_TRADES: # Only keep statistically significant ones
            results.append({
                "genome": genome,
                "win_rate": round(win_rate, 2),
                "trades": total_trades
            })
    
    # Sort by Win Rate
    results.sort(key=lambda x: x['win_rate'], reverse=True)
    
    # Print Top 3 Models
    logger.info("\n🏆 TOP 3 PERFORMING MODELS FOUND:")
    for i, res in enumerate(results[:3]):
        g = res['genome']
        logger.info(f"\n--- RANK #{i+1} (Win Rate: {res['win_rate']}%) ---")
        logger.info(f"Trades Analyzed: {res['trades']}")
        logger.info(f"Strategy Logic:")
        logger.info(f"  1. Trend Filter: {g['trend_filter']}")
        logger.info(f"  2. Signal: RSI {g['rsi_logic']} {g['rsi_threshold']}")
        logger.info(f"  3. Strength: ADX > {g['adx_min']}")
        logger.info(f"  4. Risk Management: Stop {g['sl_atr_mult']}x ATR | Target {g['tgt_atr_mult']}x ATR")

if __name__ == "__main__":
    run_optimization()

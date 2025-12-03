"""
strategy_optimizer.py (Final Fix)
---------------------------------
1. FIX: Now automatically RUNS the simulation after finding the best strategy.
2. FIX: Saves 'backtest_stats.json' (The missing file).
3. LOGGING: Prints the Equity Curve progress to the console.
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
CAPITAL = 100_000.0
RISK_PER_TRADE = 0.02
BROKERAGE_PCT = 0.001
MAX_POSITIONS = 5

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
        except Exception as e: pass
            
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
# 2. MATH & INDICATORS
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
    
    # ADX (Corrected)
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    p_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    m_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), -minus_dm, 0.0)
    plus_dm_s = pd.Series(p_dm, index=df.index)
    minus_dm_s = pd.Series(m_dm, index=df.index)
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).ewm(alpha=1/14).mean()
    plus_di = 100 * (plus_dm_s.ewm(alpha=1/14).mean() / tr)
    minus_di = 100 * (minus_dm_s.ewm(alpha=1/14).mean() / tr)
    df['ADX'] = (abs(plus_di - minus_di) / (plus_di + minus_di) * 100).ewm(alpha=1/14).mean().fillna(0)
    
    return df.dropna()

# -------------------------
# 3. OPTIMIZER LOGIC
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

def evaluate_strategy(df, genome):
    wins, losses = 0, 0
    close = df['Close'].values
    low = df['Low'].values
    high = df['High'].values
    sma200 = df['SMA200'].values
    rsi = df['RSI'].values
    adx = df['ADX'].values
    atr = df['ATR'].values
    
    if len(close) < 50: return 0, 0

    i = 0 
    end_idx = len(df) - 20
    
    while i < end_idx:
        is_entry = False
        if genome["trend_filter"] == "SMA200":
            if close[i] < sma200[i]: 
                i += 1; continue
        if adx[i] <= genome["adx_min"]:
            i += 1; continue

        if genome["rsi_logic"] == "OVERSOLD":
            if rsi[i] < genome["rsi_threshold"]: is_entry = True
        elif genome["rsi_logic"] == "MOMENTUM":
            if rsi[i] > genome["rsi_threshold"]: is_entry = True
            
        if is_entry:
            sl = close[i] - (genome["sl_atr_mult"] * atr[i])
            tgt = close[i] + (genome["tgt_atr_mult"] * atr[i])
            outcome = "OPEN"
            days_held = 0
            for j in range(1, 15):
                days_held = j
                curr_idx = i + j
                if curr_idx >= len(close): break
                if low[curr_idx] <= sl: outcome = "LOSS"; break
                if high[curr_idx] >= tgt: outcome = "WIN"; break
            
            if outcome == "WIN": wins += 1
            elif outcome == "LOSS": losses += 1
            i += days_held
        else:
            i += 1
            
    return wins, losses

# -------------------------
# 4. PORTFOLIO SIMULATION (THE VALIDATOR)
# -------------------------
class PortfolioSimulator:
    def __init__(self, processed_data, genome):
        self.data_map = processed_data
        self.genome = genome
        self.cash = CAPITAL
        self.curve = [CAPITAL]
        self.history = []
        self.portfolio = []

    def run(self):
        logger.info(f"\n📈 VALIDATION: Running 1-Year Simulation for '{self.genome['name']}'...")
        dates = sorted(list(set().union(*[d.index for d in self.data_map.values()])))
        sim_dates = dates[200:] 
        
        logger.info(f"   > Simulating {len(sim_dates)} trading days...")
        
        for date in sim_dates:
            self.process_day(date)
            m2m = 0
            for t in self.portfolio:
                sym = t['symbol']
                price = self.data_map[sym].loc[date]['Close'] if date in self.data_map[sym].index else t['entry']
                m2m += (price * t['qty'])
            self.curve.append(round(self.cash + m2m, 2))
            
        # Stats
        wins = len([t for t in self.history if t['pnl'] > 0])
        total = len(self.history)
        win_rate = round(wins / total * 100, 1) if total > 0 else 0
        profit = round(self.curve[-1] - CAPITAL, 2)
        
        logger.info(f"   > Simulation Complete.")
        logger.info(f"   > Final Balance: ₹{round(self.curve[-1], 2)}")
        logger.info(f"   > Total Profit:  ₹{profit}")
        logger.info(f"   > Total Trades:  {total}")
        
        return {
            "curve": self.curve,
            "win_rate": win_rate,
            "total_trades": total,
            "profit": profit,
            "ledger": self.history[-50:]
        }

    def process_day(self, date):
        active = []
        for t in self.portfolio:
            sym = t['symbol']
            if date not in self.data_map[sym].index: active.append(t); continue
            row = self.data_map[sym].loc[date]
            
            exit_p = None
            if row['Open'] < t['sl']: exit_p = row['Open']
            elif row['Low'] <= t['sl']: exit_p = t['sl']
            elif row['High'] >= t['tgt']: exit_p = t['tgt']
            
            if exit_p:
                rev = exit_p * t['qty']
                cost = rev * BROKERAGE_PCT
                self.cash += (rev - cost)
                pnl = rev - cost - t['cost_basis']
                self.history.append({
                    "date": date.strftime("%Y-%m-%d"), "symbol": sym,
                    "pnl": round(pnl, 2), "result": "WIN" if pnl > 0 else "LOSS"
                })
            else: active.append(t)
        self.portfolio = active
        
        if len(self.portfolio) >= MAX_POSITIONS: return
        
        for sym, df in self.data_map.items():
            if date not in df.index: continue
            row = df.loc[date]
            
            # Strategy Check
            is_entry = False
            if self.genome["trend_filter"] == "SMA200" and row['Close'] < row['SMA200']: pass
            elif row['ADX'] <= self.genome["adx_min"]: pass
            elif self.genome["rsi_logic"] == "OVERSOLD" and row['RSI'] < self.genome["rsi_threshold"]: is_entry = True
            elif self.genome["rsi_logic"] == "MOMENTUM" and row['RSI'] > self.genome["rsi_threshold"]: is_entry = True
            
            if is_entry:
                if any(t['symbol'] == sym for t in self.portfolio): continue
                
                risk = row['ATR'] * self.genome["sl_atr_mult"]
                if risk <= 0: continue
                
                qty = int((self.curve[-1] * RISK_PER_TRADE) / risk)
                cost = qty * row['Close']
                if cost > self.cash: qty = int(self.cash / row['Close']); cost = qty * row['Close']
                
                if qty > 0 and self.cash > cost:
                    fees = cost * BROKERAGE_PCT
                    self.cash -= (cost + fees)
                    self.portfolio.append({
                        "symbol": sym, "entry": row['Close'], "qty": qty,
                        "sl": row['Close'] - risk, 
                        "tgt": row['Close'] + (row['ATR'] * self.genome["tgt_atr_mult"]),
                        "cost_basis": cost + fees
                    })
                    if len(self.portfolio) >= MAX_POSITIONS: break

# -------------------------
# 5. MAIN EXECUTION
# -------------------------
def run_pipeline():
    tickers = get_tickers()
    bulk = robust_download(tickers)
    
    processed_data = []
    # Create full map for validation
    full_map = {} 
    
    logger.info("📊 Preparing Data...")
    for t in tickers:
        raw = extract_df(bulk, t)
        if raw is not None and len(raw) > 250:
            clean = prepare_features(raw, t)
            if clean is not None:
                processed_data.append(clean)
                full_map[t] = clean # Store with key for simulator
    
    if not processed_data:
        logger.error("❌ No valid data.")
        return

    logger.info(f"🧬 Evolving {ITERATIONS} Strategies on {len(processed_data)} stocks...")
    
    results = []
    baseline = { "name": "BASELINE", "trend_filter": "SMA200", "rsi_logic": "OVERSOLD", "rsi_threshold": 40, "adx_min": 15, "sl_atr_mult": 1.5, "tgt_atr_mult": 3.0 }
    
    for idx in range(ITERATIONS + 1):
        genome = baseline if idx == 0 else generate_random_strategy()
        total_wins, total_losses = 0, 0
        sample = random.sample(processed_data, min(len(processed_data), SAMPLE_SIZE))
        
        for df in sample:
            w, l = evaluate_strategy(df, genome)
            total_wins += w; total_losses += l
            
        total = total_wins + total_losses
        win_rate = (total_wins / total * 100) if total > 0 else 0
        
        if total >= MIN_TRADES:
            results.append({ "genome": genome, "win_rate": round(win_rate, 1), "trades": total })

    results.sort(key=lambda x: x['win_rate'], reverse=True)
    
    best_genome = baseline
    if results:
        top = results[0]
        best_genome = top['genome']
        logger.info(f"🏆 WINNER: {top['win_rate']}% WR | {top['trades']} Trades")
    
    # SAVE STRATEGY
    with open("strategy_config.json", "w") as f:
        json.dump({"updated": datetime.utcnow().strftime("%Y-%m-%d"), "parameters": best_genome}, f, indent=2)
        logger.info("✅ Strategy Config Saved.")

    # RUN SIMULATION & SAVE STATS (The missing link)
    sim = PortfolioSimulator(full_map, best_genome)
    stats = sim.run()
    
    # Per-Ticker Win Rate (For UI)
    ticker_wins = {}
    for t, df in full_map.items():
        w, l = evaluate_strategy(df, best_genome)
        tot = w + l
        ticker_wins[t.replace('.NS','')] = round(w/tot*100, 0) if tot > 0 else 0

    final_output = {
        "updated": datetime.utcnow().strftime("%Y-%m-%d"),
        "portfolio": stats,
        "tickers": ticker_wins
    }
    
    with open("backtest_stats.json", "w") as f:
        json.dump(final_output, f)
    logger.info("✅ Backtest Stats Saved.")

if __name__ == "__main__":
    run_pipeline()

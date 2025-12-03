"""
backtest_runner.py
THE "GENETIC EVOLUTION" ENGINE (AI-Powered)
-------------------------------------------
1. Uses Genetic Algorithms (GA) instead of Random Search.
2. Evolves strategies over Generations (Breeding + Mutation).
3. Optimizes for "Profit Factor" (Stability), not just Win Rate.
4. Detailed Logging of the Evolution Process.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import logging
import random
import copy
from datetime import datetime

# --- CONFIG ---
DATA_PERIOD = "2y"
CACHE_FILE = "backtest_stats.json"
STRATEGY_FILE = "strategy_config.json"

CAPITAL = 100_000.0
RISK_PER_TRADE = 0.02
BROKERAGE_PCT = 0.001

# AI PARAMS
POPULATION_SIZE = 50      # Strategies per generation
GENERATIONS = 5           # How many times to evolve
MUTATION_RATE = 0.2       # 20% chance to tweak a parameter

SECTOR_INDICES = {
    "NIFTY 50": "^NSEI", "BANK": "^NSEBANK", "AUTO": "^CNXAUTO", "IT": "^CNXIT",
    "METAL": "^CNXMETAL", "PHARMA": "^CNXPHARMA", "FMCG": "^CNXFMCG"
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("AI-Evo")

# -------------------------
# 1. DATA LAYER
# -------------------------
def get_tickers():
    if os.path.exists("ind_nifty500list.csv"):
        try:
            df = pd.read_csv("ind_nifty500list.csv")
            return [f"{x}.NS" for x in df['Symbol'].dropna().unique()]
        except: pass
    return ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "SBIN.NS"]

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
                return bulk[ticker].copy()
    except: pass
    return None

def prepare_features(df):
    df = df.copy()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    
    h_l = df['High'] - df['Low']
    h_c = (df['High'] - df['Close'].shift()).abs()
    l_c = (df['Low'] - df['Close'].shift()).abs()
    df['ATR'] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss)).fillna(50)
    
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    p_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    m_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), -minus_dm, 0.0)
    
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).ewm(alpha=1/14).mean()
    df['ADX'] = (abs(100*(pd.Series(p_dm, index=df.index).ewm(alpha=1/14).mean()/tr) - 
                     100*(pd.Series(m_dm, index=df.index).ewm(alpha=1/14).mean()/tr)) /
                 (100*(pd.Series(p_dm, index=df.index).ewm(alpha=1/14).mean()/tr) + 
                  100*(pd.Series(m_dm, index=df.index).ewm(alpha=1/14).mean()/tr)) * 100).ewm(alpha=1/14).mean().fillna(0)
    
    return df.dropna()

# -------------------------
# 2. GENETIC ALGORITHM ENGINE
# -------------------------

# Define the DNA (Genes) of a strategy
GENE_POOL = {
    "trend_filter": ["SMA200", "SMA50", "NONE"],
    "rsi_logic": ["OVERSOLD", "MOMENTUM"],
    "rsi_threshold": [30, 35, 40, 45, 55, 60, 65, 70],
    "adx_min": [10, 15, 20, 25, 30],
    "sl_mult": [1.0, 1.5, 2.0, 2.5],
    "tgt_mult": [2.0, 2.5, 3.0, 4.0, 5.0]
}

def create_random_genome():
    return {k: random.choice(v) for k, v in GENE_POOL.items()}

def crossover(parent_a, parent_b):
    """Mating: Mixes genes from two parents to create a child."""
    child = {}
    for key in GENE_POOL.keys():
        # 50% chance to take gene from Father or Mother
        child[key] = parent_a[key] if random.random() > 0.5 else parent_b[key]
    return child

def mutate(genome):
    """Mutation: Randomly changes one gene to introduce diversity."""
    if random.random() < MUTATION_RATE:
        gene_to_change = random.choice(list(GENE_POOL.keys()))
        genome[gene_to_change] = random.choice(GENE_POOL[gene_to_change])
    return genome

def fitness_score(df, genome):
    """Calculates a score based on Profitability (Not just Wins)."""
    wins, losses = 0, 0
    
    close = df['Close'].values
    low = df['Low'].values
    high = df['High'].values
    sma200 = df['SMA200'].values
    rsi = df['RSI'].values
    adx = df['ADX'].values
    atr = df['ATR'].values
    
    start_idx = 0
    end_idx = len(df) - 20
    i = start_idx
    
    while i < end_idx:
        is_entry = False
        
        # Gene Logic
        if genome["trend_filter"] == "SMA200" and close[i] < sma200[i]: i += 1; continue
        if adx[i] <= genome["adx_min"]: i += 1; continue
        
        if genome["rsi_logic"] == "OVERSOLD":
            if rsi[i] < genome["rsi_threshold"]: is_entry = True
        elif genome["rsi_logic"] == "MOMENTUM":
            if rsi[i] > genome["rsi_threshold"]: is_entry = True
            
        if is_entry:
            sl = close[i] - (genome["sl_mult"] * atr[i])
            tgt = close[i] + (genome["tgt_mult"] * atr[i])
            
            outcome = "OPEN"
            days_held = 0
            for j in range(1, 15):
                days_held = j
                idx = i + j
                if idx >= len(close): break
                if low[idx] <= sl: outcome = "LOSS"; break
                if high[idx] >= tgt: outcome = "WIN"; break
            
            if outcome == "WIN": wins += 1
            elif outcome == "LOSS": losses += 1
            i += days_held
        else:
            i += 1
            
    # Score Formula: (Wins * Target_Mult) - (Losses * Stop_Mult)
    # This prioritizes strategies that make money, not just win often.
    score = (wins * genome["tgt_mult"]) - (losses * genome["sl_mult"])
    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    return score, win_rate, total_trades

def run_evolution(processed_data):
    logger.info(f"🧬 STARTING EVOLUTION ({GENERATIONS} Gens, {POPULATION_SIZE} Pop)")
    
    # 1. Create Initial Population
    population = [create_random_genome() for _ in range(POPULATION_SIZE)]
    
    # Inject Baseline (Don't forget what we already know works)
    population[0] = { "trend_filter": "SMA200", "rsi_logic": "OVERSOLD", "rsi_threshold": 30, "adx_min": 20, "sl_mult": 1.0, "tgt_mult": 3.0 }

    best_genome = None
    best_score = -9999
    
    for gen in range(GENERATIONS):
        scores = []
        
        # Evaluate entire population
        # Speed Optimization: Sample 30 random stocks per generation to test fitness
        sample_data = random.sample(processed_data, min(len(processed_data), 30))
        
        for genome in population:
            gen_score = 0
            gen_wr = 0
            gen_trades = 0
            
            for df in sample_data:
                s, w, t = fitness_score(df, genome)
                gen_score += s
                gen_trades += t
            
            scores.append((genome, gen_score, gen_trades))
            
        # Sort by Score (High to Low)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        top_genome = scores[0][0]
        top_score = scores[0][1]
        
        logger.info(f"   > Gen {gen+1}: Best Score {top_score:.1f} | Trades {scores[0][2]}")
        
        if top_score > best_score:
            best_score = top_score
            best_genome = top_genome
            
        # Selection: Keep Top 20% (Elites)
        elites_count = int(POPULATION_SIZE * 0.2)
        elites = [s[0] for s in scores[:elites_count]]
        
        # Breeding: Fill the rest of population
        new_pop = elites[:] # Carry over elites
        while len(new_pop) < POPULATION_SIZE:
            parent_a = random.choice(elites)
            parent_b = random.choice(elites)
            child = crossover(parent_a, parent_b)
            child = mutate(child)
            new_pop.append(child)
            
        population = new_pop

    logger.info("\n🏆 EVOLUTION COMPLETE. Best Strategy:")
    logger.info(json.dumps(best_genome, indent=2))
    
    # Save to file
    with open(STRATEGY_FILE, "w") as f:
        json.dump({"updated": datetime.utcnow().strftime("%Y-%m-%d"), "parameters": best_genome}, f, indent=2)
    
    return best_genome

# -------------------------
# 3. PORTFOLIO VALIDATION
# -------------------------
class PortfolioSimulator:
    def __init__(self, data, genome):
        self.data = data
        self.genome = genome
        self.cash = CAPITAL
        self.curve = [CAPITAL]
        self.history = []
        self.portfolio = []

    def run(self):
        logger.info(f"\n📈 VALIDATION: Backtesting Winner on ALL stocks...")
        dates = sorted(list(set().union(*[d.index for d in self.data.values()])))
        sim_dates = dates[200:]
        
        for date in sim_dates:
            self.process_day(date)
            m2m = 0
            for t in self.portfolio:
                sym = t['symbol']
                price = self.data[sym].loc[date]['Close'] if date in self.data[sym].index else t['entry']
                m2m += (price * t['qty'])
            self.curve.append(round(self.cash + m2m, 2))
            
        wins = len([t for t in self.history if t['pnl'] > 0])
        total = len(self.history)
        win_rate = round(wins / total * 100, 1) if total > 0 else 0
        profit = round(self.curve[-1] - CAPITAL, 2)
        
        logger.info(f"   > Profit: ₹{profit} | Trades: {total} | Win Rate: {win_rate}%")
        return {"curve": self.curve, "win_rate": win_rate, "total_trades": total, "profit": profit, "ledger": self.history[-50:]}

    def process_day(self, date):
        active = []
        # Exits
        for t in self.portfolio:
            sym = t['symbol']
            if date not in self.data[sym].index: active.append(t); continue
            row = self.data[sym].loc[date]
            
            exit_p = None
            if row['Open'] < t['sl']: exit_p = row['Open']
            elif row['Low'] <= t['sl']: exit_p = t['sl']
            elif row['Open'] > t['tgt']: exit_p = row['Open']
            elif row['High'] >= t['tgt']: exit_p = t['tgt']
            
            if exit_p:
                pnl = (exit_p - t['entry']) * t['qty']
                cost = exit_p * t['qty'] * BROKERAGE_PCT
                self.cash += (exit_p * t['qty'] - cost)
                self.history.append({"date": date.strftime("%Y-%m-%d"), "symbol": sym, "pnl": round(pnl-cost, 2), "result": "WIN" if pnl>0 else "LOSS"})
            else: active.append(t)
        self.portfolio = active
        
        # Entries (Max 5)
        if len(self.portfolio) >= 5: return
        for sym, df in self.data.items():
            if date not in df.index: continue
            row = df.loc[date]
            
            g = self.genome
            trend_ok = True
            if g["trend_filter"] == "SMA200" and row['Close'] < row['SMA200']: trend_ok = False
            
            rsi_ok = False
            if g["rsi_logic"] == "OVERSOLD" and row['RSI'] < g["rsi_threshold"]: rsi_ok = True
            elif g["rsi_logic"] == "MOMENTUM" and row['RSI'] > g["rsi_threshold"]: rsi_ok = True
            
            if trend_ok and rsi_ok and row['ADX'] > g["adx_min"]:
                if any(t['symbol'] == sym for t in self.portfolio): continue
                risk = row['ATR'] * g["sl_mult"]
                if risk <= 0: continue
                
                qty = int((self.curve[-1] * RISK_PER_TRADE) / risk)
                cost = qty * row['Close']
                if cost > self.cash: qty = int(self.cash / row['Close']); cost = qty * row['Close']
                
                if qty > 0 and self.cash > cost:
                    self.cash -= (cost + (cost*BROKERAGE_PCT))
                    self.portfolio.append({"symbol": sym, "entry": row['Close'], "qty": qty, "sl": row['Close']-risk, "tgt": row['Close']+(row['ATR']*g["tgt_mult"]), "entry_cost": cost*BROKERAGE_PCT})
                    if len(self.portfolio) >= 5: break

# --- 4. MAIN ---
if __name__ == "__main__":
    tickers = get_tickers()
    bulk = robust_download(tickers)
    processed = {}
    logger.info("📊 Preparing Data...")
    for t in tickers:
        raw = extract_df(bulk, t)
        if raw is not None and len(raw) > 250:
            processed[t] = prepare_features(raw)
            
    if not processed: exit()
    
    # 1. Evolve Best Strategy
    best_genome = run_evolution(list(processed.values()))
    
    # 2. Validate on Full Portfolio
    sim = PortfolioSimulator(processed, best_genome)
    stats = sim.run()
    
    # 3. Save Stats
    output = {
        "updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "portfolio": stats,
        "tickers": {} # Fill if needed
    }
    with open(CACHE_FILE, "w") as f: json.dump(output, f)
    logger.info("✅ Backtest Complete.")

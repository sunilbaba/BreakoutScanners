"""
backtest_runner.py
THE "GENETIC AI" STRATEGY ENGINE (Fixed)
----------------------------------------
1. FIX: Added missing 'ITERATIONS' variable definition.
2. Logic: Downloads Data -> Evolves Strategy (GA) -> Validates -> Saves.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import random
import json
import os
import logging
import copy
from datetime import datetime

# --- CONFIG ---
DATA_PERIOD = "2y" 
CACHE_FILE = "backtest_stats.json"
STRATEGY_FILE = "strategy_config.json"

CAPITAL = 100_000.0
RISK_PER_TRADE = 0.02
BROKERAGE_PCT = 0.001
MAX_POSITIONS = 5

# AI PARAMS
POPULATION_SIZE = 50      
GENERATIONS = 5           
MUTATION_RATE = 0.2       
MIN_TRADES = 10
SAMPLE_SIZE = 250 

# Fix: Define total iterations for logging
ITERATIONS = GENERATIONS * POPULATION_SIZE 

SECTOR_INDICES = {
    "NIFTY 50": "^NSEI", "BANK": "^NSEBANK", "AUTO": "^CNXAUTO", "IT": "^CNXIT",
    "METAL": "^CNXMETAL", "PHARMA": "^CNXPHARMA", "FMCG": "^CNXFMCG",
    "ENERGY": "^CNXENERGY", "REALTY": "^CNXREALTY", "PSU BANK": "^CNXPSUBANK"
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("AI-Lab")

# -------------------------
# 1. DATA ENGINE
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
            if not data.empty: frames.append(data)
        except: pass
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()

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
def prepare_features(df):
    df = df.copy()
    if len(df) < 200: return None
    
    # Trend
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    
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
    p_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    m_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), -minus_dm, 0.0)
    plus_dm_s = pd.Series(p_dm, index=df.index)
    minus_dm_s = pd.Series(m_dm, index=df.index)
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).ewm(alpha=1/14).mean()
    plus_di = 100 * (plus_dm_s.ewm(alpha=1/14).mean() / tr)
    minus_di = 100 * (minus_dm_s.ewm(alpha=1/14).mean() / tr)
    df['ADX'] = (abs(plus_di - minus_di) / (plus_di + minus_di) * 100).ewm(alpha=1/14).mean().fillna(0)
    
    # Bollinger
    df['BB_MID'] = df['Close'].rolling(20).mean()
    df['BB_STD'] = df['Close'].rolling(20).std()
    df['BB_LOW'] = df['BB_MID'] - (2 * df['BB_STD'])

    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_SIG'] = df['MACD'].ewm(span=9).mean()

    return df.dropna()

# -------------------------
# 3. GENETIC ALGORITHM ENGINE
# -------------------------
GENE_POOL = {
    "strategy_type": ["RSI_DIP", "BB_REVERSAL", "MACD_CROSS"],
    "trend_filter": ["SMA200", "SMA50", "NONE"],
    "adx_min": [15, 20, 25],
    "sl_mult": [1.0, 1.5, 2.0],
    # tgt_mult is dynamic
}

def create_random_genome():
    genome = {k: random.choice(v) for k, v in GENE_POOL.items()}
    # Forced R:R
    genome['tgt_mult'] = random.choice([genome['sl_mult'] * 1.5, genome['sl_mult'] * 2.0, genome['sl_mult'] * 3.0])
    genome['name'] = f"Gen-{random.randint(1000,9999)}"
    return genome

def crossover(parent_a, parent_b):
    child = {}
    for key in GENE_POOL.keys():
        child[key] = parent_a[key] if random.random() > 0.5 else parent_b[key]
    
    # Re-enforce R:R
    child['tgt_mult'] = random.choice([child['sl_mult'] * 1.5, child['sl_mult'] * 2.0, child['sl_mult'] * 3.0])
    child['name'] = f"Child-{random.randint(1000,9999)}"
    return child

def mutate(genome):
    if random.random() < MUTATION_RATE:
        gene = random.choice(list(GENE_POOL.keys()))
        genome[gene] = random.choice(GENE_POOL[gene])
        if gene == "sl_mult":
             genome['tgt_mult'] = random.choice([genome['sl_mult'] * 1.5, genome['sl_mult'] * 2.0, genome['sl_mult'] * 3.0])
    return genome

def fast_score(df, genome):
    wins, losses = 0, 0
    profit_points = 0.0
    
    close = df['Close'].values
    low = df['Low'].values
    high = df['High'].values
    sma200 = df['SMA200'].values
    sma50 = df['SMA50'].values
    rsi = df['RSI'].values
    adx = df['ADX'].values
    atr = df['ATR'].values
    bb_low = df['BB_LOW'].values
    macd = df['MACD'].values
    macds = df['MACD_SIG'].values
    
    i = 1; end = len(df) - 20
    while i < end:
        is_entry = False
        
        # 1. Global Filters
        if genome["trend_filter"] == "SMA200" and close[i] < sma200[i]: i += 1; continue
        if genome["trend_filter"] == "SMA50" and close[i] < sma50[i]: i += 1; continue
        if adx[i] <= genome["adx_min"]: i += 1; continue
        
        # 2. Strategy Logic
        st = genome["strategy_type"]
        if st == "RSI_DIP":
            if rsi[i] < 35: is_entry = True
        elif st == "BB_REVERSAL":
            if low[i] <= bb_low[i] and close[i] > bb_low[i]: is_entry = True
        elif st == "MACD_CROSS":
            if macd[i-1] < macds[i-1] and macd[i] > macds[i]: is_entry = True
            
        if is_entry:
            sl = close[i] - (genome["sl_mult"] * atr[i])
            tgt = close[i] + (genome["tgt_mult"] * atr[i])
            
            outcome = "OPEN"
            days = 0
            for j in range(1, 15):
                days = j
                idx = i + j
                if idx >= len(close): break
                
                if low[idx] <= sl: outcome = "LOSS"; break
                if high[idx] >= tgt: outcome = "WIN"; break
            
            cost_drag = 0.1 
            if outcome == "WIN": 
                wins += 1
                profit_points += (genome["tgt_mult"] - cost_drag)
            elif outcome == "LOSS": 
                losses += 1
                profit_points -= (genome["sl_mult"] + cost_drag)
                
            i += days
        else: i += 1
        
    return wins, losses, profit_points

def run_evolution(processed_data):
    logger.info(f"🧬 EVOLUTION START: {ITERATIONS} Strategies | {POPULATION_SIZE} Pop")
    
    if len(processed_data) > SAMPLE_SIZE:
        validation_sample = random.sample(processed_data, SAMPLE_SIZE)
    else:
        validation_sample = processed_data
    logger.info(f"🧪 Training on: {len(validation_sample)} stocks")
    
    population = [create_random_genome() for _ in range(POPULATION_SIZE)]
    best_genome = None
    best_score = -9999
    
    for gen in range(GENERATIONS):
        scores = []
        for genome in population:
            g_score = 0; g_trades = 0; g_wins = 0
            for df in validation_sample:
                w, l, s = fast_score(df, genome)
                g_score += s; g_trades += (w+l); g_wins += w
            
            wr = (g_wins / g_trades * 100) if g_trades > 0 else 0
            final_score = g_score if g_trades >= MIN_TRADES else -9999
            scores.append((genome, final_score, wr, g_trades))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[0]
        
        logger.info(f"   > Gen {gen+1}: Score {top[1]:.1f} | WR {top[2]:.1f}% | Trades {top[3]}")
        if top[1] > best_score:
            best_score = top[1]
            best_genome = top[0]
            
        elites = [s[0] for s in scores[:int(POPULATION_SIZE*0.2)]]
        new_pop = elites[:]
        while len(new_pop) < POPULATION_SIZE:
            p1 = random.choice(elites)
            p2 = random.choice(elites)
            new_pop.append(mutate(crossover(p1, p2)))
        population = new_pop

    if not best_genome:
        logger.warning("⚠️ Evolution Failed. Using Default.")
        best_genome = {"strategy_type": "RSI_DIP", "trend_filter": "SMA200", "adx_min": 15, "sl_mult": 1.5, "tgt_mult": 3.0, "name": "Default"}

    logger.info(f"\n🏆 WINNER: {best_genome['strategy_type']} (Risk: {best_genome['sl_mult']}R)")
    
    with open(STRATEGY_FILE, "w") as f:
        json.dump({"updated": datetime.utcnow().strftime("%Y-%m-%d"), "parameters": best_genome}, f, indent=2)
        
    return best_genome

# -------------------------
# 4. PORTFOLIO SIMULATION
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
        logger.info(f"📈 VALIDATING on ALL stocks...")
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
            
        profit = round(self.curve[-1] - CAPITAL, 2)
        wins = len([h for h in self.history if h['pnl'] > 0])
        total_trades = len(self.history)
        wr = round(wins/total_trades*100, 1) if total_trades > 0 else 0
        
        logger.info(f"   > Final Profit: ₹{profit} | Win Rate: {wr}%")
        return {"curve": self.curve, "win_rate": wr, "total_trades": total_trades, "profit": profit, "ledger": self.history[-50:]}

    def process_day(self, date):
        active = []
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
                rev = exit_p * t['qty']
                cost = rev * BROKERAGE_PCT
                self.cash += (rev - cost)
                pnl = rev - cost - t['cost_basis']
                self.history.append({"date": date.strftime("%Y-%m-%d"), "symbol": sym, "pnl": round(pnl, 2), "result": "WIN" if pnl > 0 else "LOSS"})
            else: active.append(t)
        self.portfolio = active
        
        if len(self.portfolio) >= MAX_POSITIONS: return
        
        for sym, df in self.data.items():
            if date not in df.index: continue
            row = df.loc[date]
            
            g = self.genome
            trend_ok = True
            if g["trend_filter"] == "SMA200" and row['Close'] < row['SMA200']: trend_ok = False
            if g["trend_filter"] == "SMA50" and row['Close'] < row['SMA50']: trend_ok = False
            if row['ADX'] <= g["adx_min"]: trend_ok = False
            
            if not trend_ok: continue

            is_entry = False
            if g["strategy_type"] == "RSI_DIP" and row['RSI'] < 35: is_entry = True
            elif g["strategy_type"] == "BB_REVERSAL" and row['Low'] <= row['BB_LOW'] and row['Close'] > row['BB_LOW']: is_entry = True
            elif g["strategy_type"] == "MACD_CROSS" and row['MACD'] > row['MACD_SIG'] and df.iloc[df.index.get_loc(date)-1]['MACD'] < df.iloc[df.index.get_loc(date)-1]['MACD_SIG']: is_entry = True
            
            if is_entry:
                if any(t['symbol'] == sym for t in self.portfolio): continue
                risk = row['ATR'] * g["sl_mult"]
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
                        "tgt": row['Close'] + (row['ATR'] * g["tgt_mult"]),
                        "cost_basis": cost + fees
                    })
                    if len(self.portfolio) >= MAX_POSITIONS: break

# -------------------------
# 5. MAIN EXECUTION
# -------------------------
if __name__ == "__main__":
    tickers = get_tickers()
    bulk = robust_download(tickers)
    processed = {}
    full_map = {}
    logger.info("📊 Preparing Data...")
    for t in tickers:
        raw = extract_df(bulk, t)
        if raw is not None and len(raw) > 250:
            clean = prepare_features(raw)
            if clean is not None:
                processed[t] = clean
                full_map[t] = clean 
                
    if not processed: exit()
    
    best_genome = run_evolution(list(processed.values()))
    
    sim = PortfolioSimulator(full_map, best_genome)
    stats = sim.run()
    
    ticker_wins = {}
    for t, df in full_map.items():
        w, l, s = fast_score(df, best_genome)
        tot = w + l
        ticker_wins[t.replace('.NS','')] = round(w/tot*100, 0) if tot > 0 else 0

    output = {
        "updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "portfolio": stats,
        "tickers": ticker_wins
    }
    
    with open(CACHE_FILE, "w") as f: json.dump(output, f)
    logger.info(f"✅ Stats Saved to {CACHE_FILE}")

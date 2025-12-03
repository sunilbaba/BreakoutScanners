"""
backtest_runner.py
THE SELF-LEARNING NIGHTLY ENGINE (Verbose Logging)
--------------------------------------------------
1. Downloads 2 Years of Data (Batch 50).
2. OPTIMIZER: Evolves 500 strategies on a 50-stock sample.
3. SIMULATOR: Runs the WINNER on ALL stocks with full logging.
4. REPORTS: Prints exact PnL, Trades, and Logic used.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import random
import json
import os
import logging
from datetime import datetime

# --- CONFIGURATION ---
DATA_PERIOD = "2y"
ITERATIONS = 500          # Strategies to breed
MIN_TRADES = 10           # Min trades to accept a strategy
SAMPLE_SIZE = 50          # Stocks to sample during evolution (Requested: 50)

# Portfolio Config
CAPITAL = 100_000.0
RISK_PER_TRADE = 0.02
MAX_POSITIONS = 5
BROKERAGE_PCT = 0.001

# Output Files
CACHE_FILE = "backtest_stats.json"
STRATEGY_FILE = "strategy_config.json"

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
    return ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "SBIN.NS", "ITC.NS", "TATAMOTORS.NS"]

def robust_download(tickers):
    logger.info(f"⬇️ Downloading {len(tickers)} stocks ({DATA_PERIOD})...")
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

# -------------------------
# 2. MATH ENGINE
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
# 3. THE "AI" OPTIMIZER
# -------------------------
def generate_genome():
    return {
        "name": f"AI-{random.randint(100,999)}",
        "trend_filter": random.choice(["SMA200", "NONE"]),
        "rsi_logic": random.choice(["OVERSOLD", "MOMENTUM"]),
        "rsi_threshold": random.choice([30, 35, 40, 55, 60, 70]),
        "adx_min": random.choice([0, 15, 20, 25]),
        "sl_mult": random.choice([1.0, 1.5, 2.0]),
        "tgt_mult": random.choice([2.0, 3.0, 4.0])
    }

def fast_score(df, genome):
    """Fast check to see if a strategy works on one stock."""
    wins, losses = 0, 0
    
    # Vector Access
    close = df['Close'].values
    low = df['Low'].values
    high = df['High'].values
    sma200 = df['SMA200'].values
    rsi = df['RSI'].values
    adx = df['ADX'].values
    atr = df['ATR'].values
    
    i = 0 
    end_idx = len(df) - 20
    
    while i < end_idx:
        is_entry = False
        
        # Trend
        if genome["trend_filter"] == "SMA200":
            if close[i] < sma200[i]: 
                i += 1; continue
        
        # ADX
        if adx[i] <= genome["adx_min"]:
            i += 1; continue

        # RSI
        rsi_val = rsi[i]
        if genome["rsi_logic"] == "OVERSOLD":
            if rsi_val < genome["rsi_threshold"]: is_entry = True
        elif genome["rsi_logic"] == "MOMENTUM":
            if rsi_val > genome["rsi_threshold"]: is_entry = True
            
        if is_entry:
            entry = close[i]
            sl = entry - (genome["sl_mult"] * atr[i])
            tgt = entry + (genome["tgt_mult"] * atr[i])
            
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
            
    return wins, losses

def run_optimizer(processed_data):
    logger.info(f"\n🧬 EVOLUTION STARTED: Testing {ITERATIONS} Strategies...")
    logger.info(f"🧪 Sample Size: {SAMPLE_SIZE} random stocks per strategy")
    
    results = []
    
    for idx in range(ITERATIONS):
        genome = generate_genome()
        t_wins, t_losses = 0, 0
        
        # Test on random sample
        sample = random.sample(processed_data, min(len(processed_data), SAMPLE_SIZE))
        
        for df in sample:
            w, l = fast_score(df, genome)
            t_wins += w
            t_losses += l
            
        total = t_wins + t_losses
        win_rate = (t_wins / total * 100) if total > 0 else 0
        
        if total >= MIN_TRADES:
            results.append({ "genome": genome, "win_rate": round(win_rate, 1), "trades": total })
        
        if idx % 100 == 0:
            best_so_far = max([r['win_rate'] for r in results]) if results else 0
            logger.info(f"... Generation {idx}: Best Win Rate Found: {best_so_far}%")

    results.sort(key=lambda x: x['win_rate'], reverse=True)
    
    if results:
        best = results[0]
        logger.info("\n🏆 WINNING STRATEGY FOUND:")
        logger.info(f"   • Name: {best['genome']['name']}")
        logger.info(f"   • Win Rate: {best['win_rate']}% (Over {best['trades']} Trades)")
        logger.info(f"   • Logic: RSI {best['genome']['rsi_logic']} {best['genome']['rsi_threshold']} | ADX>{best['genome']['adx_min']}")
        logger.info(f"   • Risk: Stop {best['genome']['sl_mult']}x | Target {best['genome']['tgt_mult']}x")
        return best['genome']
    else:
        logger.warning("⚠️ No viable strategy found. Using fallback.")
        return {
            "name": "Fallback", "trend_filter": "SMA200", "rsi_logic": "OVERSOLD",
            "rsi_threshold": 35, "adx_min": 15, "sl_mult": 1.0, "tgt_mult": 3.0
        }

# -------------------------
# 4. PORTFOLIO SIMULATION (The Validator)
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
        logger.info(f"\n📈 VALIDATION: Running 1-Year Backtest with '{self.genome['name']}'...")
        logger.info(f"💰 Initial Capital: ₹{CAPITAL}")
        
        dates = sorted(list(set().union(*[d.index for d in self.data_map.values()])))
        sim_dates = dates[200:] # Warmup
        
        for date in sim_dates:
            self.process_day(date)
            # Equity Calculation
            m2m = 0
            for t in self.portfolio:
                sym = t['symbol']
                price = self.data_map[sym].loc[date]['Close'] if date in self.data_map[sym].index else t['entry']
                m2m += (price * t['qty'])
            self.curve.append(round(self.cash + m2m, 2))
            
        # Stats
        wins = len([t for t in self.history if t['pnl'] > 0])
        total_trades = len(self.history)
        win_rate = round(wins / total_trades * 100, 1) if total_trades > 0 else 0
        final_profit = round(self.curve[-1] - CAPITAL, 2)
        
        logger.info("\n📊 FINAL RESULTS:")
        logger.info(f"   • Final Balance: ₹{round(self.curve[-1], 2)}")
        logger.info(f"   • Total Profit:  ₹{final_profit} {'(PROFIT)' if final_profit>0 else '(LOSS)'}")
        logger.info(f"   • Total Trades:  {total_trades}")
        logger.info(f"   • Win Rate:      {win_rate}%")
        
        return {
            "curve": self.curve,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "profit": final_profit,
            "ledger": self.history[-50:] 
        }

    def process_day(self, date):
        # 1. Exits
        active = []
        for t in self.portfolio:
            sym = t['symbol']
            if date not in self.data_map[sym].index:
                active.append(t); continue
            row = self.data_map[sym].loc[date]
            
            exit_p = None
            if row['Open'] < t['sl']: exit_p = row['Open'] # Gap Down
            elif row['Low'] <= t['sl']: exit_p = t['sl']
            elif row['Open'] > t['tgt']: exit_p = row['Open'] # Gap Up
            elif row['High'] >= t['tgt']: exit_p = t['tgt']
            
            if exit_p:
                pnl = (exit_p - t['entry']) * t['qty']
                cost = exit_p * t['qty'] * BROKERAGE_PCT
                self.cash += (exit_p * t['qty'] - cost)
                self.history.append({
                    "date": date.strftime("%Y-%m-%d"), "symbol": sym,
                    "pnl": round(pnl - cost - t['entry_cost'], 2),
                    "result": "WIN" if pnl > 0 else "LOSS"
                })
            else: active.append(t)
        self.portfolio = active
        
        # 2. Entries
        if len(self.portfolio) >= MAX_POSITIONS: return
        
        for sym, df in self.data_map.items():
            if date not in df.index: continue
            row = df.loc[date]
            
            # Apply Winning Genome Logic
            trend_ok = True
            if self.genome["trend_filter"] == "SMA200" and row['Close'] < row['SMA200']: trend_ok = False
            
            rsi_ok = False
            if self.genome["rsi_logic"] == "OVERSOLD" and row['RSI'] < self.genome["rsi_threshold"]: rsi_ok = True
            elif self.genome["rsi_logic"] == "MOMENTUM" and row['RSI'] > self.genome["rsi_threshold"]: rsi_ok = True
            
            adx_ok = row['ADX'] > self.genome["adx_min"]
            
            if trend_ok and rsi_ok and adx_ok:
                if any(t['symbol'] == sym for t in self.portfolio): continue
                
                risk = row['ATR'] * self.genome["sl_mult"]
                if risk <= 0: continue
                
                qty = int((self.curve[-1] * RISK_PER_TRADE) / risk)
                cost = qty * row['Close']
                
                # Affordable sizing
                if cost > self.cash: qty = int(self.cash / row['Close']); cost = qty * row['Close']
                
                if qty > 0 and self.cash > cost:
                    fees = cost * BROKERAGE_PCT
                    self.cash -= (cost + fees)
                    self.portfolio.append({
                        "symbol": sym, "entry": row['Close'], "qty": qty,
                        "sl": row['Close'] - risk, 
                        "tgt": row['Close'] + (row['ATR'] * self.genome["tgt_mult"]),
                        "entry_cost": fees
                    })
                    if len(self.portfolio) >= MAX_POSITIONS: break

# -------------------------
# 6. MAIN EXECUTION
# -------------------------
if __name__ == "__main__":
    tickers = get_tickers()
    bulk = robust_download(tickers)
    
    processed_data = {}
    for t in tickers:
        raw = extract_df(bulk, t)
        if raw is not None and len(raw) > 250:
            processed_data[t] = prepare_features(raw)
            
    if not processed_data:
        logger.error("No Data.")
        exit()
        
    # STEP 1: FIND BEST STRATEGY
    best_genome = run_optimizer(list(processed_data.values()))
    
    # STEP 2: SAVE STRATEGY
    with open(STRATEGY_FILE, "w") as f:
        json.dump({"updated": datetime.now().strftime("%Y-%m-%d"), "parameters": best_genome}, f, indent=2)
        
    # STEP 3: VALIDATE (FULL SIMULATION)
    # Run the best strategy on ALL stocks (not just sample)
    sim_engine = PortfolioSimulator(processed_data, best_genome)
    stats = sim_engine.run()
    
    # STEP 4: SAVE STATS (For Dashboard)
    ticker_wins = {}
    for t, df in processed_data.items():
        w, l = fast_score(df, best_genome)
        total = w + l
        ticker_wins[t.replace('.NS','')] = round((w/total*100),0) if total > 0 else 0

    output = {
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        "portfolio": stats,
        "tickers": ticker_wins
    }
    
    with open(CACHE_FILE, "w") as f:
        json.dump(output, f)
        
    logger.info("✅ Optimization & Backtest Complete.")

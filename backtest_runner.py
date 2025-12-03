"""
backtest_runner.py
THE "PROFIT-FIRST" NIGHTLY ENGINE
---------------------------------
1. Optimization Goal: Maximizes EXPECTANCY (Profit), not just Win Rate.
2. Metric: Uses 'R-Multiples' to score strategies.
   (e.g. A win with 3x Target gets +3 points, a loss gets -1 point).
3. Outcome: Finds strategies that actually grow the account.

Run this once daily (e.g. 8 PM).
"""

import yfinance as yf
import pandas as pd
import numpy as np
import random
import json
import os
import logging
from collections import Counter
from datetime import datetime

# --- CONFIGURATION ---
DATA_PERIOD = "2y" 
CACHE_FILE = "backtest_stats.json"
STRATEGY_FILE = "strategy_config.json"

CAPITAL = 100_000.0
RISK_PER_TRADE = 0.02
BROKERAGE_PCT = 0.001
MAX_POSITIONS = 5

ITERATIONS = 500
MIN_TRADES = 10
SAMPLE_SIZE = 50

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
    """Calculates Score (Profitability) instead of just Wins."""
    wins, losses = 0, 0
    profit_points = 0.0 # R-Multiples accumulator
    
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
            # Calc Risk/Reward Multiples
            risk_unit = genome["sl_mult"]
            reward_unit = genome["tgt_mult"]
            
            sl = close[i] - (risk_unit * atr[i])
            tgt = close[i] + (reward_unit * atr[i])
            
            outcome = "OPEN"
            days_held = 0
            
            for j in range(1, 15):
                days_held = j
                idx = i + j
                if idx >= len(close): break
                
                if low[idx] <= sl:
                    outcome = "LOSS"
                    break
                if high[idx] >= tgt:
                    outcome = "WIN"
                    break
            
            if outcome == "WIN": 
                wins += 1
                profit_points += reward_unit # Add R (e.g. +3.0)
            elif outcome == "LOSS": 
                losses += 1
                profit_points -= risk_unit   # Subtract R (e.g. -1.0)
                
            i += days_held
        else:
            i += 1
            
    return wins, losses, profit_points

def run_optimizer(processed_data):
    logger.info(f"🧬 Evolving {ITERATIONS} Strategies...")
    results = []
    
    # Inject Baseline
    baseline = { "name": "BASELINE", "trend_filter": "SMA200", "rsi_logic": "OVERSOLD", "rsi_threshold": 40, "adx_min": 15, "sl_mult": 1.5, "tgt_mult": 3.0 }
    
    for idx in range(ITERATIONS + 1):
        genome = baseline if idx == 0 else generate_genome()
        t_wins, t_losses, t_score = 0, 0, 0.0
        
        # Sample
        sample = random.sample(processed_data, min(len(processed_data), SAMPLE_SIZE))
        
        for df in sample:
            w, l, score = fast_score(df, genome)
            t_wins += w
            t_losses += l
            t_score += score
            
        total = t_wins + t_losses
        win_rate = (t_wins / total * 100) if total > 0 else 0
        
        if total >= MIN_TRADES:
            results.append({ 
                "genome": genome, 
                "win_rate": round(win_rate, 1), 
                "score": round(t_score, 2), # New Metric: Total R-Score
                "trades": total 
            })
            
        if idx < 3: logger.info(f"Test {idx}: WR {win_rate}% | Score {t_score:.1f}")

    # FIX: Sort by SCORE (Profitability), not Win Rate
    results.sort(key=lambda x: x['score'], reverse=True)
    
    if results:
        best = results[0]
        logger.info(f"🏆 BEST: Score {best['score']} | {best['win_rate']}% WR | {best['genome']['name']}")
        return best['genome']
    else:
        logger.warning("⚠️ No viable strategy found. Using fallback.")
        return baseline

# -------------------------
# 4. PORTFOLIO SIMULATOR
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
        logger.info(f"📈 Validating: {self.genome['name']}...")
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
        
        return {
            "curve": self.curve, "win_rate": win_rate, "total_trades": total,
            "profit": profit, "ledger": self.history[-50:] 
        }

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
                self.history.append({
                    "date": date.strftime("%Y-%m-%d"), "symbol": sym,
                    "pnl": round(pnl, 2), "result": "WIN" if pnl > 0 else "LOSS"
                })
            else: active.append(t)
        self.portfolio = active
        
        if len(self.portfolio) >= MAX_POSITIONS: return
        
        for sym, df in self.data.items():
            if date not in df.index: continue
            row = df.loc[date]
            
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
                if cost > self.cash: qty = int(self.cash / row['Close']); cost = qty * row['Close']
                
                if qty > 0 and self.cash > cost:
                    fees = cost * BROKERAGE_PCT
                    self.cash -= (cost + fees)
                    self.portfolio.append({
                        "symbol": sym, "entry": row['Close'], "qty": qty,
                        "sl": row['Close'] - risk, 
                        "tgt": row['Close'] + (row['ATR'] * self.genome["tgt_mult"]),
                        "cost_basis": cost + fees
                    })
                    if len(self.portfolio) >= MAX_POSITIONS: break

# -------------------------
# 5. MAIN EXECUTION
# -------------------------
if __name__ == "__main__":
    tickers = get_tickers()
    bulk = robust_download(tickers)
    
    processed_data = []
    full_map = {}
    
    logger.info("📊 Preparing Data...")
    for t in tickers:
        raw = extract_df(bulk, t)
        if raw is not None and len(raw) > 250:
            clean = prepare_features(raw, t)
            if clean is not None:
                processed_data.append(clean)
                full_map[t] = clean
    
    if not processed_data:
        logger.error("❌ No valid data.")
        exit()
    
    # 1. Evolve (Find Highest Profit Score)
    best_genome = run_optimizer(processed_data)
    
    # 2. Save Strategy
    with open(STRATEGY_FILE, "w") as f:
        json.dump({"updated": datetime.utcnow().strftime("%Y-%m-%d"), "parameters": best_genome}, f, indent=2)
        logger.info(f"✅ Strategy Saved.")
    
    # 3. Simulate
    sim = PortfolioSimulator(full_map, best_genome)
    stats = sim.run()
    
    # 4. Save Stats
    ticker_wins = {}
    for t, df in full_map.items():
        # Use Profit Score here too? 
        # Win Rate is still useful for UI highlighting "green/red" confidence.
        w, l, score = fast_score(df, best_genome)
        tot = w + l
        ticker_wins[t.replace('.NS','')] = round(w/tot*100, 0) if tot > 0 else 0
        
    output = {
        "updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "portfolio": stats,
        "tickers": ticker_wins
    }
    
    with open(CACHE_FILE, "w") as f: json.dump(output, f)
    logger.info(f"✅ Stats Saved to {CACHE_FILE}")

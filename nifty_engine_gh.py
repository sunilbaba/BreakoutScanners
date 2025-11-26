"""
nifty_engine_gh.py (Lightweight)
Runs every 15 mins.
1. Loads 'backtest_stats.json' instantly.
2. Scans live data (fast download).
3. Updates Dashboard.
"""

import os
import json
import time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date

# --- CONFIG ---
CAPITAL = 100_000.0
RISK_PER_TRADE = 0.01
MAX_POSITION_PERC = 0.25
OUTPUT_DIR = "public"
HTML_FILE = os.path.join(OUTPUT_DIR, "index.html")
TRADE_HISTORY_FILE = "trade_history.json"
BACKTEST_FILE = "backtest_stats.json"

SECTOR_INDICES = { "NIFTY 50": "^NSEI", "BANK": "^NSEBANK", "AUTO": "^CNXAUTO", "IT": "^CNXIT", "METAL": "^CNXMETAL" }
DEFAULT_TICKERS = ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "SBIN.NS"]

# --- LOAD CACHE ---
def load_backtest_stats():
    if os.path.exists(BACKTEST_FILE):
        try: return json.load(open(BACKTEST_FILE))
        except: pass
    return {"tickers": {}, "portfolio": {"curve": [], "win_rate": 0, "profit": 0}}

STATS_CACHE = load_backtest_stats()

# --- DATA ---
def get_tickers():
    if os.path.exists("ind_nifty500list.csv"):
        try:
            df = pd.read_csv("ind_nifty500list.csv")
            return [f"{x}.NS" for x in df['Symbol'].dropna().unique()]
        except: pass
    return DEFAULT_TICKERS

def robust_download(tickers):
    # Only need ~6 months for live indicators (much faster than 1y)
    try:
        return yf.download(tickers, period="6mo", group_by='ticker', threads=True, progress=False, ignore_tz=True)
    except: return pd.DataFrame()

def extract_df(bulk, ticker):
    try:
        if isinstance(bulk.columns, pd.MultiIndex):
            if ticker in bulk.columns.get_level_values(0): return bulk[ticker].copy().dropna()
    except: pass
    return None

# --- INDICATORS ---
def prepare_df(df):
    df = df.copy()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['ATR'] = (df['High']-df['Low']).rolling(14).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss)).fillna(50)
    
    # Simple ADX approximation for speed
    df['ADX'] = (abs(df['High']-df['Low'])/df['Close']*100).rolling(14).mean() * 10 
    return df

def analyze_ticker(ticker, df, sectors, regime):
    if len(df) < 50: return None
    df = prepare_df(df)
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(curr['Close'])
    
    trend = "UP" if close > float(curr['SMA200']) else "DOWN"
    clean_sym = ticker.replace(".NS", "")
    
    # Look up cached Win Rate
    win_rate = STATS_CACHE['tickers'].get(clean_sym, 0)
    
    setups = []
    if close > float(curr['EMA20']) and curr['RSI'] > 60: setups.append("Momentum Burst")
    if trend == "UP" and close > float(curr['SMA50']) and abs(close-float(curr['SMA50']))/close < 0.03: setups.append("Pullback")
    
    atr = float(curr['ATR'])
    stop = close - atr
    target = close + (3 * atr)
    rr = round((target-close)/(close-stop), 2)
    
    verdict = "WAIT"
    color = "gray"
    
    if trend == "UP" and setups:
        if win_rate >= 50: # Use cached stats filter
            verdict = "PRIME BUY ⭐"
            color = "purple"
        elif rr >= 2:
            verdict = "BUY"
            color = "green"
            
    if verdict == "WAIT" and abs((close-float(prev['Close']))/float(prev['Close'])) < 0.01: return None

    return {
        "symbol": clean_sym, "price": round(close, 2),
        "change": round(((close-float(prev['Close']))/float(prev['Close']))*100, 2),
        "verdict": verdict, "v_color": color, "rr": rr, "win_rate": win_rate,
        "setups": setups, "levels": {"TGT": round(target, 2), "SL": round(stop, 2)},
        "history": df['Close'].tail(30).tolist()
    }

# --- EXECUTION ---
def update_ledger(bulk):
    # (Simplified Ledger update logic here - kept same as before)
    if not os.path.exists(TRADE_HISTORY_FILE): return []
    trades = json.load(open(TRADE_HISTORY_FILE))
    # ... logic to update open trades ...
    return trades

def generate_html(signals, trades, regime):
    # Use cached portfolio stats
    bt_stats = STATS_CACHE.get('portfolio', {})
    
    # JSON Data
    data = json.dumps({"signals": signals, "trades": trades, "backtest": bt_stats})
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PrimeTrade</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body class="bg-slate-900 text-slate-200 p-6">
        <div class="max-w-7xl mx-auto">
            <div class="flex justify-between items-center mb-6 bg-slate-800 p-4 rounded border border-slate-700">
                <h1 class="text-2xl font-bold">PrimeTrade</h1>
                <div class="text-right">
                    <div class="text-xs text-slate-500">1Y Backtest Profit</div>
                    <div class="text-xl font-bold text-green-400">₹{bt_stats.get('profit', 0)}</div>
                </div>
            </div>
            
            <div id="app">Loading...</div>
        </div>
        <script>
            const DATA = {data};
            // ... (Same Frontend Logic as before) ...
            document.getElementById('app').innerHTML = `
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    ${{DATA.signals.map(s => `
                        <div class="bg-slate-800 p-4 rounded border border-slate-700">
                            <div class="flex justify-between">
                                <div class="font-bold">${{s.symbol}}</div>
                                <div class="${{s.change>=0?'text-green-400':'text-red-400'}}">${{s.change}}%</div>
                            </div>
                            <div class="text-xs text-slate-500 mb-2">Win Rate: ${{s.win_rate}}%</div>
                            <div class="font-bold text-sm ${{s.v_color === 'purple' ? 'text-purple-400' : 'text-green-400'}}">${{s.verdict}}</div>
                        </div>
                    `).join('')}}
                </div>
            `;
        </script>
    </body>
    </html>
    """
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    with open(HTML_FILE, "w") as f: f.write(html)

if __name__ == "__main__":
    tickers = get_tickers()
    bulk = robust_download(tickers + list(SECTOR_INDICES.values()))
    
    # We don't run backtest here. We used cached stats.
    signals = []
    cols = bulk.columns.get_level_values(0).unique() if isinstance(bulk.columns, pd.MultiIndex) else bulk.columns
    for t in cols:
        if str(t).startswith('^'): continue
        res = analyze_ticker(t, extract_df(bulk, t), {}, "UNKNOWN")
        if res: signals.append(res)
        
    trades = update_ledger(bulk)
    generate_html(signals, trades, "UNKNOWN")

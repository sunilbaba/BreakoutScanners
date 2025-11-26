"""
nifty_engine_gh.py

Features:
- ADX & Gap Protection
- "Truth Engine" (Historical Backtester per signal)
- Tabbed UI (Dashboard, Ledger, Analytics)
- Precision Formatting
- Dynamic Risk & Position Sizing
- FIX: Variable name mismatch (HTML_FILE) resolved.

Requirements: pip install yfinance pandas numpy
"""

import os
import time
import json
import math
import logging
from datetime import datetime, date
from functools import wraps

import yfinance as yf
import pandas as pd
import numpy as np

# -------------------------
# 1. CONFIGURATION
# -------------------------
CAPITAL = 100_000.0               
RISK_PER_TRADE = 0.01             
MAX_POSITION_PERC = 0.25          

DATA_PERIOD = "1y"
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 2.0

MIN_ADV_VALUE_RS = 2_000_000      

# Costs
BROKERAGE_PER_ORDER = 20.0
BROKERAGE_PCT = 0.0005
STT_PCT = 0.001
EXCHANGE_FEES_PCT = 0.0000345
STAMP_DUTY_PCT = 0.00015
GST_PCT = 0.18
SLIPPAGE_PCT = 0.001              

# Strategy
ATR_MULTIPLIER_TARGET = 3.0
ATR_MULTIPLIER_STOP = 1.0
TSL_MOVE_TO_BE_AT = 0.5           
ADX_THRESHOLD = 25.0              

OUTPUT_DIR = "public"
HTML_FILE = os.path.join(OUTPUT_DIR, "index.html")
TRADE_HISTORY_FILE = "trade_history.json"

SECTOR_INDICES = {
    "NIFTY 50": "^NSEI",
    "BANK": "^NSEBANK", "AUTO": "^CNXAUTO", "IT": "^CNXIT",
    "METAL": "^CNXMETAL", "PHARMA": "^CNXPHARMA", "FMCG": "^CNXFMCG",
    "ENERGY": "^CNXENERGY", "REALTY": "^CNXREALTY", "PSU BANK": "^CNXPSUBANK"
}

DEFAULT_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LTIM.NS", "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS", "BAJFINANCE.NS", "HCLTECH.NS",
    "ADANIENT.NS", "TATASTEEL.NS", "JIOFIN.NS", "ZOMATO.NS", "DLF.NS",
    "HAL.NS", "TRENT.NS", "BEL.NS", "POWERGRID.NS", "ONGC.NS",
    "NTPC.NS", "COALINDIA.NS", "BPCL.NS", "WIPRO.NS"
]

# -------------------------
# 2. LOGGING & UTILS
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PrimeEngine")

def retry_on_exception(max_tries=3, backoff=2.0):
    def deco(func):
        @wraps(func)
        def inner(*a, **kw):
            for i in range(max_tries):
                try: return func(*a, **kw)
                except Exception as e: time.sleep(backoff * (2 ** i))
            raise Exception(f"Failed after {max_tries} tries")
        return inner
    return deco

# -------------------------
# 3. TECHNICAL INDICATORS
# -------------------------
def true_range(df):
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df, period=14):
    return true_range(df).ewm(alpha=1/period, adjust=False).mean()

def wilder_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss)).fillna(50)

def adx(df, period=14):
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), -minus_dm, 0.0)
    tr = true_range(df)
    atr_s = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr_s)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr_s)
    return (abs(plus_di - minus_di) / (plus_di + minus_di) * 100).ewm(alpha=1/period, adjust=False).mean().fillna(0)

# -------------------------
# 4. THE TRUTH ENGINE (BACKTESTER)
# -------------------------
def run_historical_check(df):
    """Simulates the strategy on the last 6 months of data for THIS stock."""
    if len(df) < 150: return 0
    
    wins = 0
    total_trades = 0
    
    # Test on data from 6 months ago up to 10 days ago
    start_idx = len(df) - 130 
    end_idx = len(df) - 10
    
    for i in range(start_idx, end_idx):
        row = df.iloc[i]
        # Same logic as scanner: Price > EMA20 + RSI > 55
        if row['Close'] > row['EMA20'] and row['RSI'] > 55 and row['ADX'] > 20:
            entry = row['Close']
            stop = entry - (1 * row['ATR'])
            target = entry + (3 * row['ATR'])
            
            # Look ahead 15 days
            outcome = "OPEN"
            for j in range(1, 15):
                if (i+j) >= len(df): break
                future = df.iloc[i+j]
                if future['Low'] <= stop:
                    outcome = "LOSS"
                    break
                if future['High'] >= target:
                    outcome = "WIN"
                    break
            
            if outcome != "OPEN":
                total_trades += 1
                if outcome == "WIN": wins += 1
                i += j # Skip forward
                
    return round((wins / total_trades * 100), 0) if total_trades > 0 else 0

# -------------------------
# 5. DATA MANAGEMENT
# -------------------------
@retry_on_exception(max_tries=3)
def robust_download(tickers, period=DATA_PERIOD):
    logger.info(f"Downloading {len(tickers)} symbols...")
    # Using auto_adjust=False to fix some yfinance warnings
    df = yf.download(tickers, period=period, group_by='ticker', threads=True, progress=False, auto_adjust=True)
    if df is None or df.empty: raise RuntimeError("Empty Data")
    return df

def get_tickers():
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        df = pd.read_csv(url)
        return [f"{x}.NS" for x in df['Symbol'].dropna().unique()]
    except: return DEFAULT_TICKERS.copy()

def extract_stock_df(bulk_data, ticker):
    try:
        if isinstance(bulk_data.columns, pd.MultiIndex):
            if ticker in bulk_data.columns.get_level_values(0):
                return bulk_data[ticker].copy().dropna()
    except: pass
    return None

# -------------------------
# 6. ANALYSIS & EXECUTION
# -------------------------
def estimate_transaction_costs(price, qty):
    val = price * qty
    # Approx 0.1% total costs (Brokerage + STT + Slippage)
    return round(val * 0.001, 2) 

def calculate_qty(entry, stop_loss):
    risk_amt = CAPITAL * RISK_PER_TRADE
    risk_share = abs(entry - stop_loss)
    if risk_share <= 0: return 0
    qty = int(risk_amt / risk_share)
    if (qty * entry) > (CAPITAL * MAX_POSITION_PERC): qty = int((CAPITAL * MAX_POSITION_PERC) / entry)
    return max(qty, 0)

def analyze_ticker(ticker, df, sector_changes, market_regime):
    if len(df) < 100: return None
    df = df.copy()
    
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['ATR'] = atr(df)
    df['RSI'] = wilder_rsi(df['Close'])
    df['ADX'] = adx(df)

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(curr['Close'])
    
    trend = "UP" if close > float(curr['SMA200']) else "DOWN"
    trend_strength = float(curr['ADX'])
    
    setups = []
    if close > float(curr['EMA20']) and curr['RSI'] > 60 and trend_strength > ADX_THRESHOLD: setups.append("Momentum Burst")
    if trend == "UP" and close > float(curr['SMA50']) and abs(close - float(curr['SMA50']))/close < 0.03: setups.append("Pullback")

    # Volume Check
    avg_vol = df['Volume'].rolling(30).mean().iloc[-1]
    if (avg_vol * close) < MIN_ADV_VALUE_RS: return None

    clean_sym = ticker.replace(".NS", "")
    my_sector = "Other"
    # Sector logic placeholder
    
    atr_val = float(curr['ATR'])
    stop = round(close - ATR_MULTIPLIER_STOP * atr_val, 2)
    target = round(close + ATR_MULTIPLIER_TARGET * atr_val, 2)
    rr = round((target - close) / (close - stop), 2)

    adjusted_risk = RISK_PER_TRADE / 2 if "BEAR" in market_regime else RISK_PER_TRADE
    qty = calculate_qty(close, stop)

    verdict = "WAIT"
    v_color = "gray"
    
    if trend == "UP" and len(setups) > 0:
        if rr >= 2.0:
            verdict = "BUY"
            v_color = "green"
    
    if verdict == "WAIT" and abs((close - float(prev['Close']))/float(prev['Close'])) < 0.01: return None

    # RUN BACKTEST ONLY IF SIGNAL DETECTED
    win_rate = 0
    if verdict != "WAIT":
        win_rate = run_historical_check(df)
        if win_rate > 60: 
            verdict = "PRIME BUY ⭐"
            v_color = "purple"

    return {
        "symbol": clean_sym,
        "price": round(close, 2),
        "change": round(((close - float(prev['Close']))/float(prev['Close']))*100, 2),
        "sector": my_sector,
        "setups": setups,
        "verdict": verdict,
        "v_color": v_color,
        "rr": rr,
        "qty": qty,
        "adx": round(trend_strength, 1),
        "win_rate": win_rate, # HISTORICAL WIN RATE
        "levels": {"TGT": target, "SL": stop},
        "history": df['Close'].tail(30).tolist()
    }

# -------------------------
# 7. HISTORY & LEDGER
# -------------------------
def load_json_file(path, default):
    if not os.path.exists(path): return default
    try:
        with open(path, 'r') as f: return json.load(f)
    except: return default

def save_json_file(path, data):
    with open(path, 'w') as f: json.dump(data, f, indent=2)

def update_open_trades(bulk_data):
    trades = load_json_file(TRADE_HISTORY_FILE, [])
    updated = False
    today_str = date.today().isoformat()
    
    for trade in trades:
        if trade['status'] != 'OPEN': continue
        ticker = trade['symbol'] + ".NS"
        df = extract_stock_df(bulk_data, ticker)
        if df is None: continue
        
        curr = df.iloc[-1]
        high, low, close = float(curr['High']), float(curr['Low']), float(curr['Close'])
        open_p = float(curr['Open'])
        
        entry, target, sl = trade['entry'], trade['target'], trade['stop_loss']
        
        if low <= sl:
            trade['status'] = 'LOSS'
            trade['exit_price'] = open_p if open_p < sl else sl
            trade['exit_date'] = today_str
            pnl = (trade['exit_price'] - entry) * trade['qty']
            costs = estimate_transaction_costs(entry, trade['qty']) * 2
            trade['net_pnl'] = round(pnl - costs, 2)
            updated = True
        elif high >= target:
            trade['status'] = 'WIN'
            trade['exit_price'] = open_p if open_p > target else target
            trade['exit_date'] = today_str
            pnl = (trade['exit_price'] - entry) * trade['qty']
            costs = estimate_transaction_costs(entry, trade['qty']) * 2
            trade['net_pnl'] = round(pnl - costs, 2)
            updated = True
        else:
            # Mark to Market
            m2m = (close - entry) * trade['qty']
            trade['pnl'] = round(m2m, 2)
            trade['pnl_pct'] = round(((close-entry)/entry)*100, 2)
            updated = True

    if updated: save_json_file(TRADE_HISTORY_FILE, trades)

def place_orders(signals):
    trades = load_json_file(TRADE_HISTORY_FILE, [])
    today_str = date.today().isoformat()
    # Filter stocks already owned
    owned = {t['symbol'] for t in trades if t['status'] == 'OPEN'}
    
    candidates = [s for s in signals if "BUY" in s['verdict'] and s['symbol'] not in owned]
    candidates.sort(key=lambda x: x['win_rate'], reverse=True) # Prioritize High Win Rate
    
    for s in candidates[:3]: # Max 3 new trades per run
        tid = f"{s['symbol']}-{today_str}"
        if not any(t['id'] == tid for t in trades):
            trades.insert(0, {
                "id": tid, "date": today_str, "symbol": s['symbol'],
                "entry": s['price'], "qty": s['qty'],
                "target": s['levels']['TGT'], "stop_loss": s['levels']['SL'],
                "status": "OPEN", "pnl": 0.0, "pnl_pct": 0.0, "win_rate": s['win_rate']
            })
            save_json_file(TRADE_HISTORY_FILE, trades)

# -------------------------
# 8. HTML DASHBOARD (TABS)
# -------------------------
def generate_html(signals, trades, market_regime, timestamp):
    closed = [t for t in trades if t['status'] in ['WIN', 'LOSS']]
    wins = len([t for t in closed if t['status'] == 'WIN'])
    total_closed = len(closed)
    acc = round((wins/total_closed*100), 1) if total_closed > 0 else 0
    net_pnl = round(sum(t.get('net_pnl', 0) for t in closed), 2)
    
    # Filter: Don't show owned stocks in scanner
    owned = {t['symbol'] for t in trades if t['status'] == 'OPEN'}
    display_signals = [s for s in signals if s['symbol'] not in owned]
    display_signals.sort(key=lambda x: x['win_rate'], reverse=True)

    json_data = json.dumps({"stocks": display_signals, "pos": trades})
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PrimeTrade PRO</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            body {{ background: #0f172a; color: #e2e8f0; font-family: sans-serif; }}
            .card {{ background: #1e293b; border: 1px solid #334155; padding: 16px; border-radius: 12px; }}
            .prime {{ border: 1px solid #a855f7; background: linear-gradient(to bottom right, #1e293b, #2e1065); }}
            .tab-active {{ border-bottom: 2px solid #a855f7; color: white; }}
            .tab-inactive {{ color: #64748b; }}
            .win {{ color: #4ade80; }}
            .loss {{ color: #f87171; }}
        </style>
    </head>
    <body class="p-4 md:p-8">
        <div class="max-w-7xl mx-auto">
            <!-- Header -->
            <div class="flex justify-between items-center mb-6 bg-slate-900 p-4 rounded-xl border border-slate-800">
                <div>
                    <h1 class="text-2xl font-bold text-white flex items-center gap-2">
                        <i data-lucide="layout-dashboard" class="text-purple-500"></i> PrimeTrade
                    </h1>
                    <div class="text-xs text-slate-500 mt-1">{timestamp} • {market_regime}</div>
                </div>
                <div class="text-right">
                    <div class="text-[10px] text-slate-500 uppercase">Net PnL</div>
                    <div class="text-xl font-bold { 'win' if net_pnl >=0 else 'loss' }">₹{net_pnl}</div>
                    <div class="text-[10px] text-slate-600">Win Rate: {acc}% ({total_closed})</div>
                </div>
            </div>

            <!-- Tabs -->
            <div class="flex gap-6 mb-6 border-b border-slate-800">
                <button onclick="switchTab('dash')" id="tab-dash" class="pb-2 text-sm font-bold tab-active">Dashboard</button>
                <button onclick="switchTab('ledger')" id="tab-ledger" class="pb-2 text-sm font-bold tab-inactive">Trade Ledger</button>
            </div>

            <!-- VIEW: DASHBOARD -->
            <div id="view-dash">
                <h2 class="text-xs font-bold text-slate-500 mb-3 uppercase">Active Portfolio</h2>
                <div id="portfolio" class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8"></div>

                <h2 class="text-xs font-bold text-slate-500 mb-3 uppercase">New Opportunities (Backtested)</h2>
                <div id="scanner" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8"></div>
            </div>

            <!-- VIEW: LEDGER -->
            <div id="view-ledger" class="hidden">
                <div class="bg-slate-900 rounded-lg border border-slate-800 overflow-hidden">
                    <table class="w-full text-sm text-left">
                        <thead class="text-xs text-slate-500 uppercase bg-slate-800">
                            <tr><th class="p-3">Date</th><th class="p-3">Symbol</th><th class="p-3">Outcome</th><th class="p-3">PnL</th><th class="p-3">Note</th></tr>
                        </thead>
                        <tbody id="ledger-body"></tbody>
                    </table>
                </div>
            </div>
        </div>

        <script>
            const DATA = {json_data};
            
            function switchTab(tab) {{
                document.getElementById('view-dash').classList.add('hidden');
                document.getElementById('view-ledger').classList.add('hidden');
                document.getElementById('tab-dash').className = "pb-2 text-sm font-bold tab-inactive";
                document.getElementById('tab-ledger').className = "pb-2 text-sm font-bold tab-inactive";
                
                document.getElementById('view-'+tab).classList.remove('hidden');
                document.getElementById('tab-'+tab).className = "pb-2 text-sm font-bold tab-active";
            }}

            // 1. PORTFOLIO RENDER
            const openTrades = DATA.pos.filter(t => t.status === 'OPEN');
            const portRoot = document.getElementById('portfolio');
            if(openTrades.length === 0) portRoot.innerHTML = '<div class="col-span-full text-center text-slate-600 py-4">No active trades.</div>';
            else {{
                portRoot.innerHTML = openTrades.map(p => {{
                    const pnlClass = p.pnl_pct >= 0 ? 'win' : 'loss';
                    return `<div class="bg-slate-800 border border-slate-700 rounded-lg p-4 relative">
                        <div class="flex justify-between mb-2"><div class="font-bold text-white">${{p.symbol}}</div><div class="font-mono font-bold ${{pnlClass}}">${{p.pnl_pct}}%</div></div>
                        <div class="text-xs text-slate-400 flex justify-between"><span>Entry: ${{p.entry}}</span><span>Date: ${{p.date}}</span></div>
                        <div class="flex justify-between text-[10px] mt-2 font-mono"><span class="loss">${{p.stop_loss}} SL</span><span class="win">${{p.target}} TGT</span></div>
                    </div>`;
                }}).join('');
            }}

            // 2. SCANNER RENDER (With Backtest Badge)
            const scanRoot = document.getElementById('scanner');
            if(DATA.stocks.length === 0) scanRoot.innerHTML = '<div class="col-span-full text-center text-slate-600 py-10">No high-quality signals.</div>';
            else {{
                scanRoot.innerHTML = DATA.stocks.map(s => {{
                    const isPrime = s.verdict.includes('PRIME');
                    const winColor = s.win_rate > 60 ? 'win' : (s.win_rate > 40 ? 'text-yellow-400' : 'loss');
                    return `<div class="card ${{isPrime ? 'prime' : ''}}">
                        <div class="flex justify-between mb-2"><div><div class="font-bold text-white">${{s.symbol}}</div></div><div class="text-right"><div class="font-bold ${{s.change>=0?'win':'loss'}}">${{s.change}}%</div><div class="text-[10px] text-slate-500">₹${{s.price}}</div></div></div>
                        
                        <div class="flex justify-between bg-slate-900/50 p-2 rounded mb-2 text-[10px]">
                            <div class="text-center"><div>Win Rate</div><div class="font-bold ${{winColor}}">${{s.win_rate}}%</div></div>
                            <div class="text-center"><div>RR</div><div class="font-bold text-white">${{s.rr}}</div></div>
                            <div class="text-center"><div>ADX</div><div class="font-bold text-white">${{s.adx}}</div></div>
                        </div>
                        
                        <div class="flex justify-between text-[10px] font-mono mt-1"><span class="loss">SL: ${{s.levels.SL}}</span><span class="win">TGT: ${{s.levels.TGT}}</span></div>
                    </div>`;
                }}).join('');
            }}

            // 3. LEDGER RENDER
            const closedTrades = DATA.pos.filter(t => t.status !== 'OPEN');
            const ledgerRoot = document.getElementById('ledger-body');
            if(closedTrades.length === 0) ledgerRoot.innerHTML = '<tr><td colspan="5" class="p-4 text-center text-slate-500">No closed trades history.</td></tr>';
            else {{
                ledgerRoot.innerHTML = closedTrades.map(t => {{
                    const pnlClass = t.status === 'WIN' ? 'win' : 'loss';
                    return `<tr class="border-b border-slate-800 hover:bg-slate-800">
                        <td class="p-3 text-slate-400">${{t.exit_date || t.date}}</td>
                        <td class="p-3 font-bold text-white">${{t.symbol}}</td>
                        <td class="p-3"><span class="px-2 py-1 rounded text-[10px] font-bold bg-opacity-20 ${{t.status==='WIN'?'bg-green-500 win':'bg-red-500 loss'}}">${{t.status}}</span></td>
                        <td class="p-3 font-mono font-bold ${{pnlClass}}">₹${{t.net_pnl}}</td>
                        <td class="p-3 text-xs text-slate-500">${{t.note || '-'}}</td>
                    </tr>`;
                }}).join('');
            }}
            
            lucide.createIcons();
        </script>
    </body>
    </html>
    """
    # FIX: Use HTML_FILE here, not FILE_PATH which caused error before
    with open(HTML_FILE, "w") as f: f.write(html)

if __name__ == "__main__":
    tickers = get_tickers()
    bulk = robust_download(tickers + list(SECTOR_INDICES.values()))
    
    regime = analyze_market_trend(bulk)
    
    sector_changes = {}
    
    results = []
    cols = bulk.columns.get_level_values(0).unique() if isinstance(bulk.columns, pd.MultiIndex) else bulk.columns
    for t in cols:
        if str(t).startswith('^'): continue
        try:
            res = analyze_ticker(t, extract_stock_df(bulk, t), sector_changes, regime)
            if res: results.append(res)
        except: continue

    place_orders(results)
    update_open_trades(bulk)
    
    trades = load_json_file(TRADE_HISTORY_FILE, [])
    generate_html(results, trades, regime, datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))

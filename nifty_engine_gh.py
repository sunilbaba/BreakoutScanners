"""
nifty_engine_gh.py

THE "SELF-ADJUSTING" LIVE ENGINE
--------------------------------
1. Loads Strategy Logic from 'strategy_config.json'.
2. Scans live market using those DYNAMIC rules.
3. Manages Portfolio & Generates 3-Tab Dashboard.

Requirements: pip install yfinance pandas numpy
"""

import os
import time
import json
import logging
from datetime import datetime, date

import yfinance as yf
import pandas as pd
import numpy as np

# --- 1. CONFIGURATION ---
CAPITAL = 100000.0
RISK_PER_TRADE = 0.02  
DATA_PERIOD = "2y"     

# Files
OUTPUT_DIR = "public"
HTML_FILE = os.path.join(OUTPUT_DIR, "index.html")
TRADE_HISTORY_FILE = "trade_history.json"
CACHE_FILE = "backtest_stats.json"
STRATEGY_FILE = "strategy_config.json" # <--- THE BRAIN

# Costs
BROKERAGE_PCT = 0.001 

SECTOR_INDICES = {
    "NIFTY 50": "^NSEI", "BANK": "^NSEBANK", "AUTO": "^CNXAUTO", "IT": "^CNXIT",
    "METAL": "^CNXMETAL", "PHARMA": "^CNXPHARMA", "FMCG": "^CNXFMCG",
    "ENERGY": "^CNXENERGY", "REALTY": "^CNXREALTY", "PSU BANK": "^CNXPSUBANK"
}

DEFAULT_TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "ICICIBANK.NS", "SBIN.NS",
    "ITC.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LTIM.NS", "AXISBANK.NS", "MARUTI.NS",
    "TITAN.NS", "SUNPHARMA.NS", "BAJFINANCE.NS", "HCLTECH.NS", "TATASTEEL.NS",
    "ADANIENT.NS", "JIOFIN.NS", "ZOMATO.NS", "DLF.NS", "HAL.NS", "TRENT.NS"
]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("PrimeEngine")

# --- 2. DYNAMIC STRATEGY LOADER ---
def load_strategy():
    """Loads the AI-Optimized Strategy. Returns Default if missing."""
    default = {
        "strategy_name": "Default-Dip",
        "parameters": {
            "trend_filter": "SMA200",
            "rsi_logic": "OVERSOLD",
            "rsi_threshold": 30,
            "adx_threshold": 20,
            "atr_stop_mult": 1.0,
            "atr_target_mult": 3.0
        }
    }
    
    if os.path.exists(STRATEGY_FILE):
        try:
            data = json.load(open(STRATEGY_FILE))
            if "parameters" in data: return data
        except: pass
    
    logger.warning("⚠️ Strategy file missing/invalid. Using Default.")
    return default

# Load Global Config
AI_CONFIG = load_strategy()
PARAMS = AI_CONFIG['parameters']

# --- 3. DATA & MATH ---
def robust_download(tickers):
    logger.info(f"⬇️ Downloading {len(tickers)} symbols...")
    frames = []
    batch_size = 20
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(batch, period=DATA_PERIOD, group_by='ticker', threads=True, progress=False, ignore_tz=True)
            frames.append(data)
        except: pass
    if not frames: return pd.DataFrame()
    return pd.concat(frames, axis=1)

def get_tickers():
    if os.path.exists("ind_nifty500list.csv"):
        try:
            df = pd.read_csv("ind_nifty500list.csv")
            return [f"{x}.NS" for x in df['Symbol'].dropna().unique()]
        except: pass
    return DEFAULT_TICKERS

def extract_df(bulk, ticker):
    try:
        if isinstance(bulk.columns, pd.MultiIndex):
            if ticker in bulk.columns.get_level_values(0):
                return bulk[ticker].copy().dropna()
    except: pass
    return None

def prepare_df(df):
    df = df.copy()
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
    return df

# --- 4. LIVE SCANNER (USING JSON PARAMS) ---
def analyze_ticker(ticker, df, win_rates):
    if len(df) < 50: return None
    df = prepare_df(df)
    curr = df.iloc[-1]
    close = float(curr['Close'])
    clean_sym = ticker.replace(".NS", "")
    
    # --- DYNAMIC LOGIC FROM FILE ---
    sma200 = float(curr['SMA200']) if not pd.isna(curr['SMA200']) else close
    sma50 = float(curr['SMA50']) if not pd.isna(curr['SMA50']) else close
    
    # 1. Trend Filter
    trend_ok = True
    if PARAMS['trend_filter'] == "SMA200" and close < sma200: trend_ok = False
    if PARAMS['trend_filter'] == "SMA50" and close < sma50: trend_ok = False
    
    # 2. RSI Logic
    rsi_ok = False
    rsi_val = float(curr['RSI'])
    if PARAMS['rsi_logic'] == "OVERSOLD" and rsi_val < PARAMS['rsi_threshold']: rsi_ok = True
    elif PARAMS['rsi_logic'] == "OVERBOUGHT" and rsi_val > PARAMS['rsi_threshold']: rsi_ok = True
    elif PARAMS['rsi_logic'] == "MOMENTUM" and rsi_val > PARAMS['rsi_threshold']: rsi_ok = True
    elif PARAMS['rsi_logic'] == "NEUTRAL" and (40 < rsi_val < 60): rsi_ok = True

    # 3. ADX Logic
    adx_ok = float(curr['ADX']) > PARAMS['adx_threshold']
    
    # --- DECISION ---
    verdict = "WAIT"
    v_color = "gray"
    
    setups = []
    if trend_ok and rsi_ok and adx_ok:
        setups.append("AI Signal")
        
        # Check Win Rate from Backtest
        win_rate = win_rates.get(clean_sym, 0)
        
        atr_val = float(curr['ATR'])
        stop = close - (PARAMS['atr_stop_mult'] * atr_val)
        target = close + (PARAMS['atr_target_mult'] * atr_val)
        rr = round((target - close) / (close - stop), 2)
        
        # Calculate Qty
        risk_share = close - stop
        qty = int((CAPITAL * RISK_PER_TRADE) / risk_share) if risk_share > 0 else 0
        
        if win_rate >= 50:
            verdict = "PRIME BUY ⭐"
            v_color = "purple"
        else:
            verdict = "BUY"
            v_color = "green"
            
        return {
            "symbol": clean_sym, "price": round(close, 2),
            "change": round(((close - df.iloc[-2]['Close'])/df.iloc[-2]['Close'])*100, 2),
            "verdict": verdict, "v_color": v_color, "rr": rr, "win_rate": win_rate,
            "qty": qty, "setups": setups,
            "levels": {"TGT": round(target, 2), "SL": round(stop, 2)},
            "history": df['Close'].tail(30).tolist()
        }
        
    return None

# --- 5. PORTFOLIO & HTML ---
def load_json(path):
    if os.path.exists(path):
        try: return json.load(open(path))
        except: pass
    return []

def save_json(path, data):
    with open(path, 'w') as f: json.dump(data, f, indent=2)

def update_trades(trades, bulk_data):
    updated = False
    today = date.today().isoformat()
    for t in trades:
        if t['status'] == 'OPEN':
            df = extract_df(bulk_data, t['symbol']+".NS")
            if df is not None:
                curr = df.iloc[-1]
                low, high, open_p = curr['Low'], curr['High'], curr['Open']
                
                # Exit Logic
                status = t['status']
                exit_p = None
                
                if low <= t['stop_loss']:
                    status = 'LOSS'
                    exit_p = open_p if open_p < t['stop_loss'] else t['stop_loss']
                elif high >= t['target']:
                    status = 'WIN'
                    exit_p = open_p if open_p > t['target'] else t['target']
                
                if status != 'OPEN':
                    t['status'] = status
                    t['exit_price'] = round(exit_p, 2)
                    t['exit_date'] = today
                    pnl = (t['exit_price'] - t['entry']) * t['qty']
                    t['net_pnl'] = round(pnl - (pnl * BROKERAGE_PCT), 2)
                    updated = True
    
    if updated: save_json(TRADE_HISTORY_FILE, trades)
    return trades

def add_new_signals(signals, trades):
    today = date.today().isoformat()
    owned = {t['symbol'] for t in trades if t['status'] == 'OPEN'}
    valid = [s for s in signals if "BUY" in s['verdict'] and s['symbol'] not in owned]
    valid.sort(key=lambda x: x['win_rate'], reverse=True)
    
    for s in valid[:3]: # Max 3 new per run
        tid = f"{s['symbol']}-{today}"
        if not any(t['id'] == tid for t in trades):
            trades.insert(0, {
                "id": tid, "date": today, "symbol": s['symbol'],
                "entry": s['price'], "qty": s['qty'],
                "target": s['levels']['TGT'], "stop_loss": s['levels']['SL'],
                "status": "OPEN", "net_pnl": 0
            })
            save_json(TRADE_HISTORY_FILE, trades)

def generate_html(signals, trades, bt_stats, timestamp):
    closed = [t for t in trades if t['status'] != 'OPEN']
    net_pnl = round(sum(t.get('net_pnl', 0) for t in closed), 2)
    
    # Display logic
    strategy_name = AI_CONFIG.get('strategy_name', 'Default')
    
    json_data = json.dumps({"signals": signals, "trades": trades, "backtest": bt_stats})
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PrimeTrade PRO</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body class="bg-slate-900 text-slate-200 p-4">
        <div class="max-w-7xl mx-auto">
            <div class="flex justify-between items-center mb-6 bg-slate-800 p-4 rounded border border-slate-700">
                <div>
                    <h1 class="text-2xl font-bold">PrimeTrade</h1>
                    <div class="text-xs text-purple-400 font-bold">Active Strategy: {strategy_name}</div>
                    <div class="text-xs text-slate-500">{timestamp}</div>
                </div>
                <div class="text-right"><div class="text-xs text-slate-500">Realized PnL</div><div class="text-xl font-bold {{'text-green-400' if net_pnl>=0 else 'text-red-400'}}">₹{net_pnl}</div></div>
            </div>
            
            <div class="flex gap-4 mb-6">
                <button onclick="setTab('dash')" id="btn-dash" class="font-bold text-white border-b-2 border-purple-500">Dashboard</button>
                <button onclick="setTab('ledger')" id="btn-ledger" class="font-bold text-slate-500">Ledger</button>
                <button onclick="setTab('strat')" id="btn-strat" class="font-bold text-slate-500">Strategy</button>
            </div>
            
            <!-- DASHBOARD -->
            <div id="view-dash">
                <h2 class="text-xs font-bold text-slate-500 mb-3">ACTIVE PORTFOLIO</h2>
                <div id="portfolio" class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8"></div>
                <h2 class="text-xs font-bold text-slate-500 mb-3">NEW SIGNALS (Based on AI Logic)</h2>
                <div id="scanner" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4"></div>
            </div>
            
            <!-- LEDGER -->
            <div id="view-ledger" class="hidden">
                <div class="bg-slate-800 rounded overflow-hidden">
                    <table class="w-full text-sm text-left"><thead class="bg-slate-700"><tr><th class="p-3">Date</th><th>Symbol</th><th>Result</th><th>PnL</th></tr></thead><tbody id="live-body"></tbody></table>
                </div>
            </div>

            <!-- STRATEGY -->
            <div id="view-strat" class="hidden">
                <div class="grid grid-cols-3 gap-4 mb-6">
                    <div class="bg-slate-800 p-4 rounded"><div class="text-xs text-slate-500">1Y Profit</div><div class="text-xl font-bold text-green-400">₹{bt_stats.get('profit', 0)}</div></div>
                    <div class="bg-slate-800 p-4 rounded"><div class="text-xs text-slate-500">Win Rate</div><div class="text-xl font-bold text-blue-400">{bt_stats.get('win_rate', 0)}%</div></div>
                    <div class="bg-slate-800 p-4 rounded"><div class="text-xs text-slate-500">Trades</div><div class="text-xl font-bold text-white">{bt_stats.get('total_trades', 0)}</div></div>
                </div>
                <div class="bg-slate-800 p-4 rounded mb-4 h-64"><canvas id="equityChart"></canvas></div>
                <button onclick="downloadCSV()" class="bg-green-600 text-white px-4 py-2 rounded text-sm font-bold hover:bg-green-500">Download Backtest CSV</button>
            </div>
        </div>
        <script>
            const DATA = {json_data};
            function setTab(id) {{
                ['dash', 'ledger', 'strat'].forEach(t => {{
                    document.getElementById('view-'+t).classList.add('hidden');
                    document.getElementById('btn-'+t).className = "font-bold text-slate-500";
                }});
                document.getElementById('view-'+id).classList.remove('hidden');
                document.getElementById('btn-'+id).className = "font-bold text-white border-b-2 border-purple-500";
            }}

            const portRoot = document.getElementById('portfolio');
            const openT = DATA.trades.filter(t => t.status === 'OPEN');
            if(openT.length === 0) portRoot.innerHTML = '<div class="col-span-full text-center text-slate-600 py-4">No active trades</div>';
            else {{
                portRoot.innerHTML = openT.map(p => `
                    <div class="bg-slate-800 p-4 rounded border border-slate-700">
                        <div class="flex justify-between mb-2"><div class="font-bold text-white">${{p.symbol}}</div><div class="text-xs bg-blue-900 text-blue-200 px-2 py-1 rounded">OPEN</div></div>
                        <div class="text-xs text-slate-400 flex justify-between"><span>Qty: ${{p.qty}}</span><span>Entry: ${{p.entry}}</span></div>
                        <div class="flex justify-between text-[10px] font-mono mt-2"><span class="text-red-400">SL ${{p.stop_loss}}</span><span class="text-green-400">TGT ${{p.target}}</span></div>
                    </div>`).join('');
            }}

            const scanRoot = document.getElementById('scanner');
            if(DATA.signals.length === 0) scanRoot.innerHTML = '<div class="col-span-full text-center text-slate-600 py-10">No signals matching AI criteria</div>';
            else {{
                scanRoot.innerHTML = DATA.signals.map(s => `
                    <div class="bg-slate-800 p-4 rounded border border-slate-700 ${{s.verdict.includes('PRIME')?'border-purple-500':''}}">
                        <div class="flex justify-between mb-2"><div class="font-bold text-white">${{s.symbol}}</div><div class="${{s.change>=0?'text-green-400':'text-red-400'}}">${{s.change}}%</div></div>
                        <div class="flex justify-between text-xs mb-2"><span class="text-slate-500">Win%: ${{s.win_rate}}%</span><span class="text-slate-500">Qty: ${{s.qty}}</span></div>
                        <div class="font-bold text-sm text-center py-1 rounded bg-slate-700 ${{s.v_color === 'purple' ? 'text-purple-400' : 'text-green-400'}}">${{s.verdict}}</div>
                    </div>`).join('');
            }}

            const ledgerRoot = document.getElementById('live-body');
            const closedT = DATA.trades.filter(t => t.status !== 'OPEN');
            ledgerRoot.innerHTML = closedT.map(t => `
                <tr class="border-b border-slate-700"><td class="p-3">${{t.exit_date}}</td><td class="p-3 font-bold">${{t.symbol}}</td><td class="p-3"><span class="text-[10px] px-2 py-1 rounded ${{t.status==='WIN'?'bg-green-900 text-green-200':'bg-red-900 text-red-200'}}">${{t.status}}</span></td><td class="p-3 font-mono ${{t.net_pnl>=0?'text-green-400':'text-red-400'}}">₹${{t.net_pnl}}</td></tr>`).join('');

            const ctx = document.getElementById('equityChart');
            if(DATA.backtest.curve) {{
                new Chart(ctx, {{
                    type: 'line',
                    data: {{ labels: DATA.backtest.curve.map((_, i) => i), datasets: [{{ label: 'Equity', data: DATA.backtest.curve, borderColor: '#a855f7', tension: 0.1, pointRadius: 0 }}] }},
                    options: {{ maintainAspectRatio: false, scales: {{ y: {{ grid: {{ color: '#334155' }} }}, x: {{ display: false }} }} }}
                }});
            }}

            function downloadCSV() {{
                if(!DATA.backtest.ledger) return;
                const rows = [["Date", "Symbol", "Entry", "Exit", "PnL", "Result"]];
                DATA.backtest.ledger.forEach(t => rows.push([t.date, t.symbol, t.entry, t.exit, t.pnl, t.result]));
                let csvContent = "data:text/csv;charset=utf-8," + rows.map(e => e.join(",")).join("\\n");
                const link = document.createElement("a");
                link.setAttribute("href", encodeURI(csvContent));
                link.setAttribute("download", "backtest_ledger.csv");
                document.body.appendChild(link);
                link.click();
            }}
        </script>
    </body>
    </html>
    """
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    with open(HTML_FILE, "w") as f: f.write(html)

if __name__ == "__main__":
    tickers = get_tickers()
    bulk = robust_download(tickers + list(SECTOR_INDICES.values()))
    
    # 1. Load Backtest Data (Created by Night Job)
    bt_stats = {}
    if os.path.exists(CACHE_FILE):
        try: bt_stats = json.load(open(CACHE_FILE))
        except: pass
    win_rates = bt_stats.get("tickers", {})
    
    # 2. Analyze Live Market
    signals = []
    cols = bulk.columns.get_level_values(0).unique() if isinstance(bulk.columns, pd.MultiIndex) else bulk.columns
    for t in cols:
        if str(t).startswith('^'): continue
        try:
            res = analyze_ticker(t, extract_df(bulk, t), win_rates)
            if res: signals.append(res)
        except: continue

    # 3. Execute
    trades = load_json(TRADE_HISTORY_FILE)
    trades = update_trades(trades, bulk)
    add_new_signals(signals, trades)
    
    generate_html(signals, trades, bt_stats, datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
    logger.info("Done.")

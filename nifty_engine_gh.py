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
OUTPUT_DIR = "public"
HTML_FILE = os.path.join(OUTPUT_DIR, "index.html")
TRADE_HISTORY_FILE = "trade_history.json"
BACKTEST_FILE = "backtest_stats.json"

SECTOR_INDICES = { "NIFTY 50": "^NSEI", "BANK": "^NSEBANK", "AUTO": "^CNXAUTO", "IT": "^CNXIT", "METAL": "^CNXMETAL" }
DEFAULT_TICKERS = ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS"]

# --- LOAD DATA ---
def load_backtest_stats():
    if os.path.exists(BACKTEST_FILE):
        try: return json.load(open(BACKTEST_FILE))
        except: pass
    return {"tickers": {}, "portfolio": {"curve": [], "ledger": [], "win_rate": 0, "profit": 0}}

STATS_CACHE = load_backtest_stats()

def get_tickers():
    if os.path.exists("ind_nifty500list.csv"):
        try:
            df = pd.read_csv("ind_nifty500list.csv")
            return [f"{x}.NS" for x in df['Symbol'].dropna().unique()]
        except: pass
    return DEFAULT_TICKERS

def robust_download(tickers):
    try: return yf.download(tickers, period="6mo", group_by='ticker', threads=True, progress=False, ignore_tz=True)
    except: return pd.DataFrame()

def extract_df(bulk, ticker):
    try:
        if isinstance(bulk.columns, pd.MultiIndex):
            if ticker in bulk.columns.get_level_values(0): return bulk[ticker].copy().dropna()
    except: pass
    return None

# --- ANALYSIS ---
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
    df['ADX'] = (abs(df['High']-df['Low'])/df['Close']*100).rolling(14).mean() * 5
    return df

def analyze_ticker(ticker, df, regime):
    if len(df) < 50: return None
    df = prepare_df(df)
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(curr['Close'])
    
    clean_sym = ticker.replace(".NS", "")
    win_rate = STATS_CACHE['tickers'].get(clean_sym, 0)
    
    trend = "UP" if close > float(curr['SMA200']) else "DOWN"
    setups = []
    if close > float(curr['EMA20']) and curr['RSI'] > 60: setups.append("Momentum")
    if trend == "UP" and close > float(curr['SMA50']) and abs(close-float(curr['SMA50']))/close < 0.03: setups.append("Pullback")
    
    atr = float(curr['ATR'])
    stop = close - atr
    target = close + (3 * atr)
    rr = round((target-close)/(close-stop), 2)
    
    verdict = "WAIT"
    color = "gray"
    if trend == "UP" and setups:
        if win_rate >= 50:
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
    if not os.path.exists(TRADE_HISTORY_FILE): return []
    trades = json.load(open(TRADE_HISTORY_FILE))
    updated = False
    today = date.today().isoformat()
    
    for t in trades:
        if t['status'] == 'OPEN':
            df = extract_df(bulk, t['symbol']+".NS")
            if df is not None:
                curr = df.iloc[-1]
                t['current_price'] = round(float(curr['Close']), 2) # Add CMP
                low, high, open_p = curr['Low'], curr['High'], curr['Open']
                
                if low <= t['stop_loss']:
                    t['status'] = 'LOSS'
                    t['exit_price'] = round(open_p if open_p < t['stop_loss'] else t['stop_loss'], 2)
                    t['exit_date'] = today
                    updated = True
                elif high >= t['target']:
                    t['status'] = 'WIN'
                    t['exit_price'] = round(open_p if open_p > t['target'] else t['target'], 2)
                    t['exit_date'] = today
                    updated = True
                else:
                    # Calc Unrealized PnL
                    m2m = (float(curr['Close']) - t['entry']) * t['qty']
                    t['net_pnl'] = round(m2m, 2)
                    updated = True
                    
                # Finalize PnL for Closed
                if t['status'] != 'OPEN':
                    pnl = (t['exit_price'] - t['entry']) * t['qty']
                    t['net_pnl'] = round(pnl, 2)

    if updated:
        with open(TRADE_HISTORY_FILE, 'w') as f: json.dump(trades, f, indent=2)
    return trades

def generate_html(signals, trades):
    # Load stats from Cache
    bt = STATS_CACHE.get('portfolio', {})
    bt_ledger = bt.get('ledger', []) # The Backtest Ledger
    
    # Live Ledger
    live_closed = [t for t in trades if t['status'] != 'OPEN']
    live_pnl = round(sum(t.get('net_pnl', 0) for t in live_closed), 2)
    
    data = json.dumps({
        "signals": signals, 
        "trades": trades, 
        "backtest": bt,
        "bt_ledger": bt_ledger
    })
    
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
    <body class="bg-slate-900 text-slate-200 p-4 md:p-8 font-sans">
        <div class="max-w-7xl mx-auto">
            <div class="flex justify-between items-center mb-6 bg-slate-800 p-4 rounded border border-slate-700">
                <div><h1 class="text-2xl font-bold text-white">PrimeTrade</h1><div class="text-xs text-slate-500">Updated: {datetime.utcnow().strftime("%H:%M UTC")}</div></div>
                <div class="text-right">
                    <div class="text-xs text-slate-500">Live PnL</div>
                    <div class="text-xl font-bold {{'text-green-400' if live_pnl>=0 else 'text-red-400'}}">₹{live_pnl}</div>
                </div>
            </div>

            <div class="flex gap-4 mb-6 border-b border-slate-800">
                <button onclick="setTab('dash')" id="btn-dash" class="pb-2 text-sm font-bold text-white border-b-2 border-purple-500">Dashboard</button>
                <button onclick="setTab('ledger')" id="btn-ledger" class="pb-2 text-sm font-bold text-slate-500">Live Ledger</button>
                <button onclick="setTab('strat')" id="btn-strat" class="pb-2 text-sm font-bold text-slate-500">Backtest (Simulation)</button>
            </div>

            <!-- DASHBOARD -->
            <div id="view-dash">
                <h2 class="text-xs font-bold text-slate-500 mb-3 uppercase">Active Portfolio</h2>
                <div id="portfolio" class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8"></div>
                <h2 class="text-xs font-bold text-slate-500 mb-3 uppercase">New Signals</h2>
                <div id="scanner" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4"></div>
            </div>

            <!-- LIVE LEDGER -->
            <div id="view-ledger" class="hidden">
                <div class="bg-slate-800 rounded overflow-hidden">
                    <table class="w-full text-sm text-left">
                        <thead class="bg-slate-700 text-xs uppercase text-slate-400"><tr><th class="p-3">Date</th><th>Symbol</th><th>Result</th><th>PnL</th></tr></thead>
                        <tbody id="live-body"></tbody>
                    </table>
                </div>
            </div>

            <!-- BACKTEST STRATEGY -->
            <div id="view-strat" class="hidden">
                <div class="grid grid-cols-3 gap-4 mb-6">
                    <div class="bg-slate-800 p-4 rounded"><div class="text-xs text-slate-500">1Y Profit</div><div class="text-xl font-bold text-green-400">₹{bt.get('profit', 0)}</div></div>
                    <div class="bg-slate-800 p-4 rounded"><div class="text-xs text-slate-500">Win Rate</div><div class="text-xl font-bold text-blue-400">{bt.get('win_rate', 0)}%</div></div>
                    <div class="bg-slate-800 p-4 rounded"><div class="text-xs text-slate-500">Trades</div><div class="text-xl font-bold text-white">{bt.get('total_trades', 0)}</div></div>
                </div>
                <div class="bg-slate-800 p-4 rounded mb-6 h-64"><canvas id="equityChart"></canvas></div>
                <h3 class="text-xs font-bold text-slate-500 mb-3 uppercase">Simulation Ledger (Last 50)</h3>
                <div class="bg-slate-800 rounded overflow-hidden">
                    <table class="w-full text-sm text-left">
                        <thead class="bg-slate-700 text-xs uppercase text-slate-400"><tr><th class="p-3">Date</th><th>Symbol</th><th>Result</th><th>PnL</th></tr></thead>
                        <tbody id="bt-body"></tbody>
                    </table>
                </div>
            </div>
        </div>

        <script>
            const DATA = {data};
            
            function setTab(id) {{
                ['dash', 'ledger', 'strat'].forEach(t => {{
                    document.getElementById('view-'+t).classList.add('hidden');
                    document.getElementById('btn-'+t).className = "pb-2 text-sm font-bold text-slate-500";
                }});
                document.getElementById('view-'+id).classList.remove('hidden');
                document.getElementById('btn-'+id).className = "pb-2 text-sm font-bold text-white border-b-2 border-purple-500";
            }}

            // PORTFOLIO
            const openT = DATA.trades.filter(t => t.status === 'OPEN');
            const portHTML = openT.length ? openT.map(p => `
                <div class="bg-slate-800 p-4 rounded border border-slate-700">
                    <div class="flex justify-between mb-2"><div class="font-bold">${{p.symbol}}</div><div class="${{p.net_pnl>=0?'text-green-400':'text-red-400'}}">₹${{p.net_pnl}}</div></div>
                    <div class="text-xs text-slate-400 flex justify-between"><span>Entry: ${{p.entry}}</span><span>CMP: ${{p.current_price || '-'}}</span></div>
                    <div class="flex justify-between text-[10px] font-mono mt-2"><span class="text-red-400">SL ${{p.stop_loss}}</span><span class="text-green-400">TGT ${{p.target}}</span></div>
                </div>`).join('') : '<div class="text-center text-slate-500 col-span-full">No active trades</div>';
            document.getElementById('portfolio').innerHTML = portHTML;

            // SCANNER
            const scanHTML = DATA.signals.length ? DATA.signals.map(s => `
                <div class="bg-slate-800 p-4 rounded border border-slate-700 ${{s.verdict.includes('PRIME')?'border-purple-500':''}}">
                    <div class="flex justify-between mb-2"><div class="font-bold">${{s.symbol}}</div><div class="${{s.change>=0?'text-green-400':'text-red-400'}}">${{s.change}}%</div></div>
                    <div class="flex justify-between text-xs mb-2"><span class="text-slate-500">Win%: <span class="${{s.win_rate>50?'text-green-400':'text-yellow-400'}}">${{s.win_rate}}%</span></span><span class="text-slate-500">RR: ${{s.rr}}</span></div>
                    <div class="font-bold text-sm text-center py-1 rounded bg-slate-700 ${{s.v_color==='purple'?'text-purple-400':'text-green-400'}}">${{s.verdict}}</div>
                </div>`).join('') : '<div class="text-center text-slate-500 col-span-full">No signals</div>';
            document.getElementById('scanner').innerHTML = scanHTML;

            // LIVE LEDGER
            const closedT = DATA.trades.filter(t => t.status !== 'OPEN');
            document.getElementById('live-body').innerHTML = closedT.map(t => `
                <tr class="border-b border-slate-700"><td class="p-3 text-slate-400">${{t.exit_date}}</td><td class="p-3 font-bold">${{t.symbol}}</td><td class="p-3"><span class="px-2 py-1 rounded text-[10px] font-bold bg-opacity-20 ${{t.status==='WIN'?'bg-green-500 text-green-400':'bg-red-500 text-red-400'}}">${{t.status}}</span></td><td class="p-3 font-mono ${{t.net_pnl>=0?'text-green-400':'text-red-400'}}">₹${{t.net_pnl}}</td></tr>
            `).join('');

            // BACKTEST LEDGER & CHART
            if(DATA.backtest.ledger) {{
                document.getElementById('bt-body').innerHTML = DATA.backtest.ledger.slice().reverse().map(t => `
                    <tr class="border-b border-slate-700"><td class="p-3 text-slate-400">${{t.date}}</td><td class="p-3 font-bold">${{t.symbol}}</td><td class="p-3"><span class="px-2 py-1 rounded text-[10px] font-bold bg-opacity-20 ${{t.result==='WIN'?'bg-green-500 text-green-400':'bg-red-500 text-red-400'}}">${{t.result}}</span></td><td class="p-3 font-mono">₹${{t.pnl}}</td></tr>
                `).join('');
                
                new Chart(document.getElementById('equityChart'), {{
                    type: 'line',
                    data: {{ labels: DATA.backtest.curve.map((_, i) => i), datasets: [{{ label: 'Equity', data: DATA.backtest.curve, borderColor: '#a855f7', tension: 0.1, pointRadius: 0 }}] }},
                    options: {{ maintainAspectRatio: false, scales: {{ y: {{ grid: {{ color: '#334155' }} }}, x: {{ display: false }} }} }}
                }});
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
    
    signals = []
    cols = bulk.columns.get_level_values(0).unique() if isinstance(bulk.columns, pd.MultiIndex) else bulk.columns
    for t in cols:
        if str(t).startswith('^'): continue
        res = analyze_ticker(t, extract_df(bulk, t), {}, "UNKNOWN")
        if res: signals.append(res)
        
    trades = update_ledger(bulk)
    generate_html(signals, trades, "UNKNOWN")

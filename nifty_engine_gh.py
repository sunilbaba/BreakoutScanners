"""
nifty_engine_gh.py

The Gold Master Trading Engine.
Features:
- Smart Filtering & Duplicate Guard (No double buying).
- ADX Trend & Gap Protection logic.
- 3-Tab Dashboard: Live | Ledger | Backtest Graph.
- JSON Persistence for Trade History.

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
RISK_PER_TRADE = 0.02             
MAX_POSITION_PERC = 0.25          

DATA_PERIOD = "1y"
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 2.0

MIN_ADV_VALUE_RS = 2_000_000      

# Costs
BROKERAGE_PCT = 0.001 # 0.1% per side

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
def prepare_df(df):
    df = df.copy()
    # Trend
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    
    # ATR
    h_l = df['High'] - df['Low']
    h_c = np.abs(df['High'] - df['Close'].shift())
    l_c = np.abs(df['Low'] - df['Close'].shift())
    df['ATR'] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss)).fillna(50)
    
    # ADX
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), -minus_dm, 0.0)
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / tr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / tr)
    df['ADX'] = (abs(plus_di - minus_di) / (plus_di + minus_di) * 100).ewm(alpha=1/14, adjust=False).mean().fillna(0)
    
    return df

# -------------------------
# 4. DATA MANAGEMENT
# -------------------------
@retry_on_exception(max_tries=3)
def robust_download(tickers, period=DATA_PERIOD):
    logger.info(f"Downloading {len(tickers)} symbols...")
    batch_size = 20
    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(batch, period=period, group_by='ticker', threads=True, progress=False)
            frames.append(data)
        except: pass
    if not frames: raise RuntimeError("Download Failed")
    return pd.concat(frames, axis=1)

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
# 5. STRATEGY & BACKTEST ENGINE
# -------------------------
class BacktestEngine:
    def __init__(self, bulk_data, tickers):
        self.data = bulk_data
        self.tickers = [t for t in tickers if not str(t).startswith('^')]
        self.cash = CAPITAL
        self.equity_curve = [CAPITAL]
        self.portfolio = []
        self.history = []
        
    def run(self):
        logger.info("⏳ Running 1-Year Backtest...")
        processed_data = {}
        for t in self.tickers:
            raw = extract_stock_df(self.data, t)
            if raw is not None and len(raw) > 200:
                processed_data[t] = prepare_df(raw)
        
        if not processed_data: return {}
        dates = sorted(list(set().union(*[df.index for df in processed_data.values()])))
        sim_dates = dates[150:] 
        
        for date in sim_dates:
            self.process_day(date, processed_data)
            m2m = 0
            for trade in self.portfolio:
                t = trade['symbol']
                if date in processed_data[t].index:
                    m2m += (processed_data[t].loc[date]['Close'] * trade['qty'])
                else: m2m += (trade['entry'] * trade['qty'])
            self.equity_curve.append(round(self.cash + m2m, 2))

        wins = [t for t in self.history if t['pnl'] > 0]
        win_rate = round(len(wins) / len(self.history) * 100, 1) if self.history else 0
        return {
            "curve": self.equity_curve,
            "win_rate": win_rate,
            "total_trades": len(self.history),
            "profit": round(self.equity_curve[-1] - CAPITAL, 2)
        }

    def process_day(self, date, data_map):
        active = []
        # Exits
        for trade in self.portfolio:
            sym = trade['symbol']
            if date not in data_map[sym].index:
                active.append(trade)
                continue
            row = data_map[sym].loc[date]
            exit_price = None
            
            # Gap Logic
            if row['Open'] < trade['stop']: exit_price = row['Open']
            elif row['Low'] <= trade['stop']: exit_price = trade['stop']
            elif row['Open'] > trade['target']: exit_price = row['Open']
            elif row['High'] >= trade['target']: exit_price = trade['target']
            
            if exit_price:
                pnl = (exit_price - trade['entry']) * trade['qty']
                cost = (exit_price * trade['qty'] * BROKERAGE_PCT) + trade['entry_cost']
                self.cash += (exit_price * trade['qty'] - cost)
                self.history.append({"symbol": sym, "pnl": pnl - cost, "result": "WIN" if pnl>0 else "LOSS"})
            else: active.append(trade)
        self.portfolio = active
        
        # Entries (Max 5)
        if len(self.portfolio) >= 5: return
        for sym, df in data_map.items():
            if date not in df.index: continue
            row = df.loc[date]
            
            if row['Close'] > row['EMA20'] and row['RSI'] > 60 and row['ADX'] > ADX_THRESHOLD and row['Close'] > row['SMA200']:
                if any(t['symbol'] == sym for t in self.portfolio): continue
                
                atr = row['ATR']
                stop = row['Close'] - (1 * atr)
                target = row['Close'] + (3 * atr)
                risk = row['Close'] - stop
                if risk <= 0: continue
                
                qty = int((self.equity_curve[-1] * RISK_PER_TRADE) / risk)
                cost = qty * row['Close']
                
                if qty > 0 and self.cash > cost:
                    fees = cost * BROKERAGE_PCT
                    self.cash -= (cost + fees)
                    self.portfolio.append({
                        "symbol": sym, "entry": row['Close'], "qty": qty,
                        "stop": stop, "target": target, "entry_cost": fees
                    })
                    if len(self.portfolio) >= 5: break

# -------------------------
# 6. LIVE SCANNER
# -------------------------
def analyze_live_ticker(ticker, df, sector_changes, market_regime):
    if len(df) < 100: return None
    df = prepare_df(df)
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(curr['Close'])
    
    trend = "UP" if close > float(curr['SMA200']) else "DOWN"
    setups = []
    
    if close > float(curr['EMA20']) and curr['RSI'] > 60 and curr['ADX'] > ADX_THRESHOLD: setups.append("Momentum Burst")
    if trend == "UP" and close > float(curr['SMA50']) and abs(close - float(curr['SMA50']))/close < 0.03: setups.append("Pullback")
    
    clean_sym = ticker.replace(".NS", "")
    has_sector = False
    for sec_name, val in sector_changes.items():
        if val > 0.5: has_sector = True 
            
    atr_val = float(curr['ATR'])
    stop = round(close - ATR_MULTIPLIER_STOP * atr_val, 2)
    target = round(close + ATR_MULTIPLIER_TARGET * atr_val, 2)
    rr = round((target - close) / (close - stop), 2)
    
    verdict = "WAIT"
    v_color = "gray"
    
    if trend == "UP" and len(setups) > 0:
        if rr >= 2.0:
            verdict = "BUY"
            v_color = "green"
            if has_sector: 
                verdict = "PRIME BUY ⭐"
                v_color = "purple"
    
    if verdict == "WAIT" and abs((close - float(prev['Close']))/float(prev['Close'])) < 0.01: return None

    return {
        "symbol": clean_sym,
        "price": round(close, 2),
        "change": round(((close - float(prev['Close']))/float(prev['Close']))*100, 2),
        "verdict": verdict,
        "v_color": v_color,
        "rr": rr,
        "adx": round(float(curr['ADX']), 1),
        "setups": setups,
        "levels": {"TGT": target, "SL": stop},
        "history": df['Close'].tail(30).tolist()
    }

# -------------------------
# 7. PORTFOLIO MANAGER (LEDGER)
# -------------------------
def load_json_file(path):
    if os.path.exists(path):
        try: return json.load(open(path))
        except: pass
    return []

def update_trades(trades, bulk_data):
    updated = False
    today = date.today().isoformat()
    
    for t in trades:
        if t['status'] == 'OPEN':
            df = extract_stock_df(bulk_data, t['symbol']+".NS")
            if df is not None:
                curr = df.iloc[-1]
                low, high, open_p = curr['Low'], curr['High'], curr['Open']
                
                # GAP Logic for Live Trades
                if low <= t['stop_loss']:
                    t['status'] = 'LOSS'
                    t['exit_price'] = open_p if open_p < t['stop_loss'] else t['stop_loss']
                    updated = True
                elif high >= t['target']:
                    t['status'] = 'WIN'
                    t['exit_price'] = open_p if open_p > t['target'] else t['target']
                    updated = True
                
                if t['status'] != 'OPEN':
                    t['exit_date'] = today
                    cost = (t['entry'] * t['qty'] * BROKERAGE_PCT) * 2
                    t['net_pnl'] = round((t['exit_price'] - t['entry']) * t['qty'] - cost, 2)
    
    if updated:
        with open(TRADE_HISTORY_FILE, 'w') as f: json.dump(trades, f, indent=2)
    return trades

def add_new_signals(signals, trades):
    today = date.today().isoformat()
    # SMART FILTER: Don't buy if already owned
    owned = {t['symbol'] for t in trades if t['status'] == 'OPEN'}
    
    valid = [s for s in signals if "BUY" in s['verdict'] and s['symbol'] not in owned]
    valid.sort(key=lambda x: x['rr'], reverse=True)
    
    for s in valid[:3]:
        tid = f"{s['symbol']}-{today}"
        # DUPLICATE GUARD
        if not any(t['id'] == tid for t in trades):
            # Dynamic Sizing
            risk_amt = CAPITAL * RISK_PER_TRADE
            risk_per_share = s['price'] - s['levels']['SL']
            qty = int(risk_amt / risk_per_share) if risk_per_share > 0 else 0
            
            trades.insert(0, {
                "id": tid, "date": today, "symbol": s['symbol'],
                "entry": s['price'], "qty": qty,
                "target": s['levels']['TGT'], "stop_loss": s['levels']['SL'],
                "status": "OPEN", "net_pnl": 0
            })
    
    with open(TRADE_HISTORY_FILE, 'w') as f: json.dump(trades, f, indent=2)

# -------------------------
# 8. HTML GENERATION
# -------------------------
def generate_html(signals, trades, bt_stats, timestamp):
    closed = [t for t in trades if t['status'] != 'OPEN']
    net_pnl = round(sum(t.get('net_pnl', 0) for t in closed), 2)
    
    owned = {t['symbol'] for t in trades if t['status'] == 'OPEN'}
    display_signals = [s for s in signals if s['symbol'] not in owned]
    display_signals.sort(key=lambda x: x['rr'], reverse=True)
    
    json_data = json.dumps({"signals": display_signals, "trades": trades, "backtest": bt_stats})
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PrimeTrade PRO</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            body {{ background: #0f172a; color: #e2e8f0; font-family: sans-serif; }}
            .card {{ background: #1e293b; border: 1px solid #334155; padding: 16px; border-radius: 12px; }}
            .tab-active {{ border-bottom: 2px solid #a855f7; color: white; }}
            .tab-inactive {{ color: #64748b; }}
            .win {{ color: #4ade80; }}
            .loss {{ color: #f87171; }}
        </style>
    </head>
    <body class="p-4 md:p-8">
        <div class="max-w-7xl mx-auto">
            <div class="flex justify-between items-center mb-6 bg-slate-900 p-4 rounded-xl border border-slate-800">
                <div><h1 class="text-2xl font-bold text-white flex items-center gap-2"><i data-lucide="layers" class="text-purple-500"></i> PrimeTrade</h1><div class="text-xs text-slate-500 mt-1">{timestamp}</div></div>
                <div class="text-right"><div class="text-[10px] text-slate-500 uppercase">Realized PnL</div><div class="text-xl font-bold {{'win' if net_pnl>=0 else 'loss'}}">₹{net_pnl}</div></div>
            </div>

            <div class="flex gap-6 mb-6 border-b border-slate-800">
                <button onclick="setTab('dash')" id="btn-dash" class="pb-2 text-sm font-bold tab-active">Dashboard</button>
                <button onclick="setTab('ledger')" id="btn-ledger" class="pb-2 text-sm font-bold tab-inactive">Ledger</button>
                <button onclick="setTab('strat')" id="btn-strat" class="pb-2 text-sm font-bold tab-inactive">Strategy Validation</button>
            </div>

            <!-- DASHBOARD -->
            <div id="view-dash">
                <h2 class="text-xs font-bold text-slate-500 mb-3 uppercase">Active Portfolio</h2>
                <div id="portfolio" class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8"></div>
                <h2 class="text-xs font-bold text-slate-500 mb-3 uppercase">New Signals</h2>
                <div id="scanner" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4"></div>
            </div>

            <!-- LEDGER -->
            <div id="view-ledger" class="hidden">
                <div class="bg-slate-900 rounded-lg border border-slate-800 overflow-hidden">
                    <table class="w-full text-sm text-left text-slate-400">
                        <thead class="bg-slate-800 text-xs uppercase text-slate-500"><tr><th class="p-3">Date</th><th class="p-3">Symbol</th><th class="p-3">Result</th><th class="p-3">PnL</th></tr></thead>
                        <tbody id="ledger-body"></tbody>
                    </table>
                </div>
            </div>

            <!-- STRATEGY -->
            <div id="view-strat" class="hidden">
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <div class="card text-center"><div class="text-xs text-slate-500">Backtest Profit (1Y)</div><div class="text-2xl font-bold text-green-400">₹{bt_stats.get('profit', 0)}</div></div>
                    <div class="card text-center"><div class="text-xs text-slate-500">Win Rate</div><div class="text-2xl font-bold text-blue-400">{bt_stats.get('win_rate', 0)}%</div></div>
                    <div class="card text-center"><div class="text-xs text-slate-500">Total Trades</div><div class="text-2xl font-bold text-white">{bt_stats.get('total_trades', 0)}</div></div>
                </div>
                <div class="card h-64"><canvas id="equityChart"></canvas></div>
            </div>
        </div>

        <script>
            const DATA = {json_data};
            
            function setTab(id) {{
                ['dash', 'ledger', 'strat'].forEach(t => {{
                    document.getElementById('view-'+t).classList.add('hidden');
                    document.getElementById('btn-'+t).className = "pb-2 text-sm font-bold tab-inactive";
                }});
                document.getElementById('view-'+id).classList.remove('hidden');
                document.getElementById('btn-'+id).className = "pb-2 text-sm font-bold tab-active";
            }}

            const portRoot = document.getElementById('portfolio');
            const openT = DATA.trades.filter(t => t.status === 'OPEN');
            if(openT.length === 0) portRoot.innerHTML = '<div class="col-span-full text-center text-slate-600 py-4">No active trades</div>';
            else {{
                portRoot.innerHTML = openT.map(p => `
                    <div class="card">
                        <div class="flex justify-between mb-2"><div class="font-bold text-white">${{p.symbol}}</div><div class="text-xs bg-blue-900 text-blue-200 px-2 py-1 rounded">OPEN</div></div>
                        <div class="text-xs text-slate-400 flex justify-between"><span>Qty: ${{p.qty}}</span><span>Entry: ${{p.entry}}</span></div>
                        <div class="flex justify-between text-[10px] font-mono mt-2"><span class="loss">${{p.stop_loss}} SL</span><span class="win">${{p.target}} TGT</span></div>
                    </div>`).join('');
            }}

            const scanRoot = document.getElementById('scanner');
            if(DATA.signals.length === 0) scanRoot.innerHTML = '<div class="col-span-full text-center text-slate-600 py-10">No new signals</div>';
            else {{
                scanRoot.innerHTML = DATA.signals.map(s => `
                    <div class="card ${{s.verdict.includes('PRIME')?'prime':''}}">
                        <div class="flex justify-between mb-2"><div class="font-bold text-white">${{s.symbol}}</div><div class="font-bold ${{s.change>=0?'win':'loss'}}">${{s.change}}%</div></div>
                        <div class="flex justify-between bg-slate-900/50 p-2 rounded mb-2 text-[10px]"><div class="text-center"><div>ADX</div><div class="text-white font-bold">${{s.adx}}</div></div><div class="text-center"><div>RR</div><div class="text-white font-bold">${{s.rr}}</div></div></div>
                        <div class="flex justify-between text-[10px] font-mono mt-2"><span class="loss">SL ${{s.levels.SL}}</span><span class="win">TGT ${{s.levels.TGT}}</span></div>
                    </div>`).join('');
            }}

            const ledgerRoot = document.getElementById('ledger-body');
            const closedT = DATA.trades.filter(t => t.status !== 'OPEN');
            ledgerRoot.innerHTML = closedT.map(t => `
                <tr class="border-b border-slate-800">
                    <td class="p-3">${{t.date}}</td><td class="p-3 font-bold">${{t.symbol}}</td>
                    <td class="p-3"><span class="text-[10px] px-2 py-1 rounded ${{t.status==='WIN'?'bg-green-900 text-green-200':'bg-red-900 text-red-200'}}">${{t.status}}</span></td>
                    <td class="p-3 font-mono ${{t.net_pnl>=0?'win':'loss'}}">₹${{t.net_pnl}}</td>
                </tr>`).join('');

            const ctx = document.getElementById('equityChart');
            if(DATA.backtest.curve) {{
                new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: DATA.backtest.curve.map((_, i) => i),
                        datasets: [{{ label: 'Portfolio Value', data: DATA.backtest.curve, borderColor: '#a855f7', tension: 0.1, pointRadius: 0 }}]
                    }},
                    options: {{ maintainAspectRatio: false, scales: {{ y: {{ grid: {{ color: '#334155' }} }}, x: {{ display: false }} }} }}
                }});
            }}
            lucide.createIcons();
        </script>
    </body>
    </html>
    """
    with open(FILE_PATH, "w") as f: f.write(html)

if __name__ == "__main__":
    logger.info("Starting Engine...")
    tickers = get_tickers()
    bulk_data = robust_download(tickers + list(SECTOR_INDICES.values()))
    
    backtester = BacktestEngine(bulk_data, tickers)
    bt_stats = backtester.run()
    
    sector_changes = {}
    signals = []
    cols = bulk_data.columns.get_level_values(0).unique() if isinstance(bulk_data.columns, pd.MultiIndex) else bulk_data.columns
    for t in cols:
        if str(t).startswith('^'): continue
        try:
            res = analyze_live_ticker(t, extract_stock_df(bulk_data, t), sector_changes, "UNKNOWN")
            if res: signals.append(res)
        except: continue

    trades = load_json_file(TRADE_HISTORY_FILE)
    trades = update_trades(trades, bulk_data)
    add_new_signals(signals, trades)
    
    generate_html(signals, trades, bt_stats, datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
    logger.info("Done.")

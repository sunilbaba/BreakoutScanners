"""
nifty_engine_gh.py
THE "DEEP DIVE" ENGINE
----------------------
1. Scans Live Market.
2. If Signal Found -> Downloads 5Y Data for that specific stock.
3. VALIDATES the strategy on that stock's long-term history.
4. Adds '5Y Win Rate' & '5Y Profit' to the dashboard card.
"""

import os
import time
import json
import logging
import requests
from datetime import datetime, date

import yfinance as yf
import pandas as pd
import numpy as np

# --- 1. CONFIGURATION ---
CAPITAL = 100000.0
RISK_PER_TRADE = 0.02  
DATA_PERIOD = "1y" # Fast scan first
DEEP_PERIOD = "5y" # Deep dive for winners

# Files
OUTPUT_DIR = "public"
HTML_FILE = os.path.join(OUTPUT_DIR, "index.html")
TRADE_HISTORY_FILE = "trade_history.json"
CACHE_FILE = "backtest_stats.json"
STRATEGY_FILE = "strategy_config.json"

# Costs
BROKERAGE_PCT = 0.001 

SECTOR_INDICES = {
    "NIFTY 50": "^NSEI", "BANK": "^NSEBANK", "AUTO": "^CNXAUTO", "IT": "^CNXIT",
    "METAL": "^CNXMETAL", "PHARMA": "^CNXPHARMA", "FMCG": "^CNXFMCG",
    "ENERGY": "^CNXENERGY", "REALTY": "^CNXREALTY", "PSU BANK": "^CNXPSUBANK"
}

DEFAULT_TICKERS = [ "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "SBIN.NS", "ITC.NS", "TATAMOTORS.NS" ]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("PrimeEngine")

# --- 2. STRATEGY LOADER ---
def load_strategy():
    default = {
        "parameters": { "strategy_type": "RSI_DIP", "trend_filter": "SMA200", "rsi_threshold": 30, "adx_min": 20, "sl_mult": 1.5, "tgt_mult": 3.0 }
    }
    if os.path.exists(STRATEGY_FILE):
        try:
            data = json.load(open(STRATEGY_FILE))
            if "parameters" in data: return data['parameters']
        except: pass
    return default['parameters']

AI_BRAIN = load_strategy()

# --- 3. DATA ENGINE ---
def robust_download(tickers, period):
    if "^NSEI" not in tickers and period == DATA_PERIOD: tickers.append("^NSEI")
    logger.info(f"⬇️ Downloading {len(tickers)} symbols ({period})...")
    try:
        return yf.download(tickers, period=period, group_by='ticker', threads=True, progress=False, ignore_tz=True)
    except: return pd.DataFrame()

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

# --- 4. INDICATORS ---
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
    p_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    m_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), -minus_dm, 0.0)
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).ewm(alpha=1/14).mean()
    p_di = 100 * (pd.Series(p_dm, index=df.index).ewm(alpha=1/14).mean() / tr)
    m_di = 100 * (pd.Series(m_dm, index=df.index).ewm(alpha=1/14).mean() / tr)
    df['ADX'] = (abs(p_di - m_di) / (p_di + m_di) * 100).ewm(alpha=1/14).mean().fillna(0)
    
    # BB
    df['BB_MID'] = df['Close'].rolling(20).mean()
    df['BB_LOW'] = df['BB_MID'] - (2 * df['Close'].rolling(20).std())

    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_SIG'] = df['MACD'].ewm(span=9).mean()
    
    df['ROC_6M'] = df['Close'].pct_change(126) * 100
    return df

# --- 5. THE "DEEP DIVE" VALIDATOR (5Y Check) ---
def run_deep_dive(ticker):
    """
    Downloads 5 Years of data for this specific stock and tests the strategy.
    Returns: 5Y Win Rate, Total Profit, Trade Count.
    """
    logger.info(f"🔎 Running 5-Year Deep Dive for {ticker}...")
    try:
        df = yf.download(ticker, period=DEEP_PERIOD, progress=False, ignore_tz=True)
        if df.empty: return 0, 0, 0
        df = prepare_df(df)
    except: return 0, 0, 0

    wins, losses = 0, 0
    
    # Simulation Loop (Vector-friendly)
    # Logic: We can reuse the 'fast_score' logic logic from backtest_runner but customized
    # To keep this file independent, we replicate the core check:
    
    st = AI_BRAIN.get('strategy_type', 'RSI_DIP')
    trend_m = AI_BRAIN.get('trend_filter', 'SMA200')
    adx_m = AI_BRAIN.get('adx_min', 15)
    sl_m = AI_BRAIN.get('sl_mult', 1.5)
    tgt_m = AI_BRAIN.get('tgt_mult', 3.0)
    
    # Iterate manually to account for hold time
    close = df['Close'].values
    low = df['Low'].values
    high = df['High'].values
    sma200 = df['SMA200'].values
    rsi = df['RSI'].values
    adx = df['ADX'].values
    atr = df['ATR'].values
    
    i = 200; end = len(df) - 20
    while i < end:
        is_entry = False
        # Filters
        if trend_m == "SMA200" and close[i] < sma200[i]: i+=1; continue
        if adx[i] <= adx_m: i+=1; continue
        
        # Signal
        if st == "RSI_DIP" and rsi[i] < AI_BRAIN.get('rsi_threshold', 30): is_entry = True
        elif st == "BREAKOUT" and close[i] >= np.max(high[max(0, i-20):i]): is_entry = True
        
        if is_entry:
            sl = close[i] - (sl_m * atr[i])
            tgt = close[i] + (tgt_m * atr[i])
            outcome = "OPEN"
            days = 0
            for j in range(1, 20):
                days = j
                idx = i + j
                if idx >= len(close): break
                if low[idx] <= sl: outcome="LOSS"; break
                if high[idx] >= tgt: outcome="WIN"; break
            
            if outcome == "WIN": wins += 1
            elif outcome == "LOSS": losses += 1
            i += days
        else: i += 1
        
    total = wins + losses
    wr = round(wins/total*100, 1) if total > 0 else 0
    return wr, total

# --- 6. LIVE SCANNER ---
def analyze_ticker(ticker, df, benchmark_roc, market_regime):
    if len(df) < 130: return None
    df = prepare_df(df)
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(curr['Close'])
    clean_sym = ticker.replace(".NS", "")
    
    # RS Filter
    stock_roc = float(curr['ROC_6M']) if not pd.isna(curr['ROC_6M']) else -999
    if stock_roc < benchmark_roc: return None
    
    # Logic Check
    st_type = AI_BRAIN.get('strategy_type', 'RSI_DIP')
    trend_mode = AI_BRAIN.get('trend_filter', 'SMA200')
    adx_min = AI_BRAIN.get('adx_min', 15)
    
    sma200 = float(curr['SMA200']) if not pd.isna(curr['SMA200']) else close
    
    trend_ok = True
    if trend_mode == "SMA200" and close < sma200: trend_ok = False
    if float(curr['ADX']) <= adx_min: trend_ok = False
    
    is_entry = False
    setup_name = ""
    
    if st_type == "RSI_DIP":
        if float(curr['RSI']) < AI_BRAIN.get('rsi_threshold', 30): is_entry = True; setup_name = "RSI Oversold"
    elif st_type == "BB_REVERSAL":
        if float(curr['Low']) <= float(curr['BB_LOW']) and close > float(curr['BB_LOW']): is_entry = True; setup_name = "BB Reversal"
    elif st_type == "MACD_CROSS":
        if float(prev['MACD']) < float(prev['MACD_SIG']) and float(curr['MACD']) > float(curr['MACD_SIG']): is_entry = True; setup_name = "MACD Cross"
    elif st_type == "BREAKOUT":
         high_20 = df['High'].rolling(20).max().iloc[-2]
         if close > high_20: is_entry = True; setup_name = "Breakout"

    verdict = "WAIT"
    v_color = "gray"
    
    # If setup detected, perform DEEP DIVE
    deep_wr = 0
    deep_trades = 0
    
    if trend_ok and is_entry:
        # *** THE DEEP DIVE ***
        # Only download 5Y history if we are about to buy.
        deep_wr, deep_trades = run_deep_dive(ticker)
        
        if deep_wr >= 60:
            verdict = "PRIME BUY ⭐"
            v_color = "purple"
        elif deep_wr >= 40:
            verdict = "BUY"
            v_color = "green"
        else:
            verdict = "RISKY (Bad History)"
            v_color = "orange"
            
    if verdict == "WAIT": return None

    # Sizing
    atr_val = float(curr['ATR'])
    stop = close - (AI_BRAIN.get('sl_mult', 1.5) * atr_val)
    target = close + (AI_BRAIN.get('tgt_mult', 3.0) * atr_val)
    rr = round((target - close) / (close - stop), 2)
    
    adjusted_risk = RISK_PER_TRADE / 2 if "BEAR" in market_regime else RISK_PER_TRADE
    qty = int((CAPITAL * adjusted_risk) / (close - stop)) if (close - stop) > 0 else 0

    return {
        "symbol": clean_sym, "price": round(close, 2),
        "change": round(((close - float(prev['Close']))/float(prev['Close']))*100, 2),
        "verdict": verdict, "v_color": v_color, "rr": rr, "qty": qty,
        "setups": [setup_name],
        "deep_dive": {"wr": deep_wr, "trades": deep_trades}, # <--- NEW DATA FOR UI
        "levels": {"TGT": round(target, 2), "SL": round(stop, 2)},
        "history": df['Close'].tail(30).tolist()
    }

# --- 6. EXECUTION & HTML ---
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
                low, high = float(curr['Low']), float(curr['High'])
                if low <= t['stop_loss']:
                    t['status'] = 'LOSS'; t['exit_price'] = t['stop_loss']; updated = True
                elif high >= t['target']:
                    t['status'] = 'WIN'; t['exit_price'] = t['target']; updated = True
                if t['status'] != 'OPEN':
                    t['exit_date'] = today
                    pnl = (t['exit_price'] - t['entry']) * t['qty']
                    t['net_pnl'] = round(pnl - (pnl * BROKERAGE_PCT), 2)
    if updated: save_json(TRADE_HISTORY_FILE, trades)
    return trades

def add_new_signals(signals, trades):
    today = date.today().isoformat()
    owned = {t['symbol'] for t in trades if t['status'] == 'OPEN'}
    
    # Sort by Deep Dive Win Rate
    valid = [s for s in signals if "BUY" in s['verdict'] and s['symbol'] not in owned]
    valid.sort(key=lambda x: x['deep_dive']['wr'], reverse=True)
    
    for s in valid[:3]: 
        tid = f"{s['symbol']}-{today}"
        if not any(t['id'] == tid for t in trades):
            trades.insert(0, {
                "id": tid, "date": today, "symbol": s['symbol'],
                "entry": s['price'], "qty": s['qty'],
                "target": s['levels']['TGT'], "stop_loss": s['levels']['SL'],
                "status": "OPEN", "net_pnl": 0
            })
            save_json(TRADE_HISTORY_FILE, trades)

def generate_html(signals, trades, bt_stats, timestamp, regime):
    closed = [t for t in trades if t['status'] != 'OPEN']
    net_pnl = round(sum(t.get('net_pnl', 0) for t in closed), 2)
    strat_name = AI_BRAIN.get('strategy_type', 'Unknown')
    
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
                    <h1 class="text-2xl font-bold text-white">PrimeTrade</h1>
                    <div class="text-xs text-purple-400 font-bold">AI: {strat_name} ({regime})</div>
                    <div class="text-xs text-slate-500">{timestamp}</div>
                </div>
                <div class="text-right"><div class="text-xs text-slate-500">Realized PnL</div><div class="text-xl font-bold {{'text-green-400' if net_pnl>=0 else 'text-red-400'}}">₹{net_pnl}</div></div>
            </div>
            
            <div class="flex gap-4 mb-6">
                <button onclick="setTab('dash')" id="btn-dash" class="font-bold text-white border-b-2 border-purple-500 pb-2">Dashboard</button>
                <button onclick="setTab('ledger')" id="btn-ledger" class="font-bold text-slate-500 pb-2">Ledger</button>
            </div>
            
            <div id="view-dash">
                <h2 class="text-xs font-bold text-slate-500 mb-3">ACTIVE PORTFOLIO</h2>
                <div id="portfolio" class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8"></div>
                <h2 class="text-xs font-bold text-slate-500 mb-3">NEW SIGNALS (5Y VALIDATED)</h2>
                <div id="scanner" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4"></div>
            </div>
            
            <div id="view-ledger" class="hidden">
                <div class="bg-slate-800 rounded overflow-hidden"><table class="w-full text-sm text-left"><thead class="bg-slate-700"><tr><th class="p-3">Date</th><th>Symbol</th><th>Result</th><th>PnL</th></tr></thead><tbody id="live-body"></tbody></table></div>
            </div>
        </div>

        <!-- MODAL -->
        <div id="stockModal" class="fixed inset-0 bg-black/80 hidden flex items-center justify-center z-50" onclick="this.classList.add('hidden')">
            <div class="bg-slate-800 p-6 rounded-lg border border-slate-600 w-96" onclick="event.stopPropagation()">
                <h2 id="m-sym" class="text-2xl font-bold text-white mb-2">SYMBOL</h2>
                <div class="grid grid-cols-2 gap-4 mb-4">
                    <div class="bg-slate-900 p-2 rounded text-center"><div class="text-xs text-slate-500">5Y Win Rate</div><div id="m-wr" class="text-xl font-bold text-purple-400">0%</div></div>
                    <div class="bg-slate-900 p-2 rounded text-center"><div class="text-xs text-slate-500">5Y Trades</div><div id="m-tr" class="text-xl font-bold text-white">0</div></div>
                </div>
                <div class="text-xs text-center text-slate-400">Historical performance of {strat_name} on this stock.</div>
            </div>
        </div>

        <script>
            const DATA = {json_data};
            function setTab(id) {{
                ['dash', 'ledger'].forEach(t => {{ document.getElementById('view-'+t).classList.add('hidden'); document.getElementById('btn-'+t).className="font-bold text-slate-500 pb-2"; }});
                document.getElementById('view-'+id).classList.remove('hidden');
                document.getElementById('btn-'+id).className="font-bold text-white border-b-2 border-purple-500 pb-2";
            }}

            function showModal(idx) {{
                const s = DATA.signals[idx];
                document.getElementById('m-sym').innerText = s.symbol;
                document.getElementById('m-wr').innerText = s.deep_dive.wr + "%";
                document.getElementById('m-tr').innerText = s.deep_dive.trades;
                document.getElementById('stockModal').classList.remove('hidden');
            }}

            const portRoot = document.getElementById('portfolio');
            const openT = DATA.trades.filter(t => t.status === 'OPEN');
            portRoot.innerHTML = openT.length ? openT.map(p => `
                <div class="bg-slate-800 p-4 rounded border border-slate-700">
                    <div class="flex justify-between mb-2"><div class="font-bold text-white">${{p.symbol}}</div><div class="text-xs bg-blue-900 text-blue-200 px-2 py-1 rounded">OPEN</div></div>
                    <div class="flex justify-between text-[10px] font-mono mt-2"><span class="text-red-400">SL ${{p.stop_loss}}</span><span class="text-green-400">TGT ${{p.target}}</span></div>
                </div>`).join('') : '<div class="col-span-full text-center text-slate-600 py-4">No active trades</div>';

            const scanRoot = document.getElementById('scanner');
            scanRoot.innerHTML = DATA.signals.length ? DATA.signals.map((s, i) => `
                <div class="bg-slate-800 p-4 rounded border border-slate-700 cursor-pointer hover:border-purple-500 transition" onclick="showModal(${{i}})">
                    <div class="flex justify-between mb-2"><div class="font-bold text-white">${{s.symbol}}</div><div class="${{s.change>=0?'text-green-400':'text-red-400'}}">${{s.change}}%</div></div>
                    <div class="flex justify-between text-xs mb-2"><span class="text-slate-500">5Y Win: <span class="${{s.deep_dive.wr>50?'text-green-400':'text-yellow-400'}}">${{s.deep_dive.wr}}%</span></span><span class="text-slate-500">RR: ${{s.rr}}</span></div>
                    <div class="font-bold text-sm text-center py-1 rounded bg-slate-700 ${{s.v_color === 'purple' ? 'text-purple-400' : 'text-green-400'}}">${{s.verdict}}</div>
                </div>`).join('') : '<div class="col-span-full text-center text-slate-600 py-10">No strong signals</div>';

            const ledgerRoot = document.getElementById('live-body');
            ledgerRoot.innerHTML = DATA.trades.filter(t => t.status !== 'OPEN').map(t => `<tr class="border-b border-slate-700"><td class="p-3">${{t.exit_date}}</td><td class="p-3 font-bold">${{t.symbol}}</td><td class="p-3"><span class="text-[10px] px-2 py-1 rounded ${{t.status==='WIN'?'bg-green-900 text-green-200':'bg-red-900 text-red-200'}}">${{t.status}}</span></td><td class="p-3 font-mono ${{t.net_pnl>=0?'text-green-400':'text-red-400'}}">₹${{t.net_pnl}}</td></tr>`).join('');
        </script>
    </body>
    </html>
    """
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    with open(HTML_FILE, "w") as f: f.write(html)

if __name__ == "__main__":
    tickers = get_tickers()
    bulk = robust_download(tickers)
    
    # Load AI Brain
    bt_stats = {}
    if os.path.exists(CACHE_FILE):
        try: bt_stats = json.load(open(CACHE_FILE))
        except: pass
    win_rates = bt_stats.get("tickers", {})
    
    # Benchmark
    nifty_roc = -999
    regime = "UNKNOWN"
    nifty_df = extract_df(bulk, "^NSEI")
    if nifty_df is not None:
        nifty_df = prepare_df(nifty_df)
        if not nifty_df.empty and 'ROC_6M' in nifty_df.columns: nifty_roc = nifty_df['ROC_6M'].iloc[-1]
        curr = nifty_df.iloc[-1]
        if curr['Close'] > curr['SMA200']: regime = "BULL MARKET 🟢"
        else: regime = "BEAR MARKET 🔴"

    # Scan
    signals = []
    cols = bulk.columns.get_level_values(0).unique() if isinstance(bulk.columns, pd.MultiIndex) else bulk.columns
    for t in cols:
        if str(t).startswith('^'): continue
        try:
            res = analyze_ticker(t, extract_df(bulk, t), win_rates, nifty_roc, regime)
            if res: signals.append(res)
        except: continue

    trades = load_json(TRADE_HISTORY_FILE)
    trades = update_trades(trades, bulk)
    add_new_signals(signals, trades)
    
    generate_html(signals, trades, bt_stats, datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), regime)
    logging.info("Done.")

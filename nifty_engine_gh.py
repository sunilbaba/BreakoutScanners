import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import warnings
from datetime import datetime, date

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# --- USER SETTINGS (MANAGE YOUR MONEY) ---
CAPITAL = 100000       # Total Capital in Rupees
RISK_PER_TRADE = 0.01  # Risk 1% of capital per trade (₹1000)
PYRAMIDING = False     # Do not add to existing positions

# --- CONFIGURATION ---
OUTPUT_DIR = "public"
FILE_PATH = os.path.join(OUTPUT_DIR, "index.html")
HISTORY_FILE = "trade_history.json"

SECTOR_INDICES = {
    "NIFTY 50": "^NSEI", # ADDED MARKET INDEX
    "BANK": "^NSEBANK", "AUTO": "^CNXAUTO", "IT": "^CNXIT",
    "METAL": "^CNXMETAL", "PHARMA": "^CNXPHARMA", "FMCG": "^CNXFMCG",
    "ENERGY": "^CNXENERGY", "REALTY": "^CNXREALTY", "PSU BANK": "^CNXPSUBANK"
}

def get_stock_sector(symbol):
    s = symbol.replace('.NS', '')
    if s in ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK"]: return "BANK"
    if s in ["TCS", "INFY", "HCLTECH", "WIPRO", "LTIM", "TECHM"]: return "IT"
    if s in ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "EICHERMOT"]: return "AUTO"
    if s in ["TATASTEEL", "HINDALCO", "JSWSTEEL", "JINDALSTEL", "VEDL"]: return "METAL"
    if s in ["SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "LUPIN"]: return "PHARMA"
    if s in ["ITC", "HINDUNILVR", "NESTLEIND", "BRITANNIA"]: return "FMCG"
    return "Other"

# --- 1. DATA ACQUISITION ---
def get_tickers():
    default_list = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LTIM.NS", "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS", "BAJFINANCE.NS", "HCLTECH.NS", "ADANIENT.NS", "TATASTEEL.NS", "JIOFIN.NS", "ZOMATO.NS", "DLF.NS", "HAL.NS", "VBL.NS", "TRENT.NS", "BEL.NS", "POWERGRID.NS", "ONGC.NS", "NTPC.NS", "COALINDIA.NS", "BPCL.NS", "WIPRO.NS"]
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        df = pd.read_csv(url, storage_options=headers)
        return [f"{x}.NS" for x in df['Symbol'].dropna().unique()]
    except: return default_list

def fetch_bulk_data(tickers):
    all_tickers = tickers + list(SECTOR_INDICES.values())
    print(f"Downloading {len(all_tickers)} symbols...")
    try:
        return yf.download(all_tickers, period="1y", threads=True, progress=True)
    except: return pd.DataFrame()

def extract_stock_df(bulk_data, ticker):
    try:
        # Handle MultiIndex (Price, Ticker) or (Ticker, Price)
        if isinstance(bulk_data.columns, pd.MultiIndex):
            if ticker in bulk_data.columns.get_level_values(1): # (Price, Ticker)
                df = pd.DataFrame({
                    'Open': bulk_data['Open'][ticker],
                    'High': bulk_data['High'][ticker],
                    'Low': bulk_data['Low'][ticker],
                    'Close': bulk_data['Close'][ticker],
                    'Volume': bulk_data['Volume'][ticker]
                })
                return df.dropna()
            elif ticker in bulk_data.columns.get_level_values(0): # (Ticker, Price)
                return bulk_data[ticker].copy().dropna()
        else:
            return bulk_data # Single ticker fallback
    except: pass
    return None

# --- 2. MARKET REGIME ANALYSIS ---
def analyze_market_trend(bulk_data):
    """Checks if Nifty 50 is Bullish or Bearish."""
    try:
        nifty = extract_stock_df(bulk_data, "^NSEI")
        if nifty is None: return "UNKNOWN"
        
        curr = nifty.iloc[-1]
        sma200 = nifty['Close'].rolling(200).mean().iloc[-1]
        sma50 = nifty['Close'].rolling(50).mean().iloc[-1]
        
        if curr['Close'] > sma50 and sma50 > sma200: return "BULL MARKET 🟢"
        if curr['Close'] > sma200: return "UPTREND 🟡"
        return "BEAR MARKET 🔴"
    except: return "UNKNOWN"

# --- 3. SWING ENGINE & POSITION SIZING ---
def calculate_qty(entry, stop_loss):
    """Calculates position size based on Risk Management."""
    risk_amount = CAPITAL * RISK_PER_TRADE
    risk_per_share = entry - stop_loss
    if risk_per_share <= 0: return 0
    qty = int(risk_amount / risk_per_share)
    # Sanity check: Don't put more than 25% capital in one stock
    max_capital_allocation = CAPITAL * 0.25
    if (qty * entry) > max_capital_allocation:
        qty = int(max_capital_allocation / entry)
    return qty

def analyze_ticker(ticker, df, sector_changes, market_regime):
    if len(df) < 200: return None
    df = df.copy()
    
    # Indicators
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    
    # ATR
    h_l = df['High'] - df['Low']
    h_c = np.abs(df['High'] - df['Close'].shift())
    l_c = np.abs(df['Low'] - df['Close'].shift())
    df['ATR'] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0)).rolling(14).mean() / (df['Close'].diff().clip(upper=0).abs()).rolling(14).mean()))
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(curr['Close'])
    
    # Trend
    trend = "UP" if close > float(curr['SMA200']) else "DOWN"
    
    # Setups
    setups = []
    ema20 = float(curr['EMA20'])
    
    # Momentum
    if close > ema20 and curr['RSI'] > 60: setups.append("Momentum")
    # Pullback
    sma50 = float(curr['SMA50'])
    if trend == "UP" and close > sma50 and abs(close - sma50)/close < 0.03 and curr['RSI'] < 60: setups.append("Pullback")
    
    # Sector
    clean_sym = ticker.replace(".NS", "")
    my_sector = get_stock_sector(clean_sym)
    has_sector = sector_changes.get(my_sector, 0) > 0.5
    if has_sector: setups.append("Sector Support")

    # Risk Management
    atr = float(curr['ATR'])
    target = close + (3 * atr) # 1:3 Target
    stop = close - (1 * atr)   # 1 ATR Stop
    rr = round((target - close)/(close - stop), 1)
    
    # Calculate Suggested Quantity
    qty = calculate_qty(close, stop)

    # Verdict
    verdict = "WAIT"
    v_color = "gray"
    
    if trend == "UP" and len(setups) > 0:
        # Market Filter: Be cautious if market is Bearish
        if "BEAR" in market_regime:
            verdict = "RISKY (MKT DOWN)"
            v_color = "orange"
        elif rr >= 2.0:
            verdict = "PRIME BUY ⭐" if has_sector else "BUY"
            v_color = "purple" if has_sector else "green"
    
    # Minimal display filter
    if verdict == "WAIT" and abs((close - float(prev['Close']))/float(prev['Close'])) < 0.01: return None

    return {
        "symbol": clean_sym,
        "price": round(close, 2),
        "change": round(((close - float(prev['Close']))/float(prev['Close']))*100, 2),
        "sector": my_sector,
        "setups": setups,
        "verdict": verdict,
        "v_color": v_color,
        "rr": rr,
        "qty": qty, # NEW: Position Sizing
        "investment": round(qty * close, 0),
        "levels": {"TGT": round(target, 1), "SL": round(stop, 1)},
        "history": df['Close'].tail(30).tolist()
    }

# --- 4. PORTFOLIO TRACKER (History + TSL) ---
def update_history(current_signals, bulk_data):
    history_file = "trade_history.json"
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f: history = json.load(f)
        except: pass
        
    today_str = date.today().strftime("%Y-%m-%d")
    
    # A. Update Existing Trades (Trailing Logic)
    for trade in history:
        if trade['status'] == 'OPEN':
            ticker = trade['symbol'] + ".NS"
            df = extract_stock_df(bulk_data, ticker)
            if df is not None:
                curr = df.iloc[-1]
                high = float(curr['High'])
                low = float(curr['Low'])
                close = float(curr['Close'])
                
                # Check Outcomes
                if low <= trade['stop_loss']:
                    trade['status'] = 'LOSS'
                    trade['exit_price'] = trade['stop_loss']
                elif high >= trade['target']:
                    trade['status'] = 'WIN'
                    trade['exit_price'] = trade['target']
                else:
                    # TRAILING STOP LOGIC
                    # If price moved 50% towards target, move SL to Entry (Breakeven)
                    entry = trade['entry']
                    target = trade['target']
                    move_needed = (target - entry) * 0.5
                    if close > (entry + move_needed) and trade['stop_loss'] < entry:
                        trade['stop_loss'] = entry
                        trade['note'] = "TSL Moved to BE"
                        
                    # Calculate Unrealized PnL
                    trade['pnl_pct'] = round(((close - entry) / entry) * 100, 2)

    # B. Add New Trades
    existing_ids = {t['id'] for t in history}
    for s in current_signals:
        if "BUY" in s['verdict']:
            tid = f"{s['symbol']}-{today_str}"
            if tid not in existing_ids:
                history.insert(0, {
                    "id": tid, "date": today_str, "symbol": s['symbol'],
                    "entry": s['price'], "qty": s['qty'],
                    "target": s['levels']['TGT'], "stop_loss": s['levels']['SL'],
                    "status": "OPEN", "pnl_pct": 0.0, "note": ""
                })

    with open(history_file, 'w') as f: json.dump(history, f, indent=2)
    
    # Stats
    wins = len([x for x in history if x['status'] == 'WIN'])
    total = len([x for x in history if x['status'] in ['WIN', 'LOSS']])
    acc = round((wins/total*100),1) if total > 0 else 0
    open_pos = [x for x in history if x['status'] == 'OPEN']
    
    return open_pos, {"accuracy": acc, "total_trades": total}

# --- 5. GENERATE HTML ---
def generate_html(stocks, sectors, market_regime, open_positions, stats, updated_time):
    stocks.sort(key=lambda x: (x['verdict'] != "PRIME BUY ⭐", "BUY" not in x['verdict'], -x['change']))
    
    json_data = json.dumps({"stocks": stocks, "pos": open_positions})
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PrimeTrade Fund</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            body {{ background: #0f172a; color: #e2e8f0; font-family: sans-serif; }}
            .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px; }}
            .prime {{ border: 1px solid #a855f7; background: linear-gradient(to bottom right, #1e293b, #2e1065); }}
            .badge {{ font-size: 9px; padding: 2px 6px; border-radius: 4px; font-weight: bold; }}
        </style>
    </head>
    <body class="p-4 md:p-8">
        <div class="max-w-7xl mx-auto">
            <!-- Market Header -->
            <div class="flex justify-between items-center mb-6 bg-slate-900 p-4 rounded-xl border border-slate-800">
                <div>
                    <h1 class="text-2xl font-bold text-white flex items-center gap-2">
                        <i data-lucide="briefcase" class="text-purple-500"></i> PrimeTrade Fund
                    </h1>
                    <div class="text-xs text-slate-500 mt-1">{updated_time} • {market_regime}</div>
                </div>
                <div class="text-right">
                    <div class="text-[10px] text-slate-500 uppercase">Strategy Accuracy</div>
                    <div class="text-2xl font-bold text-green-400">{stats['accuracy']}%</div>
                    <div class="text-[10px] text-slate-600">{stats['total_trades']} Closed Trades</div>
                </div>
            </div>

            <!-- Active Portfolio (From History) -->
            <h2 class="text-sm font-bold text-slate-400 mb-3 uppercase tracking-wider">Active Portfolio</h2>
            <div id="portfolio-area" class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8"></div>

            <!-- New Signals -->
            <h2 class="text-sm font-bold text-slate-400 mb-3 uppercase tracking-wider">New Signals Scanner</h2>
            <div id="signal-area" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4"></div>
        </div>

        <script>
            const DATA = {json_data};
            
            // Render Portfolio
            const portRoot = document.getElementById('portfolio-area');
            if(DATA.pos.length === 0) portRoot.innerHTML = '<div class="col-span-full text-center py-4 text-slate-600 italic">No open positions.</div>';
            else {{
                portRoot.innerHTML = DATA.pos.map(p => {{
                    const pnlClass = p.pnl_pct >= 0 ? 'text-green-400' : 'text-red-400';
                    return `<div class="bg-slate-800 border border-slate-700 rounded-lg p-4 relative overflow-hidden">
                        <div class="flex justify-between mb-2">
                            <div class="font-bold text-white">${{p.symbol}}</div>
                            <div class="font-mono font-bold ${{pnlClass}}">${{p.pnl_pct}}%</div>
                        </div>
                        <div class="text-xs text-slate-400 flex justify-between mb-2">
                            <span>Entry: ${{p.entry}}</span>
                            <span>Qty: ${{p.qty}}</span>
                        </div>
                        <div class="w-full bg-slate-700 h-1.5 rounded-full mt-2">
                            <div class="bg-blue-500 h-1.5 rounded-full" style="width: 50%"></div> 
                        </div>
                        <div class="flex justify-between text-[10px] mt-1 text-slate-500 font-mono">
                            <span class="text-red-400">${{p.stop_loss}} (SL)</span>
                            <span class="text-green-400">${{p.target}} (TGT)</span>
                        </div>
                        ${{p.note ? `<div class="absolute top-0 right-0 bg-blue-600 text-[9px] text-white px-2 py-0.5 rounded-bl">TSL ACTIVE</div>` : ''}}
                    </div>`;
                }}).join('');
            }}

            // Render Signals
            const sigRoot = document.getElementById('signal-area');
            if(DATA.stocks.length === 0) sigRoot.innerHTML = '<div class="col-span-full text-center py-10 text-slate-500">No new setups found.</div>';
            else {{
                sigRoot.innerHTML = DATA.stocks.map(s => {{
                    const isPrime = s.verdict.includes('PRIME');
                    const pts = s.history.map((p,i) => `${{(i/29)*100}},${{30-((p-Math.min(...s.history))/(Math.max(...s.history)-Math.min(...s.history)||1))*30}}`).join(' ');
                    
                    return `<div class="card ${{isPrime ? 'prime' : ''}}">
                        <div class="flex justify-between mb-2">
                            <div><div class="font-bold text-white">${{s.symbol}}</div><div class="text-[10px] text-slate-400">${{s.sector}}</div></div>
                            <div class="text-right"><div class="font-bold ${{s.change>=0?'text-green-400':'text-red-400'}}">${{s.change}}%</div><div class="text-[10px] text-slate-500">₹${{s.price}}</div></div>
                        </div>
                        <div class="flex justify-between bg-slate-900/50 p-2 rounded mb-3 text-[10px]">
                            <div class="text-center">
                                <div class="text-slate-500">Rec. Qty</div>
                                <div class="font-bold text-blue-400 text-lg">${{s.qty}}</div>
                            </div>
                            <div class="text-center border-l border-slate-700 pl-2">
                                <div class="text-slate-500">Inv. Amt</div>
                                <div class="font-bold text-slate-300">₹${{s.investment}}</div>
                            </div>
                        </div>
                        <div class="flex justify-between items-end border-t border-slate-700 pt-2">
                            <div class="font-bold text-xs ${{s.v_color==='purple'?'text-purple-400':'text-green-400'}}">${{s.verdict.replace('⭐','')}}</div>
                            <div class="text-[10px] font-mono text-slate-400">SL: <span class="text-red-400">${{s.levels.SL}}</span></div>
                        </div>
                        <svg class="mt-2 opacity-20" height="30" width="100%"><polyline points="${{pts}}" fill="none" stroke="${{s.change>=0?'#4ade80':'#f87171'}}" stroke-width="2"/></svg>
                    </div>`;
                }}).join('');
            }}
            lucide.createIcons();
        </script>
    </body>
    </html>
    """
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    with open(FILE_PATH, "w", encoding="utf-8") as f: f.write(html)

if __name__ == "__main__":
    tickers = get_tickers()
    bulk_data = fetch_bulk_data(tickers)
    
    sector_changes = {}
    results = []
    market_trend = "UNKNOWN"

    if not bulk_data.empty:
        market_trend = analyze_market_trend(bulk_data)
        
        # Sectors
        for name, ticker in SECTOR_INDICES.items():
            df = extract_stock_df(bulk_data, ticker)
            if df is not None and len(df)>1:
                s = df['Close']
                sector_changes[name] = round(((s.iloc[-1]-s.iloc[-2])/s.iloc[-2])*100, 2)
        
        # Stocks
        if isinstance(bulk_data.columns, pd.MultiIndex):
            cols = bulk_data.columns.get_level_values(0).unique()
        else:
            cols = bulk_data.columns
            
        for t in cols:
            if str(t).startswith('^'): continue
            df = extract_stock_df(bulk_data, t)
            if df is not None:
                try:
                    res = analyze_ticker(t, df, sector_changes, market_trend)
                    if res: results.append(res)
                except: continue

    # Update History
    open_positions, stats = update_history(results, bulk_data)
    
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    generate_html(results, sector_changes, market_trend, open_positions, stats, timestamp)

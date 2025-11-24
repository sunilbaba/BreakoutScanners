import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import warnings
from datetime import datetime

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURATION ---
OUTPUT_DIR = "public"
FILE_PATH = os.path.join(OUTPUT_DIR, "index.html")

# 1. Sector Indices
SECTOR_INDICES = {
    "BANK": "^NSEBANK",
    "AUTO": "^CNXAUTO",
    "IT": "^CNXIT",
    "METAL": "^CNXMETAL",
    "PHARMA": "^CNXPHARMA",
    "FMCG": "^CNXFMCG",
    "ENERGY": "^CNXENERGY",
    "REALTY": "^CNXREALTY",
    "PSU BANK": "^CNXPSUBANK"
}

# 2. Map Stocks to Sectors (Expanded List)
def get_stock_sector(symbol):
    s = symbol.replace('.NS', '')
    if s in ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK", "BANKBARODA", "PNB"]: return "BANK"
    if s in ["TCS", "INFY", "HCLTECH", "WIPRO", "LTIM", "TECHM", "PERSISTENT", "COFORGE"]: return "IT"
    if s in ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO", "TVSMOTOR"]: return "AUTO"
    if s in ["TATASTEEL", "HINDALCO", "JSWSTEEL", "JINDALSTEL", "VEDL", "SAIL", "NMDC"]: return "METAL"
    if s in ["SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "LUPIN", "APOLLOHOSP", "AUROPHARMA"]: return "PHARMA"
    if s in ["ITC", "HINDUNILVR", "NESTLEIND", "BRITANNIA", "TATACONSUM", "VBL", "COLPAL", "DABUR"]: return "FMCG"
    return "Other"

# --- 1. DATA ACQUISITION ---
def get_tickers():
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        df = pd.read_csv(url, storage_options=headers)
        tickers = [f"{x}.NS" for x in df['Symbol'].dropna().unique()]
        print(f"Fetched {len(tickers)} tickers from NSE.")
        return tickers
    except Exception as e:
        print(f"Using Fallback List due to: {e}")
        return [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS", 
            "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LTIM.NS", "LT.NS", 
            "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS", "ULTRACEMCO.NS", 
            "SUNPHARMA.NS", "BAJFINANCE.NS", "HCLTECH.NS", "ADANIENT.NS", "TATASTEEL.NS",
            "JIOFIN.NS", "ZOMATO.NS", "DLF.NS", "HAL.NS", "VBL.NS", "TRENT.NS", "BEL.NS",
            "POWERGRID.NS", "ONGC.NS", "NTPC.NS", "COALINDIA.NS", "BPCL.NS"
        ]

def fetch_bulk_data(tickers):
    """
    Downloads data using group_by='ticker' which is the safest structure.
    """
    all_tickers = tickers + list(SECTOR_INDICES.values())
    print(f"Downloading data for {len(all_tickers)} symbols...")
    
    try:
        # group_by='ticker' ensures we get a DataFrame where top level columns are Tickers
        data = yf.download(all_tickers, period="6mo", group_by='ticker', threads=True, progress=True)
        return data
    except Exception as e:
        print(f"Bulk download failed: {e}")
        return pd.DataFrame()

# --- 2. ANALYSIS ENGINE ---
def calculate_indicators(df):
    # Ensure sorted by date
    df = df.sort_index()
    
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    df['VolAvg'] = df['Volume'].rolling(20).mean()
    return df

def analyze_ticker(ticker, df, sector_changes):
    # Data Cleaning: Drop rows with NaN in Close
    df = df.dropna(subset=['Close'])
    
    # We need enough data for 200 SMA
    if len(df) < 200: return None
    
    df = calculate_indicators(df)
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    close = float(curr['Close'])
    prev_close = float(prev['Close'])
    
    if prev_close == 0: return None
    change_pct = round(((close - prev_close) / prev_close) * 100, 2)
    
    clean_sym = ticker.replace(".NS", "")
    
    # --- LOGIC ---
    setups = []
    
    # Trend Check
    sma200 = float(curr['SMA200'])
    trend = "UP" if close > sma200 else "DOWN"
    
    # Setup 1: Momentum Burst
    ema20 = float(curr['EMA20'])
    if close > ema20 and curr['RSI'] > 60 and curr['Volume'] > (curr['VolAvg'] * 1.5):
        setups.append("Momentum Burst")

    # Setup 2: Pullback
    sma50 = float(curr['SMA50'])
    dist_to_50 = abs(close - sma50) / close
    if trend == "UP" and close > sma50 and dist_to_50 < 0.02 and curr['RSI'] < 55:
        setups.append("Pullback Buy")

    # Setup 3: Golden Cross
    if df['SMA50'].iloc[-1] > df['SMA200'].iloc[-1] and df['SMA50'].iloc[-2] < df['SMA200'].iloc[-2]:
         setups.append("Golden Cross")

    # Setup 4: NR7 Breakout
    ranges = df['High'] - df['Low']
    if ranges.iloc[-2] == ranges.iloc[-8:-1].min() and close > prev['High']:
        setups.append("NR7 Breakout")
            
    # Gap Trap Warning
    if (curr['Open'] - prev_close) / prev_close > 0.02:
        setups.append("⚠️ GAP UP")

    # Sector Match
    my_sector = get_stock_sector(clean_sym)
    sector_strength = sector_changes.get(my_sector, 0)
    has_sector_support = sector_strength > 0.5
    if has_sector_support: setups.append(f"Sector Support (+{sector_strength}%)")

    # Risk Reward
    atr = float(curr['ATR'])
    target = close + (3 * atr)
    stop = close - (0.75 * atr)
    
    rr_ratio = 0
    risk = close - stop
    if risk > 0:
        rr_ratio = round((target - close) / risk, 1)

    # Verdict
    verdict = "WAIT"
    v_color = "gray"
    
    # Only show Buy if Setup exists
    if trend == "UP" and len(setups) > 0 and "⚠️ GAP UP" not in setups:
        if rr_ratio >= 1.5:
            verdict = "PRIME BUY ⭐" if has_sector_support else "BUY"
            v_color = "purple" if has_sector_support else "green"
        else:
            verdict = "BAD R:R"
            v_color = "orange"
    elif trend == "DOWN" and len(setups) > 0:
        verdict = "CTR-TREND"
        v_color = "blue"

    # Minimal filter: Show stock if there is a verdict OR significant movement
    if verdict == "WAIT" and abs(change_pct) < 1.5: return None

    return {
        "symbol": clean_sym,
        "price": round(close, 2),
        "change": change_pct,
        "sector": my_sector,
        "setups": setups,
        "verdict": verdict,
        "v_color": v_color,
        "rr": rr_ratio,
        "levels": {"TGT": round(target, 1), "SL": round(stop, 1)},
        "history": df['Close'].tail(30).tolist()
    }

# --- 3. HTML GENERATION ---
def generate_html(stocks, sector_data, updated):
    adv = len([x for x in stocks if x['change'] > 0])
    dec = len([x for x in stocks if x['change'] < 0])
    
    # Sort: Prime first, then Buy, then by Change%
    def sort_key(x):
        v_rank = 0
        if "PRIME" in x['verdict']: v_rank = 3
        elif "BUY" in x['verdict'] and "BAD" not in x['verdict']: v_rank = 2
        elif "CTR" in x['verdict']: v_rank = 1
        return (v_rank, x['change'])
        
    stocks.sort(key=sort_key, reverse=True)

    json_data = json.dumps({
        "stocks": stocks,
        "sectors": [{"name": k, "avg": v} for k,v in sector_data.items()],
        "market": {"adv": adv, "dec": dec}
    })

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="refresh" content="300">
        <title>PrimeTrade</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            body {{ background: #0b0e14; color: #e2e8f0; font-family: 'Inter', sans-serif; }}
            .card {{ background: #151921; border: 1px solid #2d3342; border-radius: 12px; padding: 16px; position: relative; overflow: hidden; }}
            .card:hover {{ border-color: #4b5563; transform: translateY(-2px); transition: all 0.2s; }}
            
            .prime {{ border: 1px solid #9333ea; background: linear-gradient(135deg, #151921 60%, #2e1065 100%); }}
            .prime::before {{ content: ""; position: absolute; left:0; top:0; bottom:0; width: 3px; background: #a855f7; }}
            
            .badge {{ font-size: 10px; padding: 2px 6px; border-radius: 4px; font-weight: 600; margin-right: 4px; display: inline-block; margin-bottom: 4px; }}
            .badge-sec {{ background: #172554; color: #60a5fa; border: 1px solid #1e40af; }}
            .badge-mom {{ background: #064e3b; color: #34d399; border: 1px solid #059669; }}
            .badge-gap {{ background: #450a0a; color: #fca5a5; border: 1px solid #b91c1c; }}
        </style>
    </head>
    <body class="p-4 lg:p-8">
        <div class="max-w-7xl mx-auto">
            
            <!-- Header -->
            <div class="flex flex-col md:flex-row justify-between items-center mb-8 border-b border-slate-800 pb-6">
                <div>
                    <h1 class="text-2xl md:text-3xl font-bold text-white tracking-tight flex items-center gap-2">
                        <i data-lucide="scan-search" class="text-purple-500"></i> PrimeTrade
                    </h1>
                    <div class="text-xs text-slate-500 mt-1 font-mono">{updated} • Auto-Refresh 5m</div>
                </div>
                <div class="mt-4 md:mt-0 flex gap-4">
                    <div class="text-center px-4">
                        <div class="text-[10px] text-slate-500 uppercase tracking-widest">Advances</div>
                        <div class="text-xl font-bold text-green-500">{adv}</div>
                    </div>
                    <div class="w-px bg-slate-800"></div>
                    <div class="text-center px-4">
                        <div class="text-[10px] text-slate-500 uppercase tracking-widest">Declines</div>
                        <div class="text-xl font-bold text-red-500">{dec}</div>
                    </div>
                </div>
            </div>

            <!-- Sectors -->
            <div id="sector-row" class="grid grid-cols-3 sm:grid-cols-5 lg:grid-cols-9 gap-2 mb-8"></div>

            <!-- Filter Tabs -->
            <div class="flex gap-2 mb-6 p-1 bg-slate-900/50 rounded-lg inline-flex">
                <button onclick="filter('ALL')" id="btn-ALL" class="px-5 py-2 rounded-md text-xs font-bold transition-all bg-slate-700 text-white">Market</button>
                <button onclick="filter('PRIME')" id="btn-PRIME" class="px-5 py-2 rounded-md text-xs font-bold transition-all text-slate-400 hover:text-white">Prime ⭐</button>
                <button onclick="filter('BUY')" id="btn-BUY" class="px-5 py-2 rounded-md text-xs font-bold transition-all text-slate-400 hover:text-white">Buys</button>
            </div>

            <!-- Grid -->
            <div id="grid-root" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4"></div>
        </div>

        <script>
            const DATA = {json_data};
            let activeFilter = 'ALL';

            function init() {{
                const sRoot = document.getElementById('sector-row');
                sRoot.innerHTML = DATA.sectors.map(s => {{
                    const color = s.avg >= 0 ? 'text-green-400' : 'text-red-400';
                    const bg = s.avg >= 0 ? 'bg-green-500/5 border-green-500/20' : 'bg-red-500/5 border-red-500/20';
                    return `<div class="p-2 rounded border ${{bg}} text-center transition hover:bg-slate-800">
                        <div class="text-[9px] text-slate-400 font-bold uppercase tracking-wider">${{s.name}}</div>
                        <div class="text-sm font-bold ${{color}} mt-0.5">${{s.avg}}%</div>
                    </div>`;
                }}).join('');
                renderGrid();
                lucide.createIcons();
            }}

            function filter(type) {{
                activeFilter = type;
                ['ALL','PRIME','BUY'].forEach(t => {{
                    const btn = document.getElementById('btn-'+t);
                    if(t===type) btn.className = "px-5 py-2 rounded-md text-xs font-bold transition-all bg-purple-600 text-white shadow-lg shadow-purple-900/50";
                    else btn.className = "px-5 py-2 rounded-md text-xs font-bold transition-all text-slate-400 hover:text-white hover:bg-slate-800";
                }});
                renderGrid();
            }}

            function renderGrid() {{
                const root = document.getElementById('grid-root');
                const filtered = DATA.stocks.filter(s => {{
                    if(activeFilter === 'PRIME') return s.verdict.includes('PRIME');
                    if(activeFilter === 'BUY') return s.verdict.includes('BUY');
                    return true;
                }});

                if(filtered.length === 0) {{ root.innerHTML = '<div class="col-span-full text-center py-20 text-slate-600">No stocks matching criteria found.</div>'; return; }}

                root.innerHTML = filtered.map(s => {{
                    const isPrime = s.verdict.includes('PRIME');
                    const hist = s.history;
                    const min = Math.min(...hist); const max = Math.max(...hist);
                    const pts = hist.map((p, i) => `${{(i/(hist.length-1))*100}},${{40-((p-min)/(max-min||1))*40}}`).join(' ');
                    const stroke = s.change >= 0 ? '#22c55e' : '#ef4444';
                    
                    const badges = s.setups.map(b => {{
                        let cls = 'bg-slate-800 text-slate-400 border border-slate-700';
                        if(b.includes('Sector')) cls = 'badge-sec';
                        if(b.includes('Momentum')) cls = 'badge-mom';
                        if(b.includes('GAP')) cls = 'badge-gap';
                        return `<span class="badge ${{cls}}">${{b}}</span>`;
                    }}).join('');

                    let vColor = 'text-slate-500';
                    if(s.v_color === 'green') vColor = 'text-green-400';
                    if(s.v_color === 'purple') vColor = 'text-purple-400';
                    if(s.v_color === 'orange') vColor = 'text-orange-400';

                    return `
                    <div class="card ${{isPrime ? 'prime' : ''}} group cursor-default">
                        <div class="flex justify-between items-start mb-3">
                            <div>
                                <div class="font-bold text-lg text-white tracking-tight">${{s.symbol}}</div>
                                <div class="text-[10px] text-slate-400 uppercase font-bold">${{s.sector}}</div>
                            </div>
                            <div class="text-right">
                                <div class="text-xl font-mono font-bold ${{s.change>=0?'text-green-400':'text-red-400'}}">${{s.change}}%</div>
                                <div class="text-[10px] text-slate-500">₹${{s.price}}</div>
                            </div>
                        </div>
                        
                        <div class="mb-4 min-h-[24px] flex flex-wrap content-start">${{badges}}</div>
                        
                        <div class="flex justify-between items-end border-t border-slate-800 pt-3 mt-2">
                             <div>
                                <div class="text-[9px] text-slate-500 uppercase font-bold">Verdict</div>
                                <div class="text-xs font-bold ${{vColor}}">${{s.verdict.replace('⭐','')}}</div>
                             </div>
                             <div class="text-right">
                                <div class="text-[9px] text-slate-500 uppercase font-bold">R:R</div>
                                <div class="text-xs font-bold text-slate-300">${{s.rr}}</div>
                             </div>
                        </div>

                        <div class="absolute bottom-0 left-0 right-0 h-12 opacity-20 group-hover:opacity-40 transition pointer-events-none">
                            <svg width="100%" height="100%" preserveAspectRatio="none"><polyline points="${{pts}}" fill="none" stroke="${{stroke}}" stroke-width="2" vector-effect="non-scaling-stroke" /></svg>
                        </div>
                    </div>`;
                }}).join('');
                lucide.createIcons();
            }}
            init();
        </script>
    </body>
    </html>
    """
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    with open(FILE_PATH, "w", encoding="utf-8") as f: f.write(html)

if __name__ == "__main__":
    print("--- SCANNER STARTING ---")
    tickers = get_tickers()
    
    # Force group_by='ticker' to get specific structure: [Ticker] -> [Open, High...]
    bulk_data = fetch_bulk_data(tickers)
    
    sector_changes = {}
    results = []

    if not bulk_data.empty:
        # Extract Sector Changes First
        for name, ticker in SECTOR_INDICES.items():
            if ticker in bulk_data.columns: # Top level check
                try:
                    s = bulk_data[ticker]['Close'].dropna()
                    if len(s) > 1:
                        sector_changes[name] = round(((s.iloc[-1] - s.iloc[-2]) / s.iloc[-2]) * 100, 2)
                except: pass

        # Extract Stocks
        # With group_by='ticker', the columns top level is the Tickers
        # We iterate over unique values in level 0
        stock_list = bulk_data.columns.levels[0] if isinstance(bulk_data.columns, pd.MultiIndex) else bulk_data.columns

        print(f"Processing {len(stock_list)} items...")
        
        for t in stock_list:
            if t.startswith('^'): continue # Skip sector indices in the loop
            
            try:
                # Direct Access via Ticker Key
                df = bulk_data[t].copy()
                
                # Analyze
                res = analyze_ticker(t, df, sector_changes)
                if res: results.append(res)
            except Exception as e:
                continue

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"Generating Report for {len(results)} stocks...")
    generate_html(results, sector_changes, now)
    print("Done.")

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
    "NIFTY BANK": "^NSEBANK",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY IT": "^CNXIT",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY ENERGY": "^CNXENERGY",
    "NIFTY REALTY": "^CNXREALTY",
    "NIFTY PSU BANK": "^CNXPSUBANK"
}

# 2. Map Stocks to Sectors
def get_stock_sector(symbol):
    if symbol in ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK"]: return "NIFTY BANK"
    if symbol in ["TCS", "INFY", "HCLTECH", "WIPRO", "LTIM", "TECHM"]: return "NIFTY IT"
    if symbol in ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "EICHERMOT"]: return "NIFTY AUTO"
    if symbol in ["TATASTEEL", "HINDALCO", "JSWSTEEL", "JINDALSTEL", "VEDL"]: return "NIFTY METAL"
    if symbol in ["SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB"]: return "NIFTY PHARMA"
    if symbol in ["ITC", "HINDUNILVR", "NESTLEIND", "BRITANNIA"]: return "NIFTY FMCG"
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
            "JIOFIN.NS", "ZOMATO.NS", "DLF.NS", "HAL.NS", "VBL.NS", "TRENT.NS", "BEL.NS"
        ]

def fetch_bulk_data(tickers):
    """Robust download that handles yfinance structure changes."""
    all_tickers = tickers + list(SECTOR_INDICES.values())
    print(f"Downloading data for {len(all_tickers)} symbols...")
    
    try:
        # Standard download (Group by Column is safer for extracting Price types)
        data = yf.download(all_tickers, period="6mo", threads=True, progress=True)
        return data
    except Exception as e:
        print(f"Bulk download failed: {e}")
        return pd.DataFrame()

# --- 2. ANALYSIS ENGINE ---
def calculate_indicators(df):
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
    # Ensure data is numeric and clean
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    if len(df) < 200: return None
    
    df = calculate_indicators(df)
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    close = curr['Close']
    prev_close = prev['Close']
    change_pct = round(((close - prev_close) / prev_close) * 100, 2)
    
    clean_sym = ticker.replace(".NS", "")
    
    # Setups
    setups = []
    trend = "UP" if close > curr['SMA200'] else "DOWN"
    
    # Momentum
    if close > curr['EMA20'] and curr['RSI'] > 60 and curr['Volume'] > (curr['VolAvg'] * 1.5):
        setups.append("Momentum Burst")

    # Pullback
    if trend == "UP" and close > curr['SMA50'] and abs(close - curr['SMA50'])/close < 0.02 and curr['RSI'] < 55:
        setups.append("Pullback Buy")

    # Golden Cross
    if df['SMA50'].iloc[-1] > df['SMA200'].iloc[-1] and df['SMA50'].iloc[-2] < df['SMA200'].iloc[-2]:
         setups.append("Golden Cross")

    # NR7
    ranges = df['High'] - df['Low']
    if ranges.iloc[-2] == ranges.iloc[-8:-1].min() and close > prev['High']:
        setups.append("NR7 Breakout")
            
    # Gap
    if (curr['Open'] - prev_close) / prev_close > 0.02:
        setups.append("⚠️ GAP UP")

    # Sector Support
    my_sector = get_stock_sector(clean_sym)
    sector_strength = sector_changes.get(my_sector, 0)
    has_sector_support = sector_strength > 0.5
    if has_sector_support: setups.append(f"Sector Support (+{sector_strength}%)")

    # R:R
    atr = curr['ATR']
    target = close + (3 * atr)
    stop = close - (0.75 * atr)
    rr_ratio = round((target - close) / (close - stop), 1)

    # Verdict
    verdict = "WAIT"
    v_color = "gray"
    
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

    if verdict == "WAIT" and abs(change_pct) < 2.0: return None

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
    stocks.sort(key=lambda x: (x['verdict'] != "PRIME BUY ⭐", -x['change']))

    json_data = json.dumps({
        "stocks": stocks,
        "sectors": [{"name": k.replace('NIFTY ', ''), "avg": v} for k,v in sector_data.items()],
        "market": {"adv": adv, "dec": dec}
    })

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="refresh" content="300">
        <title>PrimeTrade Scanner</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            body {{ background: #0f172a; color: #e2e8f0; font-family: sans-serif; }}
            .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px; position: relative; overflow: hidden; }}
            .card:hover {{ border-color: #64748b; transform: translateY(-2px); transition: all 0.2s; }}
            .prime {{ border: 1px solid #a855f7; background: linear-gradient(to bottom right, #1e293b, #3b0764); }}
            .prime::after {{ content: "⭐ PRIME"; position: absolute; top: 0; right: 0; background: #a855f7; color: white; font-size: 10px; padding: 2px 6px; border-bottom-left-radius: 8px; font-weight: bold; }}
            .badge {{ font-size: 10px; padding: 2px 6px; border-radius: 4px; font-weight: bold; margin-right: 4px; display: inline-block; margin-bottom: 4px; }}
            .badge-std {{ background: #334155; color: #94a3b8; }}
            .badge-sec {{ background: #1e3a8a; color: #93c5fd; border: 1px solid #3b82f6; }}
            .badge-mom {{ background: #064e3b; color: #6ee7b7; border: 1px solid #10b981; }}
            .badge-gap {{ background: #450a0a; color: #fca5a5; border: 1px solid #ef4444; }}
        </style>
    </head>
    <body class="p-4 md:p-8">
        <div class="max-w-7xl mx-auto">
            <div class="flex flex-col md:flex-row justify-between items-start md:items-end mb-6 border-b border-slate-800 pb-4">
                <div>
                    <h1 class="text-3xl font-bold text-white flex items-center gap-2"><i data-lucide="radar" class="text-blue-500"></i> PrimeTrade Scanner</h1>
                    <div class="text-sm text-slate-500 mt-1">Updated: {updated} (IST) • Auto-Refresh: 5m</div>
                </div>
                <div class="mt-4 md:mt-0 flex items-center gap-3 bg-slate-800 px-4 py-2 rounded border border-slate-700">
                    <span class="text-green-400 font-bold">ADV {adv}</span> <span class="text-slate-600">|</span> <span class="text-red-400 font-bold">DEC {dec}</span>
                </div>
            </div>
            <div id="sector-row" class="grid grid-cols-3 md:grid-cols-6 lg:grid-cols-9 gap-2 mb-8"></div>
            <div class="flex gap-2 mb-6">
                <button onclick="filter('ALL')" id="btn-ALL" class="px-4 py-1.5 rounded-full text-xs font-bold bg-blue-600 text-white">All</button>
                <button onclick="filter('PRIME')" id="btn-PRIME" class="px-4 py-1.5 rounded-full text-xs font-bold bg-slate-800 text-slate-400 border border-slate-700">Prime ⭐</button>
                <button onclick="filter('BUY')" id="btn-BUY" class="px-4 py-1.5 rounded-full text-xs font-bold bg-slate-800 text-slate-400 border border-slate-700">Valid Buys</button>
            </div>
            <div id="grid-root" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4"></div>
        </div>
        <script>
            const DATA = {json_data};
            let activeFilter = 'ALL';
            function init() {{
                const sRoot = document.getElementById('sector-row');
                sRoot.innerHTML = DATA.sectors.map(s => {{
                    const color = s.avg >= 0 ? 'text-green-400' : 'text-red-400';
                    const bg = s.avg >= 0 ? 'bg-green-900/10 border-green-900/30' : 'bg-red-900/10 border-red-900/30';
                    return `<div class="px-2 py-2 rounded border ${{bg}} flex flex-col items-center justify-center"><span class="text-[10px] text-slate-400 font-bold">${{s.name}}</span><span class="text-xs font-bold ${{color}}">${{s.avg}}%</span></div>`;
                }}).join('');
                renderGrid();
                lucide.createIcons();
            }}
            function filter(type) {{
                activeFilter = type;
                ['ALL','PRIME','BUY'].forEach(t => {{ document.getElementById('btn-'+t).className = (t===type) ? "px-4 py-1.5 rounded-full text-xs font-bold bg-blue-600 text-white" : "px-4 py-1.5 rounded-full text-xs font-bold bg-slate-800 text-slate-400 border border-slate-700"; }});
                renderGrid();
            }}
            function renderGrid() {{
                const root = document.getElementById('grid-root');
                const filtered = DATA.stocks.filter(s => {{
                    if(activeFilter === 'PRIME') return s.verdict.includes('PRIME');
                    if(activeFilter === 'BUY') return s.verdict.includes('BUY');
                    return true;
                }});
                if(filtered.length === 0) {{ root.innerHTML = '<div class="col-span-full text-center py-20 text-slate-500">No stocks found.</div>'; return; }}
                root.innerHTML = filtered.map(s => {{
                    const isPrime = s.verdict.includes('PRIME');
                    const hist = s.history;
                    const min = Math.min(...hist); const max = Math.max(...hist);
                    const pts = hist.map((p, i) => `${{(i/(hist.length-1))*100}},${{40-((p-min)/(max-min||1))*40}}`).join(' ');
                    const stroke = s.change >= 0 ? '#4ade80' : '#f87171';
                    const badges = s.setups.map(b => {{
                        let cls = 'badge-std';
                        if(b.includes('Sector')) cls = 'badge-sec';
                        if(b.includes('Momentum')) cls = 'badge-mom';
                        if(b.includes('GAP')) cls = 'badge-gap';
                        return `<span class="badge ${{cls}}">${{b}}</span>`;
                    }}).join('');
                    let vColor = 'text-slate-400';
                    if(s.v_color === 'green') vColor = 'text-green-400';
                    if(s.v_color === 'purple') vColor = 'text-purple-300';
                    if(s.v_color === 'orange') vColor = 'text-orange-400';
                    return `<div class="card ${{isPrime ? 'prime' : ''}} group">
                        <div class="flex justify-between items-start mb-2">
                            <div><div class="font-bold text-lg text-white group-hover:text-blue-400 transition">${{s.symbol}}</div><div class="text-xs text-slate-400">₹${{s.price}} <span class="${{s.change>=0?'text-green-400':'text-red-400'}} ml-1">${{s.change}}%</span></div></div>
                            <div class="text-right"><div class="font-bold text-xs ${{vColor}}">${{s.verdict.replace('⭐','')}}</div><div class="text-[10px] text-slate-500">R:R <span class="text-white">${{s.rr}}</span></div></div>
                        </div>
                        <div class="mb-4 min-h-[20px]">${{badges}}</div>
                        <div class="grid grid-cols-2 gap-2 text-[10px] font-mono border-t border-slate-700/50 pt-2 mb-2"><div><span class="text-slate-500">STOP</span> <span class="text-red-400">₹${{s.levels.SL}}</span></div><div class="text-right"><span class="text-slate-500">TARGET</span> <span class="text-green-400">₹${{s.levels.TGT}}</span></div></div>
                        <div class="absolute bottom-0 left-0 right-0 h-10 opacity-30 group-hover:opacity-50 transition"><svg width="100%" height="100%" preserveAspectRatio="none"><polyline points="${{pts}}" fill="none" stroke="${{stroke}}" stroke-width="2" /></svg></div>
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
    print("--- SCANNER START ---")
    tickers = get_tickers()
    bulk_data = fetch_bulk_data(tickers)
    
    sector_changes = {}
    results = []

    if not bulk_data.empty:
        # ROBUST DATA EXTRACTION
        # 1. Identify valid tickers (columns)
        # 2. Reconstruct individual stock DFs safely
        
        # Check if MultiIndex or Single Index
        if isinstance(bulk_data.columns, pd.MultiIndex):
            # 'Close' is likely Level 0 or Level 1. 
            # We want to iterate over the 'Ticker' level.
            # Usually yfinance download(..., group_by='column') (default) -> (Price, Ticker)
            # If so, tickers are in level 1.
            # But let's check which level has 'Close'
            
            # Safe way: Extract 'Close' dataframe first
            try:
                close_df = bulk_data['Close'] # Works if 'Close' is top level
                valid_tickers = close_df.columns
            except KeyError:
                # Maybe Ticker is top level (group_by='ticker')
                # But we used default download.
                # Let's try xs or direct access
                try:
                    close_df = bulk_data.xs('Close', axis=1, level=0)
                    valid_tickers = close_df.columns
                except:
                    valid_tickers = []
                    print("Could not parse dataframe columns.")

        else:
             # Single ticker downloaded or Flat format
             valid_tickers = [] # Edge case
             print("Data is not MultiIndex.")

        # Process Sectors First
        for name, ticker in SECTOR_INDICES.items():
            if ticker in valid_tickers:
                try:
                    s_series = bulk_data['Close'][ticker].dropna()
                    if len(s_series) >= 2:
                        sector_changes[name] = round(((s_series.iloc[-1] - s_series.iloc[-2]) / s_series.iloc[-2]) * 100, 2)
                except: pass

        # Process Stocks
        print("Analyzing stocks...")
        for t in valid_tickers:
            if t.startswith('^'): continue # Skip indices
            
            try:
                # Reconstruct DF for this specific ticker
                # We need Open, High, Low, Close, Volume
                # Accessing via bulk_data['PriceType'][Ticker]
                stock_df = pd.DataFrame({
                    'Open': bulk_data['Open'][t],
                    'High': bulk_data['High'][t],
                    'Low': bulk_data['Low'][t],
                    'Close': bulk_data['Close'][t],
                    'Volume': bulk_data['Volume'][t]
                })
                
                res = analyze_ticker(t, stock_df, sector_changes)
                if res: results.append(res)
            except Exception as e:
                # print(f"Error analyzing {t}: {e}")
                continue

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    generate_html(results, sector_changes, now)
    print(f"Scan Complete. Found {len(results)} stocks.")

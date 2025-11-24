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

# 1. Sector Indices (For Heatmap & Confirmation)
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

# 2. Map Stocks to Sectors (Crucial for "Prime Buy" logic)
# This is a helper to map common stocks. If a stock isn't here, it defaults to "Other".
def get_stock_sector(symbol):
    # Simplified mapping for demonstration. In a real app, you might map all 500.
    if symbol in ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK"]: return "NIFTY BANK"
    if symbol in ["TCS", "INFY", "HCLTECH", "WIPRO", "LTIM", "TECHM"]: return "NIFTY IT"
    if symbol in ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "EICHERMOT"]: return "NIFTY AUTO"
    if symbol in ["TATASTEEL", "HINDALCO", "JSWSTEEL", "JINDALSTEL", "VEDL"]: return "NIFTY METAL"
    if symbol in ["SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB"]: return "NIFTY PHARMA"
    if symbol in ["ITC", "HINDUNILVR", "NESTLEIND", "BRITANNIA"]: return "NIFTY FMCG"
    return "Other"

# --- 1. DATA ACQUISITION ---
def get_tickers():
    """Fetches Nifty 500 tickers."""
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        df = pd.read_csv(url, storage_options=headers)
        tickers = [f"{x}.NS" for x in df['Symbol'].dropna().unique()]
        print(f"Fetched {len(tickers)} tickers from NSE.")
        return tickers
    except Exception as e:
        print(f"Using Fallback List due to: {e}")
        # Fallback list ensuring we always have something to scan
        return [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS", 
            "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LTIM.NS", "LT.NS", 
            "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS", "ULTRACEMCO.NS", 
            "SUNPHARMA.NS", "BAJFINANCE.NS", "HCLTECH.NS", "ADANIENT.NS", "TATASTEEL.NS",
            "JIOFIN.NS", "ZOMATO.NS", "DLF.NS", "HAL.NS", "VBL.NS", "TRENT.NS", "BEL.NS"
        ]

def fetch_bulk_data(tickers):
    """Downloads data for ALL tickers + Sectors in ONE request."""
    # Add Sector Indices to the download list so we get them in the same batch (faster)
    all_tickers = tickers + list(SECTOR_INDICES.values())
    print(f"Downloading data for {len(all_tickers)} symbols...")
    try:
        data = yf.download(all_tickers, period="6mo", group_by='ticker', threads=True, progress=True)
        return data
    except Exception as e:
        print(f"Bulk download failed: {e}")
        return pd.DataFrame()

# --- 2. ANALYSIS ENGINE ---
def calculate_indicators(df):
    """Adds technical indicators."""
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ATR (Volatility)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # Volume Average
    df['VolAvg'] = df['Volume'].rolling(20).mean()
    
    return df

def analyze_ticker(ticker, df, sector_changes):
    """Analyzes a single stock with ALL previous logic + Sector Awareness."""
    # Clean data
    df = df.dropna()
    if len(df) < 200: return None
    
    df = calculate_indicators(df)
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    close = curr['Close']
    prev_close = prev['Close']
    change_pct = round(((close - prev_close) / prev_close) * 100, 2)
    
    clean_sym = ticker.replace(".NS", "")
    
    # --- 1. SETUPS ---
    setups = []
    trend = "UP" if close > curr['SMA200'] else "DOWN"
    
    # Setup A: Momentum Burst (Price > EMA20, High Vol, Strong RSI)
    if close > curr['EMA20'] and curr['RSI'] > 60 and curr['Volume'] > (curr['VolAvg'] * 1.5):
        setups.append("Momentum Burst")

    # Setup B: Pullback (Trend UP, Price dipped to SMA50, RSI cooled)
    dist_to_50 = abs(close - curr['SMA50']) / close
    if trend == "UP" and close > curr['SMA50'] and dist_to_50 < 0.02 and curr['RSI'] < 55:
        setups.append("Pullback Buy")

    # Setup C: Golden Cross (SMA50 crossed SMA200 recently)
    for i in range(1, 4):
        if df['SMA50'].iloc[-i] > df['SMA200'].iloc[-i] and df['SMA50'].iloc[-(i+1)] < df['SMA200'].iloc[-(i+1)]:
            setups.append("Golden Cross")
            break

    # Setup D: NR7 Breakout (Low volatility yesterday, breakout today)
    ranges = df['High'] - df['Low']
    if ranges.iloc[-2] == ranges.iloc[-8:-1].min():
        if close > prev['High']:
            setups.append("NR7 Breakout")
            
    # Pattern: Gap Warning (Trap)
    if (curr['Open'] - prev_close) / prev_close > 0.02:
        setups.append("⚠️ GAP UP")

    # --- 2. SECTOR CONFIRMATION (The Kingmaker) ---
    my_sector = get_stock_sector(clean_sym)
    sector_strength = sector_changes.get(my_sector, 0)
    
    # If Sector is Strong (> 0.5%) AND Stock is moving, it's a huge plus
    has_sector_support = False
    if sector_strength > 0.5:
        has_sector_support = True
        setups.append(f"Sector Support (+{sector_strength}%)")

    # --- 3. RISK / REWARD ---
    atr = curr['ATR']
    # Target: 3x ATR (Aim high)
    target = close + (3 * atr)
    # Stop: 0.75x ATR (Tight stop)
    stop = close - (0.75 * atr)
    
    potential_reward = target - close
    potential_risk = close - stop
    
    rr_ratio = 0
    if potential_risk > 0:
        rr_ratio = round(potential_reward / potential_risk, 1)

    # --- 4. VERDICT ---
    verdict = "WAIT"
    v_color = "gray"
    
    # Logic: Trend UP + At least 1 Setup + No Gaps
    if trend == "UP" and len(setups) > 0 and "⚠️ GAP UP" not in setups:
        # Check RR
        if rr_ratio >= 1.5:
            if has_sector_support:
                verdict = "PRIME BUY ⭐"
                v_color = "purple"
            else:
                verdict = "BUY"
                v_color = "green"
        else:
            verdict = "BAD R:R"
            v_color = "orange"
            
    elif trend == "DOWN" and len(setups) > 0:
        verdict = "CTR-TREND"
        v_color = "blue"

    # Filter: Only show interesting stocks
    if verdict == "WAIT" and abs(change_pct) < 2.0:
        return None

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
    # Stats
    adv = len([x for x in stocks if x['change'] > 0])
    dec = len([x for x in stocks if x['change'] < 0])
    
    # Sort: Prime Buys first, then highest change
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
            
            /* Special styling for PRIME picks */
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
            <!-- Header -->
            <div class="flex flex-col md:flex-row justify-between items-start md:items-end mb-6 border-b border-slate-800 pb-4">
                <div>
                    <h1 class="text-3xl font-bold text-white flex items-center gap-2">
                        <i data-lucide="radar" class="text-blue-500"></i> PrimeTrade Scanner
                    </h1>
                    <div class="text-sm text-slate-500 mt-1">Updated: {updated} (IST) • Auto-Refresh: 5m</div>
                </div>
                <div class="mt-4 md:mt-0 flex items-center gap-3 bg-slate-800 px-4 py-2 rounded border border-slate-700">
                    <span class="text-green-400 font-bold">ADV {adv}</span> <span class="text-slate-600">|</span> <span class="text-red-400 font-bold">DEC {dec}</span>
                </div>
            </div>

            <!-- Sectors -->
            <div id="sector-row" class="grid grid-cols-3 md:grid-cols-6 lg:grid-cols-9 gap-2 mb-8"></div>

            <!-- Filters -->
            <div class="flex gap-2 mb-6">
                <button onclick="filter('ALL')" id="btn-ALL" class="px-4 py-1.5 rounded-full text-xs font-bold bg-blue-600 text-white">All</button>
                <button onclick="filter('PRIME')" id="btn-PRIME" class="px-4 py-1.5 rounded-full text-xs font-bold bg-slate-800 text-slate-400 border border-slate-700">Prime ⭐</button>
                <button onclick="filter('BUY')" id="btn-BUY" class="px-4 py-1.5 rounded-full text-xs font-bold bg-slate-800 text-slate-400 border border-slate-700">Valid Buys</button>
            </div>

            <!-- Grid -->
            <div id="grid-root" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4"></div>
        </div>

        <script>
            const DATA = {json_data};
            let activeFilter = 'ALL';

            function init() {{
                // Render Sectors
                const sRoot = document.getElementById('sector-row');
                sRoot.innerHTML = DATA.sectors.map(s => {{
                    const color = s.avg >= 0 ? 'text-green-400' : 'text-red-400';
                    const bg = s.avg >= 0 ? 'bg-green-900/10 border-green-900/30' : 'bg-red-900/10 border-red-900/30';
                    return `<div class="px-2 py-2 rounded border ${{bg}} flex flex-col items-center justify-center">
                        <span class="text-[10px] text-slate-400 font-bold">${{s.name}}</span>
                        <span class="text-xs font-bold ${{color}}">${{s.avg}}%</span>
                    </div>`;
                }}).join('');

                renderGrid();
                lucide.createIcons();
            }}

            function filter(type) {{
                activeFilter = type;
                ['ALL','PRIME','BUY'].forEach(t => {{
                    document.getElementById('btn-'+t).className = (t===type) ? 
                        "px-4 py-1.5 rounded-full text-xs font-bold bg-blue-600 text-white" : 
                        "px-4 py-1.5 rounded-full text-xs font-bold bg-slate-800 text-slate-400 border border-slate-700";
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

                if(filtered.length === 0) {{ root.innerHTML = '<div class="col-span-full text-center py-20 text-slate-500">No stocks found for this filter.</div>'; return; }}

                root.innerHTML = filtered.map(s => {{
                    const isPrime = s.verdict.includes('PRIME');
                    
                    // Sparkline
                    const hist = s.history;
                    const min = Math.min(...hist); const max = Math.max(...hist);
                    const pts = hist.map((p, i) => `${{(i/(hist.length-1))*100}},${{40-((p-min)/(max-min||1))*40}}`).join(' ');
                    const stroke = s.change >= 0 ? '#4ade80' : '#f87171';

                    // Badges
                    const badges = s.setups.map(b => {{
                        let cls = 'badge-std';
                        if(b.includes('Sector')) cls = 'badge-sec';
                        if(b.includes('Momentum')) cls = 'badge-mom';
                        if(b.includes('GAP')) cls = 'badge-gap';
                        return `<span class="badge ${{cls}}">${{b}}</span>`;
                    }}).join('');

                    // Verdict Color
                    let vColor = 'text-slate-400';
                    if(s.v_color === 'green') vColor = 'text-green-400';
                    if(s.v_color === 'purple') vColor = 'text-purple-300';
                    if(s.v_color === 'orange') vColor = 'text-orange-400';

                    return `
                    <div class="card ${{isPrime ? 'prime' : ''}} group">
                        <div class="flex justify-between items-start mb-2">
                            <div>
                                <div class="font-bold text-lg text-white group-hover:text-blue-400 transition">${{s.symbol}}</div>
                                <div class="text-xs text-slate-400">₹${{s.price}} <span class="${{s.change>=0?'text-green-400':'text-red-400'}} ml-1">${{s.change}}%</span></div>
                            </div>
                            <div class="text-right">
                                <div class="font-bold text-xs ${{vColor}}">${{s.verdict.replace('⭐','')}}</div>
                                <div class="text-[10px] text-slate-500">R:R <span class="text-white">${{s.rr}}</span></div>
                            </div>
                        </div>
                        
                        <div class="mb-4 min-h-[20px]">${{badges}}</div>
                        
                        <div class="grid grid-cols-2 gap-2 text-[10px] font-mono border-t border-slate-700/50 pt-2 mb-2">
                            <div><span class="text-slate-500">STOP</span> <span class="text-red-400">₹${{s.levels.SL}}</span></div>
                            <div class="text-right"><span class="text-slate-500">TARGET</span> <span class="text-green-400">₹${{s.levels.TGT}}</span></div>
                        </div>

                        <div class="absolute bottom-0 left-0 right-0 h-10 opacity-30 group-hover:opacity-50 transition">
                            <svg width="100%" height="100%" preserveAspectRatio="none"><polyline points="${{pts}}" fill="none" stroke="${{stroke}}" stroke-width="2" /></svg>
                        </div>
                    </div>
                    `;
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
    print("--- MASTER SCANNER STARTED ---")
    
    tickers = get_tickers()
    bulk_data = fetch_bulk_data(tickers)
    
    # Process Sector Performance first (from the bulk download)
    sector_changes = {}
    if not bulk_data.empty:
        # Check if we have multi-index columns (yfinance structure)
        is_multi = isinstance(bulk_data.columns, pd.MultiIndex)
        
        # Extract Sector Data
        for name, index_ticker in SECTOR_INDICES.items():
            try:
                if is_multi:
                    # Access specific ticker from multi-index
                    df = bulk_data[index_ticker]
                else:
                    # Should not happen in bulk mode, but safety check
                    continue
                    
                series = df['Close'].dropna()
                if len(series) >= 2:
                    change = ((series.iloc[-1] - series.iloc[-2]) / series.iloc[-2]) * 100
                    sector_changes[name] = round(change, 2)
            except:
                continue

    results = []
    if not bulk_data.empty:
        # Iterate over stocks
        # If MultiIndex, tickers are at level 0
        stock_list = bulk_data.columns.levels[0] if isinstance(bulk_data.columns, pd.MultiIndex) else []
        
        for t in stock_list:
            # Skip indices (they start with ^)
            if t.startswith('^'): continue
            
            try:
                df = bulk_data[t].copy()
                # Check for empty data
                if df['Close'].isnull().all(): continue
                
                res = analyze_ticker(t, df, sector_changes)
                if res: results.append(res)
            except Exception as e:
                continue

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    generate_html(results, sector_changes, now)
    print("Scan Complete.")

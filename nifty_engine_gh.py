import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import warnings
import math
from datetime import datetime

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURATION ---
OUTPUT_DIR = "public"
FILE_PATH = os.path.join(OUTPUT_DIR, "index.html")
# Fallback list of major Nifty 500 stocks if CSV fails
DEFAULT_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS", 
    "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LTIM.NS", "LT.NS", 
    "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS", "ULTRACEMCO.NS", 
    "SUNPHARMA.NS", "BAJFINANCE.NS", "HCLTECH.NS", "ADANIENT.NS", "TATASTEEL.NS",
    "JIOFIN.NS", "ZOMATO.NS", "DLF.NS", "HAL.NS", "VBL.NS", "TRENT.NS"
]

SECTOR_INDICES = {
    "NIFTY BANK": "^NSEBANK",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY IT": "^CNXIT",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY ENERGY": "^CNXENERGY",
    "NIFTY REALTY": "^CNXREALTY"
}

# --- 1. DATA ACQUISITION ---
def get_tickers():
    """Fetches Nifty 500 tickers, defaults to list if fails."""
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        # User-Agent needed for NSE website
        headers = {"User-Agent": "Mozilla/5.0"}
        df = pd.read_csv(url, storage_options=headers)
        tickers = [f"{x}.NS" for x in df['Symbol'].dropna().unique()]
        return tickers
    except Exception as e:
        print(f"Warning: Could not fetch Nifty 500 list ({e}). Using default list.")
        return DEFAULT_TICKERS

def fetch_bulk_data(tickers):
    """
    Downloads data for ALL tickers in ONE request.
    This is 10x faster than looping.
    """
    print(f"Downloading data for {len(tickers)} stocks...")
    # We need about 1 year of data to calculate 200 SMA correctly
    try:
        data = yf.download(tickers, period="1y", group_by='ticker', threads=True, progress=True)
        return data
    except Exception as e:
        print(f"Bulk download failed: {e}")
        return pd.DataFrame()

def fetch_sector_performance():
    """Fetches live performance of Sector Indices."""
    print("Fetching Sector Indices...")
    results = []
    try:
        tickers = list(SECTOR_INDICES.values())
        data = yf.download(tickers, period="5d", threads=True, progress=False)
        
        # yf.download with multiple tickers returns MultiIndex columns
        # Structure: (Price, Ticker)
        # We need to extract Close prices
        if isinstance(data.columns, pd.MultiIndex):
            close_data = data['Close']
        else:
            close_data = data # Fallback if single ticker (unlikely here)

        for name, ticker in SECTOR_INDICES.items():
            if ticker in close_data.columns:
                series = close_data[ticker].dropna()
                if len(series) >= 2:
                    curr = series.iloc[-1]
                    prev = series.iloc[-2]
                    change = ((curr - prev) / prev) * 100
                    results.append({"name": name, "avg": round(change, 2)})
    except Exception as e:
        print(f"Sector fetch error: {e}")
    
    return sorted(results, key=lambda x: x['avg'], reverse=True)

# --- 2. TECHNICAL ANALYSIS ENGINE ---
def calculate_indicators(df):
    """Calculates all technical indicators on the dataframe."""
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ATR (Average True Range) for Stops/Targets
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # Volume Average
    df['VolAvg'] = df['Volume'].rolling(20).mean()
    
    return df

def analyze_ticker(ticker, df):
    """Analyzes a single stock for specific setups."""
    # Clean data
    df = df.dropna()
    if len(df) < 200: return None
    
    # Run Calcs
    df = calculate_indicators(df)
    
    # Current Candle
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    close = curr['Close']
    prev_close = prev['Close']
    change_pct = round(((close - prev_close) / prev_close) * 100, 2)
    
    # --- DETECT SETUPS ---
    setups = []
    
    # 1. Trend Filter (The Baseline)
    trend = "UP" if close > curr['SMA200'] else "DOWN"
    
    # 2. Setup: Momentum Burst
    # Logic: Price > EMA20, RSI > 60 (Strong), Volume > 1.5x Avg
    if close > curr['EMA20'] and curr['RSI'] > 60 and curr['Volume'] > (curr['VolAvg'] * 1.5):
        setups.append("Momentum Burst")

    # 3. Setup: Pullback to Value
    # Logic: Trend UP, Price drops near EMA20/SMA50, RSI cools off (<55)
    dist_to_50 = abs(close - curr['SMA50']) / close
    if trend == "UP" and close > curr['SMA50'] and dist_to_50 < 0.02 and curr['RSI'] < 55:
        setups.append("Pullback Buy")

    # 4. Setup: Golden Cross (Recent)
    # Logic: SMA50 crosses above SMA200 in last 3 days
    for i in range(1, 4):
        if df['SMA50'].iloc[-i] > df['SMA200'].iloc[-i] and df['SMA50'].iloc[-(i+1)] < df['SMA200'].iloc[-(i+1)]:
            setups.append("Golden Cross")
            break

    # 5. Setup: NR7 Breakout
    # Logic: Yesterday's range was narrowest in 7 days, Today price broke High
    ranges = df['High'] - df['Low']
    if ranges.iloc[-2] == ranges.iloc[-8:-1].min():
        if close > prev['High']:
            setups.append("NR7 Breakout")
            
    # 6. Trap: Gap Warning
    # Logic: Opened > 2% higher than prev close (Chasing gaps is dangerous)
    is_gap_up = (curr['Open'] - prev_close) / prev_close > 0.02
    if is_gap_up:
        setups.append("⚠️ GAP UP")

    # --- RISK / REWARD CALCULATION ---
    # Target: Resistance proxy (Close + 2*ATR)
    # Stop: Support proxy (Close - 1*ATR)
    atr = curr['ATR']
    target = close + (2 * atr)
    stop = close - (1 * atr)
    
    potential_reward = target - close
    potential_risk = close - stop
    
    rr_ratio = 0
    if potential_risk > 0:
        rr_ratio = round(potential_reward / potential_risk, 1)

    # --- FINAL VERDICT ---
    verdict = "WAIT"
    v_color = "gray"
    
    # Only give a Buy verdict if there is a Setup AND Trend is UP AND RR is decent
    if trend == "UP" and len(setups) > 0 and "⚠️ GAP UP" not in setups:
        if rr_ratio >= 1.5:
            verdict = "BUY"
            v_color = "green"
        else:
            verdict = "BAD R:R"
            v_color = "orange"
    elif trend == "DOWN" and len(setups) > 0:
        verdict = "CTR-TREND" # Counter Trend trade
        v_color = "blue"

    # Only return interesting stocks
    if verdict == "WAIT" and abs(change_pct) < 2.0:
        return None

    return {
        "symbol": ticker.replace(".NS", ""),
        "price": round(close, 2),
        "change": change_pct,
        "trend": trend,
        "rsi": round(curr['RSI'], 1),
        "vol_mult": round(curr['Volume'] / curr['VolAvg'], 1) if curr['VolAvg'] > 0 else 0,
        "setups": setups,
        "verdict": verdict,
        "v_color": v_color,
        "rr": rr_ratio,
        "levels": {
            "TGT": round(target, 1),
            "SL": round(stop, 1)
        },
        "history": df['Close'].tail(30).tolist()
    }

# --- 3. HTML GENERATION ---
def generate_html(stocks, sectors, nifty_data):
    
    # Calculate Market Breadth
    adv = len([x for x in stocks if x['change'] > 0])
    dec = len([x for x in stocks if x['change'] < 0])
    
    # Prepare JSON for JS
    json_data = json.dumps({
        "stocks": stocks,
        "sectors": sectors,
        "market": {"adv": adv, "dec": dec, "nifty_change": nifty_data}
    })

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <!-- Auto Refresh every 5 minutes -->
        <meta http-equiv="refresh" content="300"> 
        <title>ProTrade Scanner</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            body {{ background-color: #0f172a; color: #cbd5e1; font-family: 'Inter', sans-serif; }}
            .card {{ transition: transform 0.2s; }}
            .card:hover {{ transform: translateY(-2px); border-color: #475569; }}
            
            /* Setup Badges */
            .badge-mom {{ background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }}
            .badge-pull {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
            .badge-cross {{ background: rgba(250, 204, 21, 0.15); color: #facc15; border: 1px solid rgba(250, 204, 21, 0.3); }}
            .badge-nr7 {{ background: rgba(236, 72, 153, 0.15); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.3); }}
            .badge-gap {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}
        </style>
    </head>
    <body class="p-4 md:p-8">
        <div class="max-w-7xl mx-auto">
            
            <!-- Header -->
            <div class="flex flex-col md:flex-row justify-between items-start md:items-end mb-6 border-b border-slate-800 pb-4">
                <div>
                    <h1 class="text-3xl font-bold text-white flex items-center gap-2">
                        <i data-lucide="radar" class="text-blue-500"></i> ProTrade Scanner
                    </h1>
                    <div class="text-sm text-slate-500 mt-1">
                        EOD/Swing Analysis • Auto-Refreshes every 5m
                    </div>
                </div>
                <div class="mt-4 md:mt-0 text-right">
                     <div class="text-xs font-mono text-slate-400">MARKET BREADTH</div>
                     <div class="flex items-center gap-3 bg-slate-800 px-3 py-1 rounded border border-slate-700 mt-1">
                        <span class="text-green-400 font-bold">ADV {adv}</span>
                        <span class="text-slate-600">|</span>
                        <span class="text-red-400 font-bold">DEC {dec}</span>
                     </div>
                </div>
            </div>

            <!-- Sector Heatmap -->
            <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-2 mb-8">
                <!-- JS Injected Sectors -->
                <div id="sector-root" class="contents"></div>
            </div>

            <!-- Filters -->
            <div class="flex gap-2 mb-6 overflow-x-auto pb-2">
                <button onclick="filter('ALL')" id="btn-ALL" class="px-4 py-1.5 rounded-full text-xs font-bold bg-blue-600 text-white">All</button>
                <button onclick="filter('BUY')" id="btn-BUY" class="px-4 py-1.5 rounded-full text-xs font-bold bg-slate-800 text-slate-400 hover:text-white border border-slate-700">Valid Buys</button>
                <button onclick="filter('MOM')" id="btn-MOM" class="px-4 py-1.5 rounded-full text-xs font-bold bg-slate-800 text-slate-400 hover:text-white border border-slate-700">Momentum</button>
                <button onclick="filter('PULL')" id="btn-PULL" class="px-4 py-1.5 rounded-full text-xs font-bold bg-slate-800 text-slate-400 hover:text-white border border-slate-700">Pullbacks</button>
            </div>

            <!-- Grid -->
            <div id="grid-root" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                <!-- JS Injected Cards -->
            </div>

        </div>

        <script>
            const DATA = {json_data};
            let activeFilter = 'ALL';

            function init() {{
                renderSectors();
                renderGrid();
                lucide.createIcons();
            }}

            function renderSectors() {{
                const root = document.getElementById('sector-root');
                root.innerHTML = DATA.sectors.map(s => {{
                    const color = s.avg >= 0 ? 'text-green-400' : 'text-red-400';
                    const bg = s.avg >= 0 ? 'bg-green-900/10 border-green-900/30' : 'bg-red-900/10 border-red-900/30';
                    return `
                        <div class="px-3 py-2 rounded border ${{bg}} flex flex-col items-center justify-center">
                            <span class="text-[10px] text-slate-400 uppercase tracking-wide">${{s.name.replace('NIFTY ', '')}}</span>
                            <span class="text-sm font-bold ${{color}}">${{s.avg}}%</span>
                        </div>
                    `;
                }}).join('');
            }}

            function filter(type) {{
                activeFilter = type;
                // Update buttons
                document.querySelectorAll('button').forEach(b => {{
                    if(b.id === 'btn-'+type) b.className = "px-4 py-1.5 rounded-full text-xs font-bold bg-blue-600 text-white shadow-lg shadow-blue-900/20";
                    else b.className = "px-4 py-1.5 rounded-full text-xs font-bold bg-slate-800 text-slate-400 hover:text-white border border-slate-700";
                }});
                renderGrid();
            }}

            function renderGrid() {{
                const root = document.getElementById('grid-root');
                
                const filtered = DATA.stocks.filter(s => {{
                    if(activeFilter === 'ALL') return true;
                    if(activeFilter === 'BUY') return s.verdict === 'BUY';
                    if(activeFilter === 'MOM') return s.setups.includes('Momentum Burst');
                    if(activeFilter === 'PULL') return s.setups.includes('Pullback Buy');
                    return true;
                }});

                if(filtered.length === 0) {{
                    root.innerHTML = `<div class="col-span-full text-center py-20 text-slate-600">No stocks match this filter.</div>`;
                    return;
                }}

                root.innerHTML = filtered.map(s => {{
                    // Sparkline
                    const hist = s.history;
                    const min = Math.min(...hist);
                    const max = Math.max(...hist);
                    const pts = hist.map((p, i) => {{
                        const x = (i / (hist.length-1)) * 100;
                        const y = 40 - ((p - min) / (max - min || 1)) * 40;
                        return `${{x}},${{y}}`;
                    }}).join(' ');
                    const stroke = s.change >= 0 ? '#4ade80' : '#f87171';

                    // Badges
                    const badges = s.setups.map(bg => {{
                        let cls = 'badge-mom';
                        if(bg.includes('Pullback')) cls = 'badge-pull';
                        if(bg.includes('Cross')) cls = 'badge-cross';
                        if(bg.includes('NR7')) cls = 'badge-nr7';
                        if(bg.includes('GAP')) cls = 'badge-gap';
                        return `<span class="px-1.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${{cls}}">${{bg}}</span>`;
                    }}).join('');

                    // Verdict Color
                    let vClass = 'text-slate-400 border-slate-700';
                    if(s.v_color === 'green') vClass = 'text-green-400 border-green-500/50 bg-green-500/10';
                    if(s.v_color === 'orange') vClass = 'text-orange-400 border-orange-500/50 bg-orange-500/10';
                    if(s.v_color === 'blue') vClass = 'text-blue-400 border-blue-500/50 bg-blue-500/10';

                    return `
                        <div class="card bg-slate-800 rounded-xl border border-slate-700 p-4 relative overflow-hidden group">
                            <!-- Header -->
                            <div class="flex justify-between items-start mb-2">
                                <div>
                                    <div class="font-bold text-lg text-white group-hover:text-blue-400 transition cursor-pointer">${{s.symbol}}</div>
                                    <div class="text-xs font-mono mt-0.5 text-slate-400">
                                        ₹${{s.price}} 
                                        <span class="${{s.change >= 0 ? 'text-green-400' : 'text-red-400'}} ml-1">${{s.change}}%</span>
                                    </div>
                                </div>
                                <div class="text-right">
                                    <div class="px-2 py-1 rounded text-[10px] font-bold border ${{vClass}} inline-block mb-1">
                                        ${{s.verdict}}
                                    </div>
                                    <div class="text-[10px] text-slate-500">R:R <span class="text-white">${{s.rr}}</span></div>
                                </div>
                            </div>

                            <!-- Badges -->
                            <div class="flex flex-wrap gap-1 mb-4 h-6">${{badges}}</div>

                            <!-- Levels -->
                            <div class="grid grid-cols-2 gap-2 mb-4 text-[10px] font-mono border-t border-b border-slate-700/50 py-2">
                                <div>
                                    <span class="text-slate-500">STOP</span>
                                    <span class="text-red-400 block text-xs">₹${{s.levels.SL}}</span>
                                </div>
                                <div class="text-right">
                                    <span class="text-slate-500">TARGET</span>
                                    <span class="text-green-400 block text-xs">₹${{s.levels.TGT}}</span>
                                </div>
                            </div>

                            <!-- Mini Chart -->
                            <div class="absolute bottom-0 left-0 right-0 h-10 opacity-30 group-hover:opacity-50 transition">
                                <svg width="100%" height="100%" preserveAspectRatio="none">
                                    <polyline points="${{pts}}" fill="none" stroke="${{stroke}}" stroke-width="2" />
                                </svg>
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
    print(f"Report generated: {FILE_PATH}")

# --- MAIN ENGINE ---
if __name__ == "__main__":
    print("--- STARTING PRO-GRADE SCANNER ---")
    
    # 1. Get Sectors
    sectors = fetch_sector_performance()
    
    # 2. Get Tickers
    tickers = get_tickers()
    
    # 3. Bulk Download (The Speed Boost)
    # This returns a multi-index dataframe: (PriceType, Ticker)
    bulk_data = fetch_bulk_data(tickers)
    
    results = []
    
    if not bulk_data.empty:
        print("Analyzing stocks...")
        
        # Iterate over tickers in the bulk data
        # Note: bulk_data.columns levels are (Price, Ticker) or (Ticker, Price) depending on version
        # We assume new yfinance format: columns are (Price, Ticker)
        
        # Safe extraction of tickers from columns
        # If columns are MultiIndex, the tickers are in one of the levels
        if isinstance(bulk_data.columns, pd.MultiIndex):
            # Extract unique tickers from the MultiIndex
            # Usually level 1 is Ticker if level 0 is 'Close', 'Open', etc.
            # But yfinance download(group_by='ticker') puts Ticker at Level 0
            downloaded_tickers = bulk_data.columns.get_level_values(0).unique()
        else:
             downloaded_tickers = [] # Should not happen with group_by='ticker'

        total = len(downloaded_tickers)
        for i, t in enumerate(downloaded_tickers):
            try:
                # Extract single stock dataframe
                df = bulk_data[t].copy()
                
                # Check if data is valid (not all NaNs)
                if df['Close'].isnull().all(): continue
                
                res = analyze_ticker(t, df)
                if res: results.append(res)
                
            except Exception as e:
                continue

    # 4. Generate Report
    # Mock Nifty Change for context (using First stock in list as proxy or separate fetch)
    nifty_chg = 0.0 # Placeholder
    
    generate_html(results, sectors, nifty_chg)
    print("Done.")

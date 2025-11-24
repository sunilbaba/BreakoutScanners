import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import warnings
import pytz
from datetime import datetime

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURATION ---
OUTPUT_DIR = "public"
FILE_PATH = os.path.join(OUTPUT_DIR, "index.html")

SECTOR_INDICES = {
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
    # Robust fallback list
    default_list = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS", 
        "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LTIM.NS", "LT.NS", 
        "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS", "ULTRACEMCO.NS", 
        "SUNPHARMA.NS", "BAJFINANCE.NS", "HCLTECH.NS", "ADANIENT.NS", "TATASTEEL.NS",
        "JIOFIN.NS", "ZOMATO.NS", "DLF.NS", "HAL.NS", "VBL.NS", "TRENT.NS", "BEL.NS",
        "POWERGRID.NS", "ONGC.NS", "NTPC.NS", "COALINDIA.NS", "BPCL.NS"
    ]
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        df = pd.read_csv(url, storage_options=headers)
        tickers = [f"{x}.NS" for x in df['Symbol'].dropna().unique()]
        print(f"Fetched {len(tickers)} tickers from NSE.")
        return tickers
    except:
        return default_list

def fetch_bulk_data(tickers):
    all_tickers = tickers + list(SECTOR_INDICES.values())
    print(f"Downloading {len(all_tickers)} symbols...")
    try:
        data = yf.download(all_tickers, period="6mo", group_by='ticker', threads=True, progress=True)
        return data
    except Exception as e:
        print(f"Download failed: {e}")
        return pd.DataFrame()

# --- 2. ANALYSIS ENGINE ---
def calculate_indicators(df):
    df = df.copy()
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
    df = df.dropna(subset=['Close'])
    if len(df) < 200: return None
    
    df = calculate_indicators(df)
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    close = float(curr['Close'])
    prev_close = float(prev['Close'])
    if prev_close == 0: return None
    
    change_pct = round(((close - prev_close) / prev_close) * 100, 2)
    clean_sym = ticker.replace(".NS", "")
    
    # Logic
    setups = []
    trend = "UP" if close > float(curr['SMA200']) else "DOWN"
    
    # Momentum
    vol_avg = float(curr['VolAvg']) if not np.isnan(curr['VolAvg']) else 1.0
    if close > float(curr['EMA20']) and curr['RSI'] > 60 and curr['Volume'] > (vol_avg * 1.5):
        setups.append("Momentum Burst")

    # Pullback
    sma50 = float(curr['SMA50'])
    if trend == "UP" and close > sma50 and abs(close - sma50)/close < 0.02 and curr['RSI'] < 55:
        setups.append("Pullback Buy")

    # Golden Cross
    if df['SMA50'].iloc[-1] > df['SMA200'].iloc[-1] and df['SMA50'].iloc[-2] < df['SMA200'].iloc[-2]:
        setups.append("Golden Cross")
            
    # Gap
    if (curr['Open'] - prev_close) / prev_close > 0.02:
        setups.append("⚠️ GAP UP")

    # Sector
    my_sector = get_stock_sector(clean_sym)
    sector_strength = sector_changes.get(my_sector, 0)
    has_sector_support = sector_strength > 0.5
    if has_sector_support: setups.append(f"Sector Support")

    # R:R
    atr = float(curr['ATR'])
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

    # Minimal Filter for Dashboard
    if verdict == "WAIT" and abs(change_pct) < 1.0: return None

    return {
        "symbol": clean_sym,
        "price": round(close, 2),
        "change": change_pct,
        "sector": my_sector,
        "rsi": round(curr['RSI'], 1),
        "vol_mult": round(curr['Volume']/vol_avg, 1),
        "setups": setups,
        "verdict": verdict,
        "v_color": v_color,
        "rr": rr_ratio,
        "levels": {"TGT": round(target, 1), "SL": round(stop, 1)},
        "history": df['Close'].tail(30).tolist()
    }

# --- 3. HTML GENERATION ---
def generate_html(stocks, sector_data, updated_time, is_market_open):
    adv = len([x for x in stocks if x['change'] > 0])
    dec = len([x for x in stocks if x['change'] < 0])
    stocks.sort(key=lambda x: (x['verdict'] != "PRIME BUY ⭐", "BUY" not in x['verdict'], -x['change']))

    json_data = json.dumps({
        "stocks": stocks,
        "sectors": [{"name": k, "avg": v} for k,v in sector_data.items()],
        "market": {"adv": adv, "dec": dec, "status": "OPEN" if is_market_open else "CLOSED"}
    })

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <!-- Only refresh if market is OPEN -->
        { '<meta http-equiv="refresh" content="300">' if is_market_open else '' }
        <title>PrimeTrade Analyst</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            body {{ background: #0f172a; color: #e2e8f0; font-family: sans-serif; }}
            .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px; }}
            .card:hover {{ border-color: #64748b; }}
            .prime {{ border: 1px solid #a855f7; background: #1e293b; }}
            .badge {{ font-size: 9px; padding: 2px 4px; border-radius: 3px; font-weight: bold; margin-right: 3px; display: inline-block; }}
            .badge-mom {{ background: #064e3b; color: #6ee7b7; }}
            .badge-sec {{ background: #1e3a8a; color: #93c5fd; }}
            
            /* Table Styles */
            table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
            th {{ text-align: left; padding: 8px; background: #0f172a; color: #94a3b8; border-bottom: 1px solid #334155; }}
            td {{ padding: 8px; border-bottom: 1px solid #1e293b; }}
            tr:hover {{ background: #1e293b; }}
        </style>
    </head>
    <body class="p-4 md:p-8">
        <div class="max-w-7xl mx-auto">
            
            <!-- Header -->
            <div class="flex flex-col md:flex-row justify-between items-center mb-6">
                <div>
                    <h1 class="text-2xl font-bold text-white flex items-center gap-2">
                        <i data-lucide="bar-chart-2" class="text-purple-500"></i> PrimeTrade 
                        <span class="text-xs px-2 py-0.5 rounded { 'bg-green-900 text-green-300' if is_market_open else 'bg-red-900 text-red-300' }">
                            { '🟢 LIVE' if is_market_open else '🔴 CLOSED (EOD)' }
                        </span>
                    </h1>
                    <div class="text-xs text-slate-500 mt-1">{updated_time}</div>
                </div>
                <div class="flex gap-4 mt-4 md:mt-0">
                    <div class="text-center"><div class="text-[10px] text-slate-500">ADV</div><div class="font-bold text-green-500">{adv}</div></div>
                    <div class="text-center"><div class="text-[10px] text-slate-500">DEC</div><div class="font-bold text-red-500">{dec}</div></div>
                </div>
            </div>

            <!-- Sectors -->
            <div id="sector-row" class="grid grid-cols-5 md:grid-cols-9 gap-2 mb-6 text-center"></div>

            <!-- Controls -->
            <div class="flex justify-between items-center mb-4 bg-slate-800 p-2 rounded-lg">
                <div class="flex gap-2">
                    <button onclick="setView('card')" id="btn-card" class="px-3 py-1 text-xs font-bold rounded bg-blue-600 text-white"><i data-lucide="grid" class="w-3 h-3 inline mr-1"></i>Cards</button>
                    <button onclick="setView('table')" id="btn-table" class="px-3 py-1 text-xs font-bold rounded text-slate-400 hover:text-white"><i data-lucide="list" class="w-3 h-3 inline mr-1"></i>Table</button>
                </div>
                <button onclick="downloadCSV()" class="px-3 py-1 text-xs font-bold rounded bg-green-700 text-white hover:bg-green-600"><i data-lucide="download" class="w-3 h-3 inline mr-1"></i>Excel</button>
            </div>

            <!-- Content Area -->
            <div id="content-area"></div>
        </div>

        <script>
            const DATA = {json_data};
            let currentView = 'card';

            function init() {{
                const sRoot = document.getElementById('sector-row');
                sRoot.innerHTML = DATA.sectors.map(s => {{
                    const color = s.avg >= 0 ? 'text-green-400' : 'text-red-400';
                    return `<div class="p-1 rounded bg-slate-800/50 border border-slate-700/50">
                        <div class="text-[8px] text-slate-400">${{s.name}}</div>
                        <div class="text-xs font-bold ${{color}}">${{s.avg}}%</div>
                    </div>`;
                }}).join('');
                render();
                lucide.createIcons();
            }}

            function setView(v) {{
                currentView = v;
                document.getElementById('btn-card').className = v==='card' ? "px-3 py-1 text-xs font-bold rounded bg-blue-600 text-white" : "px-3 py-1 text-xs font-bold rounded text-slate-400";
                document.getElementById('btn-table').className = v==='table' ? "px-3 py-1 text-xs font-bold rounded bg-blue-600 text-white" : "px-3 py-1 text-xs font-bold rounded text-slate-400";
                render();
            }}

            function render() {{
                const root = document.getElementById('content-area');
                if(currentView === 'card') {{
                    root.className = "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4";
                    root.innerHTML = DATA.stocks.map(s => {{
                        const isPrime = s.verdict.includes('PRIME');
                        const badges = s.setups.map(b => {{
                            let cls = 'bg-slate-700 text-slate-300';
                            if(b.includes('Momentum')) cls = 'badge-mom';
                            if(b.includes('Sector')) cls = 'badge-sec';
                            return `<span class="badge ${{cls}}">${{b}}</span>`;
                        }}).join('');
                        let vColor = s.v_color === 'purple' ? 'text-purple-400' : (s.v_color === 'green' ? 'text-green-400' : 'text-slate-500');
                        
                        // Sparkline
                        const h = s.history; 
                        const pts = h.map((p,i) => `${{(i/(h.length-1))*100}},${{30-((p-Math.min(...h))/(Math.max(...h)-Math.min(...h)||1))*30}}`).join(' ');

                        return `<div class="card ${{isPrime ? 'prime' : ''}}">
                            <div class="flex justify-between mb-2">
                                <div><div class="font-bold text-white">${{s.symbol}}</div><div class="text-[10px] text-slate-400">${{s.sector}}</div></div>
                                <div class="text-right"><div class="font-mono font-bold ${{s.change>=0?'text-green-400':'text-red-400'}}">${{s.change}}%</div><div class="text-[10px] text-slate-500">₹${{s.price}}</div></div>
                            </div>
                            <div class="mb-3 h-5">${{badges}}</div>
                            <div class="flex justify-between items-end border-t border-slate-700 pt-2">
                                <div class="font-bold text-xs ${{vColor}}">${{s.verdict.replace('⭐','')}}</div>
                                <div class="text-[10px] text-slate-400">RR: <span class="text-white">${{s.rr}}</span></div>
                            </div>
                            <svg class="mt-2 opacity-30" height="30" width="100%"><polyline points="${{pts}}" fill="none" stroke="${{s.change>=0?'#4ade80':'#f87171'}}" stroke-width="2"/></svg>
                        </div>`;
                    }}).join('');
                }} else {{
                    root.className = "overflow-x-auto";
                    let rows = DATA.stocks.map(s => `<tr>
                        <td class="font-bold text-white">${{s.symbol}} <span class="text-[10px] text-slate-500 block">${{s.sector}}</span></td>
                        <td class="${{s.change>=0?'text-green-400':'text-red-400'}} font-mono">${{s.change}}%</td>
                        <td>${{s.price}}</td>
                        <td>${{s.rsi}}</td>
                        <td>${{s.vol_mult}}x</td>
                        <td class="text-xs text-slate-300">${{s.setups.join(', ')}}</td>
                        <td class="font-bold ${{s.v_color==='purple'?'text-purple-400':(s.v_color==='green'?'text-green-400':'text-slate-500')}}">${{s.verdict}}</td>
                        <td>${{s.rr}}</td>
                        <td class="font-mono text-green-400">${{s.levels.TGT}}</td>
                        <td class="font-mono text-red-400">${{s.levels.SL}}</td>
                    </tr>`).join('');
                    root.innerHTML = `<table><thead><tr><th>Stock</th><th>%</th><th>Price</th><th>RSI</th><th>Vol</th><th>Setup</th><th>Verdict</th><th>RR</th><th>Target</th><th>Stop</th></tr></thead><tbody>${{rows}}</tbody></table>`;
                }}
                lucide.createIcons();
            }}

            function downloadCSV() {{
                const headers = ["Symbol", "Sector", "Price", "Change%", "RSI", "VolMult", "Verdict", "RR", "Target", "Stop", "Setups"];
                const rows = DATA.stocks.map(s => [
                    s.symbol, s.sector, s.price, s.change, s.rsi, s.vol_mult, 
                    s.verdict.replace('⭐',''), s.rr, s.levels.TGT, s.levels.SL, s.setups.join('|')
                ]);
                let csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(e => e.join(","))].join("\\n");
                const link = document.createElement("a");
                link.setAttribute("href", encodeURI(csvContent));
                link.setAttribute("download", "prime_trade_data.csv");
                document.body.appendChild(link);
                link.click();
            }}
            
            init();
        </script>
    </body>
    </html>
    """
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    with open(FILE_PATH, "w", encoding="utf-8") as f: f.write(html)

if __name__ == "__main__":
    # Time Logic: Check if market is open (09:15 - 15:30 IST)
    # IST = UTC + 5:30
    utc_now = datetime.utcnow()
    ist_hour = (utc_now.hour + 5) + (utc_now.minute + 30) // 60
    ist_min = (utc_now.minute + 30) % 60
    
    # Simple market check: Weekday AND (Time > 09:15 AND Time < 15:30)
    # Note: GitHub Actions runs on UTC, so we rely on the scheduler mostly.
    # But we calculate status for the badge.
    # 03:45 UTC = 09:15 IST | 10:00 UTC = 15:30 IST
    utc_time = utc_now.hour * 60 + utc_now.minute
    market_open = (utc_now.weekday() < 5) and (225 <= utc_time <= 600) # 03:45 to 10:00 UTC

    tickers = get_tickers()
    bulk_data = fetch_bulk_data(tickers)
    
    sector_changes = {}
    results = []
    
    if not bulk_data.empty:
        # Extract Sector Data
        for name, ticker in SECTOR_INDICES.items():
            if ticker in bulk_data.columns:
                try:
                    s = bulk_data[ticker]['Close'].dropna()
                    if len(s)>1: sector_changes[name] = round(((s.iloc[-1]-s.iloc[-2])/s.iloc[-2])*100, 2)
                except: pass
                
        # Extract Stocks
        stock_list = bulk_data.columns.levels[0] if isinstance(bulk_data.columns, pd.MultiIndex) else bulk_data.columns
        for t in stock_list:
            if t.startswith('^'): continue
            try:
                res = analyze_ticker(t, bulk_data[t].copy(), sector_changes)
                if res: results.append(res)
            except: continue

    timestamp = utc_now.strftime("%Y-%m-%d %H:%M UTC")
    generate_html(results, sector_changes, timestamp, market_open)

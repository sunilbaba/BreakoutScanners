import yfinance as yf
import pandas as pd
import os
import time
import requests
import io
import json
import warnings
import math
from datetime import datetime

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURATION ---
OUTPUT_DIR = "public"
FILE_PATH = os.path.join(OUTPUT_DIR, "index.html")
CSV_FILENAME = "ind_nifty500list.csv"
NSE_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
FNO_URL = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"

# --- 1. SESSION HELPER ---
def get_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml"
    })
    return s

# --- 2. DATA FETCHING (With Sector Mapping) ---
def get_nifty500_data(session):
    print("Fetching Nifty 500 List...")
    sector_map = {}
    tickers = []
    
    try:
        # Read CSV (Local or Web)
        if os.path.exists(CSV_FILENAME):
            df = pd.read_csv(CSV_FILENAME)
        else:
            r = session.get(NSE_URL, timeout=10)
            df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
        
        # Map Symbol -> Industry
        # NSE CSV usually has 'Industry' column
        if 'Industry' in df.columns:
            for index, row in df.iterrows():
                sym = f"{row['Symbol']}.NS"
                sector_map[sym] = row['Industry']
                tickers.append(sym)
        else:
            tickers = [f"{x}.NS" for x in df['Symbol'].dropna().tolist()]
            
        return tickers, sector_map
    except:
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"], {}

def get_fno_list(session):
    try:
        r = session.get(FNO_URL, timeout=10)
        df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
        df.columns = [c.strip() for c in df.columns]
        if 'SYMBOL' in df.columns:
            return [f"{x.strip()}.NS" for x in df['SYMBOL'].dropna().unique().tolist()]
    except: pass
    return []

# --- 3. GET MARKET CONTEXT (Nifty 50 Performance) ---
def get_nifty_change():
    try:
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period="5d")
        curr = hist['Close'].iloc[-1]
        prev = hist['Close'].iloc[-2]
        return ((curr - prev) / prev) * 100
    except:
        return 0.0

# --- 4. ANALYSIS ENGINE ---
def analyze_stock(ticker, is_fno, sector, nifty_change):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="5y", interval="1d")
        if df.empty or len(df) < 300: return None

        # Basic Data
        close = df['Close']
        high = df['High']
        low = df['Low']
        vol = df['Volume']
        opn = df['Open']
        
        curr = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        change = round(((curr - prev) / prev) * 100, 2)
        
        # --- RELATIVE STRENGTH (RS) ---
        # RS = Stock % Change - Nifty % Change
        # If Positive, Stock is outperforming Market
        rs_score = round(change - nifty_change, 2)
        
        # --- TRENDS (M/W/D) ---
        agg = {'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}
        df_w = df.resample('W').agg(agg).dropna()
        df_m = df_d = df # Keep Monthly logic simpler or calculate manually if needed, using df for speed here
        
        # Simple Trend Helper
        def trend(series): return "UP" if series.iloc[-1] > series.ewm(span=20, adjust=False).mean().iloc[-1] else "DOWN"
        
        d_trend = trend(df['Close'])
        # Approximate Weekly Trend using 100 Day MA (~20 Week)
        w_trend = "UP" if curr > close.rolling(100).mean().iloc[-1] else "DOWN"
        # Approximate Monthly Trend using 200 Day MA
        m_trend = "UP" if curr > close.rolling(200).mean().iloc[-1] else "DOWN"

        # --- INDICATORS ---
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        curr_rsi = float(rsi.iloc[-1])
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        sig = macd.ewm(span=9, adjust=False).mean()
        
        # Volume
        vol_avg = float(vol.rolling(20).mean().iloc[-1])
        curr_vol = float(vol.iloc[-1])
        vol_mult = round(curr_vol / vol_avg, 1) if vol_avg > 0 else 0
        
        # ATR Target
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        target = round(curr + (2 * atr), 1)
        stop = round(curr - (1 * atr), 1)

        # --- SIGNALS ---
        signals = []
        
        # Moving Avg Crosses
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        if float(sma50.iloc[-2]) < float(sma200.iloc[-2]) and float(sma50.iloc[-1]) > float(sma200.iloc[-1]): signals.append("Golden Cross")
        
        # Momentum
        if float(macd.iloc[-2]) < float(sig.iloc[-2]) and float(macd.iloc[-1]) > float(sig.iloc[-1]): signals.append("MACD Buy")
        
        # Price/Vol
        if prev < float(high.iloc[-253:-1].max()) and curr > float(high.iloc[-253:-1].max()): signals.append("52W High")
        if vol_mult >= 2.5: signals.append(f"Vol {vol_mult}x")
        
        # RSI
        if curr_rsi < 20: signals.append("RSI < 20") # Panic
        elif curr_rsi < 30: signals.append("Oversold")
        
        # Engulfing
        prev_open = float(opn.iloc[-2])
        curr_open = float(opn.iloc[-1])
        if (prev < prev_open) and (curr > curr_open) and (curr_open < prev) and (curr > prev_open):
            signals.append("Bull Engulfing")

        # Filter
        confluence = 0
        if m_trend == "UP": confluence += 1
        if w_trend == "UP": confluence += 1
        if d_trend == "UP": confluence += 1
        
        interesting = len(signals) > 0 or (is_fno and confluence == 3) or (rs_score > 2.0)
        if not interesting: return None
        
        return {
            "symbol": ticker.replace(".NS", ""),
            "price": round(curr, 2),
            "change": change,
            "sector": sector if sector else "Other",
            "rs": rs_score,
            "trends": {"M": m_trend, "W": w_trend, "D": d_trend},
            "signals": signals,
            "vol_mult": vol_mult,
            "is_fno": is_fno,
            "target": target,
            "stop": stop,
            "rsi": round(curr_rsi, 1),
            "history": [x if not math.isnan(x) else 0 for x in close.tail(30).tolist()]
        }

    except Exception:
        return None

# --- 5. GENERATE HTML ---
def generate_html(results, nifty_chg):
    # 1. Calculate Sector Heatmap
    sector_perf = {}
    for r in results:
        sec = r['sector']
        if sec not in sector_perf: sector_perf[sec] = []
        sector_perf[sec].append(r['change'])
    
    # Avg change per sector
    sector_summary = []
    for sec, changes in sector_perf.items():
        avg = sum(changes) / len(changes)
        sector_summary.append({"name": sec, "avg": round(avg, 2), "count": len(changes)})
    
    # Sort sectors by performance
    sector_summary.sort(key=lambda x: x['avg'], reverse=True)
    top_sectors = sector_summary[:5] # Top 5

    json_data = json.dumps({"stocks": results, "sectors": top_sectors, "nifty_chg": round(nifty_chg, 2)})
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Institutional Scanner</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            body {{ background-color: #0f172a; color: #e2e8f0; font-family: sans-serif; }}
            .card {{ transition: all 0.2s; }}
            .card:hover {{ transform: translateY(-3px); border-color: #60a5fa; }}
            .modal {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.9); z-index: 50; backdrop-filter: blur(5px); }}
            .modal-content {{ background: #1e293b; margin: 5vh auto; width: 95%; max-width: 600px; max-height: 90vh; overflow-y: auto; border-radius: 12px; border: 1px solid #334155; }}
            
            /* Dots */
            .dot {{ height: 8px; width: 8px; border-radius: 50%; display: inline-block; margin-right: 2px; }}
            .dot-up {{ background-color: #4ade80; box-shadow: 0 0 5px #4ade80; }}
            .dot-down {{ background-color: #f87171; opacity: 0.4; }}
            
            /* Badges */
            .b-std {{ background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }}
            .b-gold {{ background: rgba(250, 204, 21, 0.15); color: #facc15; border: 1px solid rgba(250, 204, 21, 0.3); }}
            .b-panic {{ background: #9333ea; color: white; border: 1px solid #d8b4fe; box-shadow: 0 0 8px #a855f7; }}
            .b-fno {{ background: rgba(14, 165, 233, 0.2); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.4); }}
        </style>
    </head>
    <body>
        <div class="max-w-7xl mx-auto p-4">
            <!-- HEADER -->
            <header class="mb-6 border-b border-slate-700 pb-4">
                <div class="flex justify-between items-end mb-4">
                    <div>
                        <h1 class="text-2xl font-bold text-blue-400 flex items-center gap-2">
                            <i data-lucide="briefcase"></i> Institutional Scanner
                        </h1>
                        <p class="text-xs text-slate-500 mt-1">Sector Rotation + Relative Strength (RS) + Breakouts</p>
                    </div>
                    <div class="text-right">
                        <div class="text-[10px] text-slate-500 uppercase">Nifty 50</div>
                        <div class="text-lg font-bold ${{nifty_chg >= 0 ? 'text-green-400' : 'text-red-400'}}">{{RAW.nifty_chg}}%</div>
                    </div>
                </div>

                <!-- SECTOR BAR -->
                <div class="flex gap-2 overflow-x-auto pb-2 scrollbar-hide">
                    {{RAW.sectors.map(s => (
                        <div class="bg-slate-800 px-3 py-2 rounded border border-slate-700 min-w-[120px]">
                            <div class="text-[10px] text-slate-400 truncate">{{s.name}}</div>
                            <div class="flex justify-between items-end">
                                <div class="font-bold text-sm ${{s.avg >= 0 ? 'text-green-400' : 'text-red-400'}}">{{s.avg}}%</div>
                                <div class="text-[10px] text-slate-600">{{s.count}} stocks</div>
                            </div>
                        </div>
                    ))}}
                </div>
                
                <!-- FILTERS -->
                <div class="flex bg-slate-800 p-1 rounded-lg mt-4 overflow-x-auto gap-1">
                    <button onclick="setFilter('ALL')" id="btn-all" class="px-4 py-1.5 rounded text-xs font-bold bg-blue-600 text-white">All</button>
                    <button onclick="setFilter('FNO')" id="btn-fno" class="px-4 py-1.5 rounded text-xs font-bold text-slate-400">F&O</button>
                    <button onclick="setFilter('RS')" id="btn-rs" class="px-4 py-1.5 rounded text-xs font-bold text-slate-400">Strong RS</button>
                    <button onclick="setFilter('VOL')" id="btn-vol" class="px-4 py-1.5 rounded text-xs font-bold text-slate-400">Vol Shock</button>
                </div>
            </header>
            
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4" id="grid"></div>
        </div>

        <!-- MODAL -->
        <div id="modal" class="modal">
            <div class="modal-content">
                <div class="sticky top-0 bg-slate-800 p-4 border-b border-slate-700 flex justify-between items-center">
                    <div>
                        <h2 id="m-sym" class="text-2xl font-bold text-white">SYMBOL</h2>
                        <div id="m-sec" class="text-xs text-blue-400">SECTOR</div>
                    </div>
                    <button onclick="closeModal()" class="p-2 bg-slate-700 rounded-full hover:bg-slate-600"><i data-lucide="x"></i></button>
                </div>
                <div class="p-6 space-y-6">
                    <div class="grid grid-cols-2 gap-4">
                        <div class="bg-green-900/20 border border-green-500/30 p-3 rounded-lg text-center">
                            <div class="text-xs text-green-400 uppercase font-bold">Target</div>
                            <div id="m-target" class="text-2xl font-bold text-white">0</div>
                        </div>
                        <div class="bg-red-900/20 border border-red-500/30 p-3 rounded-lg text-center">
                            <div class="text-xs text-red-400 uppercase font-bold">Stop Loss</div>
                            <div id="m-stop" class="text-2xl font-bold text-white">0</div>
                        </div>
                    </div>
                    <div class="bg-slate-800 p-3 rounded-lg border border-slate-700">
                        <div class="flex justify-between text-sm mb-1">
                            <span class="text-slate-400">Relative Strength (vs Nifty)</span>
                            <span id="m-rs" class="font-bold text-white">0</span>
                        </div>
                        <p class="text-[10px] text-slate-500">Positive means outperforming market.</p>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const RAW = {json_data};
            const data = RAW.stocks;
            let currentFilter = 'ALL';
            lucide.createIcons();
            
            // Render Sectors (Handled in Template Literal above)

            function setFilter(type) {{
                currentFilter = type;
                ['btn-all','btn-fno','btn-rs','btn-vol'].forEach(id => document.getElementById(id).className = "px-4 py-1.5 rounded text-xs font-bold text-slate-400 hover:text-white transition");
                document.getElementById('btn-'+type.toLowerCase()).className = "px-4 py-1.5 rounded text-xs font-bold bg-blue-600 text-white transition";
                render();
            }}

            function render() {{
                const grid = document.getElementById('grid');
                grid.innerHTML = '';
                
                const filtered = data.filter(s => {{
                    if(currentFilter === 'FNO') return s.is_fno;
                    if(currentFilter === 'RS') return s.rs > 1.5; // Outperforming by 1.5%
                    if(currentFilter === 'VOL') return s.vol_mult >= 2.0;
                    return true;
                }});

                if(filtered.length === 0) grid.innerHTML = '<div class="col-span-3 text-center text-slate-500 p-10">No stocks match this filter.</div>';

                filtered.forEach((s, i) => {{
                    // RS Color
                    const rsColor = s.rs > 0 ? 'text-green-400' : 'text-red-400';
                    
                    // Trends
                    const mDot = s.trends.M === 'UP' ? 'dot-up' : 'dot-down';
                    const wDot = s.trends.W === 'UP' ? 'dot-up' : 'dot-down';
                    const dDot = s.trends.D === 'UP' ? 'dot-up' : 'dot-down';
                    
                    // Badges
                    let badges = '';
                    if(s.is_fno) badges += `<span class="b-fno px-2 py-1 rounded text-[10px] font-bold mr-1 mb-1 inline-block">F&O</span>`;
                    
                    s.signals.forEach(sig => {{
                        let cls = 'b-std';
                        if(sig.includes('Golden')) cls = 'b-gold';
                        if(sig.includes('20')) cls = 'b-panic';
                        badges += `<span class="${{cls}} px-2 py-1 rounded text-[10px] font-bold uppercase mr-1 mb-1 inline-block">${{sig}}</span>`;
                    }});

                    const pts = s.history.map((d, j) => {{
                        const min = Math.min(...s.history); const max = Math.max(...s.history);
                        const x = (j / (s.history.length - 1)) * 100;
                        const y = 30 - ((d - min) / (max - min || 1)) * 30;
                        return `${{x}},${{y}}`;
                    }}).join(' ');

                    grid.innerHTML += `
                        <div class="card bg-slate-800 p-4 rounded-xl border border-slate-700 shadow-lg relative group" onclick="openModal('${{s.symbol}}')">
                            <div class="flex justify-between items-start mb-2">
                                <div>
                                    <div class="flex items-center gap-2">
                                        <div class="font-bold text-lg text-white group-hover:text-blue-400 transition">${{s.symbol}}</div>
                                        <span class="text-[10px] text-slate-500 border border-slate-600 px-1 rounded">${{s.sector.substring(0,10)}}</span>
                                    </div>
                                    <div class="flex items-center gap-2 mt-1">
                                        <span class="text-slate-400 font-mono text-sm">₹${{s.price}}</span>
                                        <span class="text-xs font-bold ${{s.change >= 0 ? 'text-green-400' : 'text-red-400'}}">${{s.change}}%</span>
                                    </div>
                                </div>
                                <div class="text-right">
                                    <div class="text-[10px] text-slate-500 uppercase mb-1">RS Score</div>
                                    <div class="text-sm font-mono font-bold ${{rsColor}}">${{s.rs > 0 ? '+' : ''}}${{s.rs}}</div>
                                </div>
                            </div>
                            
                            <div class="flex justify-between items-center mb-3">
                                <div class="flex flex-wrap gap-1 content-start w-2/3">${{badges}}</div>
                                <div class="text-right">
                                    <span class="dot ${{mDot}}"></span><span class="dot ${{wDot}}"></span><span class="dot ${{dDot}}"></span>
                                </div>
                            </div>

                            <div class="h-8 w-full opacity-60">
                                <svg width="100%" height="100%" preserveAspectRatio="none" class="overflow-visible"><polyline points="${{pts}}" fill="none" stroke="${{s.change >= 0 ? '#4ade80' : '#f87171'}}" stroke-width="2" /></svg>
                            </div>
                        </div>
                    `;
                }});
                lucide.createIcons();
            }}

            function openModal(sym) {{
                const s = data.find(x => x.symbol === sym);
                document.getElementById('m-sym').innerText = s.symbol;
                document.getElementById('m-sec').innerText = s.sector;
                document.getElementById('m-target').innerText = s.target;
                document.getElementById('m-stop').innerText = s.stop;
                document.getElementById('m-rs').innerText = s.rs;
                document.getElementById('modal').style.display = 'block';
            }}
            
            function closeModal() {{ document.getElementById('modal').style.display = 'none'; }}
            window.onclick = function(e) {{ if(e.target == document.getElementById('modal')) closeModal(); }}
            
            render();
        </script>
    </body>
    </html>
    """
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    with open(FILE_PATH, "w", encoding="utf-8") as f: f.write(html)
    print("Generated Institutional Dashboard.")

if __name__ == "__main__":
    session = get_session()
    tickers, sector_map = get_nifty500_data(session)
    fno_list = get_fno_list(session)
    nifty_chg = get_nifty_change()
    
    results = []
    print(f"Scanning {len(tickers)} stocks...")
    for i, t in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] {t}...", end="\r")
        time.sleep(0.1)
        is_fno = t in fno_list
        sec = sector_map.get(t, "Unknown")
        res = analyze_stock(t, is_fno, sec, nifty_chg)
        if res: results.append(res)
    
    generate_html(results, nifty_chg)

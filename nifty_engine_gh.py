import yfinance as yf
import pandas as pd
import os
import time
import requests
import io
import json
import warnings
import math
from concurrent.futures import ThreadPoolExecutor

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURATION ---
OUTPUT_DIR = "public"
FILE_PATH = os.path.join(OUTPUT_DIR, "index.html")
CSV_FILENAME = "ind_nifty500list.csv"
# We use a fallback list if NSE blocks the request
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

# --- 2. DATA FETCHING ---
def get_nifty500_data(session):
    print("Fetching Nifty 500 List...")
    sector_map = {}
    tickers = []
    try:
        # Try fetching live
        r = session.get(NSE_URL, timeout=10)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
        else:
            raise Exception("NSE Download Failed")
        
        if 'Industry' in df.columns:
            for index, row in df.iterrows():
                sym = f"{row['Symbol']}.NS"
                sector_map[sym] = row['Industry']
                tickers.append(sym)
        else:
            tickers = [f"{x}.NS" for x in df['Symbol'].dropna().tolist()]
    except Exception as e:
        print(f"Warning: Could not fetch Nifty 500 list ({e}). Using fallback top 20.")
        # Fallback list to ensure script always runs
        tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
                   "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
                   "LTIM.NS", "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
                   "TITAN.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS", "BAJFINANCE.NS", "HCLTECH.NS"]
            
    return tickers, sector_map

def get_fno_list(session):
    try:
        r = session.get(FNO_URL, timeout=10)
        df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
        df.columns = [c.strip() for c in df.columns]
        if 'SYMBOL' in df.columns:
            return [f"{x.strip()}.NS" for x in df['SYMBOL'].dropna().unique().tolist()]
    except: pass
    return []

# --- 3. HELPER: PIVOTS & CANDLES ---
def get_pivots(high, low, close):
    p = (high + low + close) / 3
    r1 = (2 * p) - low
    s1 = (2 * p) - high
    return round(r1, 2), round(s1, 2), round(p, 2)

def is_hammer(open_p, high, low, close):
    body = abs(close - open_p)
    if body == 0: return False
    lower = min(open_p, close) - low
    upper = high - max(open_p, close)
    return (lower > 2 * body) and (upper < body)

# --- 4. DECISION LOGIC ---
def calculate_verdict(trend_score, signals, is_fno, win_rate):
    score = 0
    reasons = []
    
    if trend_score == 3: score += 40; reasons.append("Full Bull Trend (M/W/D)")
    elif trend_score == 2: score += 25; reasons.append("Primary Trend Up")
    else: score -= 20; reasons.append("Weak Trend")

    if is_fno: score += 20; reasons.append("High Liquidity (F&O)")
    
    if "Golden Cross" in signals or "Bull Divergence" in signals: score += 20; reasons.append("Strong Signal")
    elif len(signals) > 0: score += 10; reasons.append("Breakout Detected")

    if win_rate > 60: score += 20; reasons.append("High Win Rate")
    elif win_rate < 40: score -= 10

    verdict = "WATCH"
    color = "gray"
    if score >= 80: verdict = "PRIME BUY"; color = "green"
    elif score >= 60: verdict = "STRONG BUY"; color = "blue"
    elif score >= 40: verdict = "RISKY"; color = "orange"
    else: verdict = "AVOID"; color = "red"

    return verdict, color, score, reasons

# --- 5. BACKTESTER ---
def run_backtest(df):
    trades = []
    wins, losses = 0, 0
    if len(df) < 200: return {"win_rate": 0, "total": 0, "log": []}

    df = df.copy()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['Trend'] = df['Close'] > df['SMA200']
    
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9).mean()
    df['Signal'] = (macd > sig) & (macd.shift(1) < sig.shift(1))

    in_pos = False
    entry_price = 0
    
    start = len(df) - 250
    if start < 200: start = 200
    
    # Simple simulation loop
    # Note: iterating rows is slow, but acceptable for backtesting small datasets
    close_prices = df['Close'].values
    trend_vals = df['Trend'].values
    sig_vals = df['Signal'].values
    dates = df.index
    
    for i in range(start, len(df)):
        price = close_prices[i]
        if in_pos:
            pnl = ((price - entry_price) / entry_price) * 100
            if pnl >= 6.0 or pnl <= -3.0:
                res = "WIN" if pnl > 0 else "LOSS"
                trades.append({"date": dates[i].strftime('%Y-%m-%d'), "entry": round(entry_price,2), "result": res, "pnl": round(pnl,2)})
                if pnl > 0: wins += 1
                else: losses += 1
                in_pos = False
        elif trend_vals[i] and sig_vals[i]:
            in_pos = True
            entry_price = price

    total = wins + losses
    rate = round((wins / total * 100), 0) if total > 0 else 0
    return {"win_rate": rate, "total": total, "log": trades[-5:]}

# --- 6. MASTER ANALYSIS ---
def analyze_stock(ticker, is_fno, sector, nifty_chg):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y", interval="1d") # 2y is enough for calculation, faster
        if df.empty or len(df) < 200: return None

        # A. TRENDS
        agg = {'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}
        df_w = df.resample('W').agg(agg).dropna()
        df_m = df.resample('ME').agg(agg).dropna()
        
        def get_trend(d): 
            if len(d) < 20: return "FLAT"
            return "UP" if d['Close'].iloc[-1] > d['Close'].ewm(span=20).mean().iloc[-1] else "DOWN"
            
        m_trend = get_trend(df_m)
        w_trend = get_trend(df_w)
        
        # B. DAILY
        close = df['Close']
        high = df['High']
        low = df['Low']
        vol = df['Volume']
        opn = df['Open']
        curr = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        change = round(((curr - prev) / prev) * 100, 2)
        rs_score = round(change - nifty_chg, 2)
        
        # C. INDICATORS
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        curr_sma200 = float(sma200.iloc[-1])
        d_trend = "UP" if curr > curr_sma200 else "DOWN"
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain / loss)))
        curr_rsi = float(rsi.iloc[-1])
        
        # RSI Divergence (20 Days)
        div_bull = False
        if len(close) > 20 and curr < close.iloc[-20:].min() and curr_rsi > rsi.iloc[-20:].min(): div_bull = True

        # Volume
        vol_avg = float(vol.rolling(20).mean().iloc[-1])
        vol_mult = round(float(vol.iloc[-1]) / vol_avg, 1) if vol_avg > 0 else 0
        
        # Buzz
        buzz = 0
        # News fetching can be slow, skipping for speed in this version
        # try:
        #    if stock.news: buzz = len([n for n in stock.news if (time.time() - n['providerPublishTime']) < 86400])
        # except: pass

        # D. SIGNALS
        signals = []
        if float(sma50.iloc[-2]) < float(sma200.iloc[-2]) and float(sma50.iloc[-1]) > float(sma200.iloc[-1]): signals.append("Golden Cross")
        if div_bull: signals.append("Bull Divergence")
        
        # NR7
        if (high.iloc[-1] - low.iloc[-1]) == (high - low).tail(7).min(): signals.append("NR7 Squeeze")
        
        # Hammer
        if is_hammer(float(opn.iloc[-1]), float(high.iloc[-1]), float(low.iloc[-1]), curr): signals.append("Hammer")
        
        if curr_rsi < 30: signals.append("Oversold")
        if vol_mult >= 2.5: signals.append(f"Vol {vol_mult}x")
        
        # 52W High check (last 250 days)
        if len(high) > 250:
            if prev < float(high.iloc[-250:-1].max()) and curr > float(high.iloc[-250:-1].max()): signals.append("52W High")

        # Filter: Only show interesting stocks to keep HTML size down
        trend_score = 0
        if m_trend == "UP": trend_score += 1
        if w_trend == "UP": trend_score += 1
        if d_trend == "UP": trend_score += 1
        
        # Skip junk stocks
        if not signals and trend_score < 2: return None
        
        bt = run_backtest(df)
        verdict, v_color, score, reasons = calculate_verdict(trend_score, signals, is_fno, bt['win_rate'])
        
        # Targets
        atr = float((high - low).rolling(14).mean().iloc[-1])
        r1, s1, pivot = get_pivots(float(high.iloc[-2]), float(low.iloc[-2]), float(close.iloc[-2]))

        return {
            "symbol": ticker.replace(".NS", ""),
            "price": round(curr, 2),
            "change": change,
            "sector": sector if sector else "Other",
            "rs": rs_score,
            "trends": {"M": m_trend, "W": w_trend, "D": d_trend},
            "signals": signals,
            "verdict": verdict,
            "v_color": v_color,
            "score": score,
            "reasons": reasons,
            "is_fno": is_fno,
            "buzz": buzz,
            "backtest": bt,
            "levels": {"R1": r1, "S1": s1, "P": pivot, "TGT": round(curr+(2*atr),1), "SL": round(curr-atr,1)},
            "history": [x if not math.isnan(x) else 0 for x in close.tail(30).tolist()]
        }

    except Exception as e:
        return None

# --- 7. GENERATE HTML ---
def generate_html(results, nifty_chg):
    # Market Stats calculation in Python, but passed to JS
    adv = len([x for x in results if x['change'] > 0])
    dec = len([x for x in results if x['change'] < 0])
    
    # Sector Stats calculation
    sec_perf = {}
    for r in results:
        s = r['sector']
        if s not in sec_perf: sec_perf[s] = []
        sec_perf[s].append(r['change'])
    
    sectors = sorted([{"name": k, "avg": round(sum(v)/len(v),2)} for k,v in sec_perf.items()], key=lambda x: x['avg'], reverse=True)[:6]

    # Prepare data for Injection
    final_data = {
        "stocks": results, 
        "sectors": sectors, 
        "stats": {"adv": adv, "dec": dec, "nifty": round(nifty_chg, 2)}
    }
    json_data = json.dumps(final_data)
    
    # HTML Template
    # NOTICE: Double curly braces {{ }} for CSS/JS, Single { } for Python f-string variable injection
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Ultra Scanner</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            body {{ background-color: #0f172a; color: #e2e8f0; font-family: sans-serif; }}
            .modal {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.9); z-index: 50; backdrop-filter: blur(5px); }}
            .modal-content {{ background: #1e293b; margin: 5vh auto; width: 95%; max-width: 600px; max-height: 90vh; overflow-y: auto; border-radius: 12px; border: 1px solid #334155; }}
            
            .dot {{ height: 8px; width: 8px; border-radius: 50%; display: inline-block; margin-right: 2px; }}
            .dot-up {{ background-color: #4ade80; box-shadow: 0 0 5px #4ade80; }}
            .dot-down {{ background-color: #f87171; opacity: 0.4; }}
            
            .b-std {{ background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }}
            .b-gold {{ background: rgba(250, 204, 21, 0.15); color: #facc15; border: 1px solid rgba(250, 204, 21, 0.3); }}
            .b-nr7 {{ background: rgba(236, 72, 153, 0.15); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.3); }}
            .b-div {{ background: #ec4899; color: white; box-shadow: 0 0 8px #ec4899; }}
            
            .v-green {{ color: #4ade80; border: 1px solid #4ade80; background: rgba(74, 222, 128, 0.1); }}
            .v-blue {{ color: #60a5fa; border: 1px solid #60a5fa; background: rgba(96, 165, 250, 0.1); }}
            .v-orange {{ color: #fb923c; border: 1px solid #fb923c; background: rgba(251, 146, 60, 0.1); }}
        </style>
    </head>
    <body>
        <div class="max-w-7xl mx-auto p-4">
            <!-- HEADER -->
            <header class="mb-4 border-b border-slate-700 pb-4">
                <div class="flex justify-between items-end mb-4">
                    <div>
                        <h1 class="text-2xl font-bold text-blue-400 flex items-center gap-2"><i data-lucide="zap"></i> Ultra Scanner</h1>
                        <div class="flex items-center gap-2 text-xs mt-1 text-slate-400">
                            <span class="text-green-400 font-bold">ADV <span id="stat-adv">--</span></span> / 
                            <span class="text-red-400 font-bold">DEC <span id="stat-dec">--</span></span>
                            <span>• NIFTY <span id="stat-nifty">--</span>%</span>
                        </div>
                    </div>
                </div>
                <!-- SECTORS -->
                <div id="sector-container" class="flex gap-2 overflow-x-auto pb-2">
                    <!-- JS Injected -->
                </div>
                <!-- FILTERS -->
                <div class="flex bg-slate-800 p-1 rounded-lg mt-3 overflow-x-auto gap-1">
                    <button onclick="setFilter('ALL')" id="btn-all" class="px-4 py-1.5 rounded text-xs font-bold bg-blue-600 text-white">All</button>
                    <button onclick="setFilter('PRIME')" id="btn-prime" class="px-4 py-1.5 rounded text-xs font-bold text-slate-400">Prime Buy</button>
                    <button onclick="setFilter('FNO')" id="btn-fno" class="px-4 py-1.5 rounded text-xs font-bold text-slate-400">F&O</button>
                    <button onclick="setFilter('RS')" id="btn-rs" class="px-4 py-1.5 rounded text-xs font-bold text-slate-400">Strong RS</button>
                </div>
            </header>
            
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4" id="grid">
                <div class="col-span-3 text-center text-slate-500 mt-10">Loading Data...</div>
            </div>
        </div>

        <!-- MODAL -->
        <div id="modal" class="modal">
            <div class="modal-content">
                <div class="sticky top-0 bg-slate-800 p-4 border-b border-slate-700 flex justify-between items-center">
                    <div><h2 id="m-sym" class="text-2xl font-bold text-white">SYMBOL</h2></div>
                    <button onclick="closeModal()" class="p-2 bg-slate-700 rounded-full hover:bg-slate-600"><i data-lucide="x"></i></button>
                </div>
                <div class="p-6 space-y-6">
                    <div class="bg-slate-800 p-4 rounded-xl border border-slate-700">
                        <h3 class="text-xs text-slate-500 uppercase mb-2">Analysis Verdict</h3>
                        <ul id="m-reasons" class="list-disc list-inside text-sm text-slate-300 space-y-1"></ul>
                        <div class="mt-3 pt-3 border-t border-slate-700 flex justify-between text-sm">
                            <span class="text-slate-500">SCORE</span><span id="m-score" class="font-bold text-white">0/100</span>
                        </div>
                    </div>
                    <div class="grid grid-cols-2 gap-4">
                        <div class="bg-green-900/20 border border-green-500/30 p-3 rounded text-center">
                            <div class="text-xs text-green-400 font-bold">TARGET</div><div id="m-tgt" class="text-xl font-bold text-white">0</div>
                        </div>
                        <div class="bg-red-900/20 border border-red-500/30 p-3 rounded text-center">
                            <div class="text-xs text-red-400 font-bold">STOP</div><div id="m-sl" class="text-xl font-bold text-white">0</div>
                        </div>
                    </div>
                    <div class="bg-slate-800 p-3 rounded border border-slate-700 flex justify-between text-sm">
                        <div class="text-center"><div class="text-red-400">R1</div><div id="m-r1">0</div></div>
                        <div class="text-center border-x border-slate-600 px-4"><div class="text-blue-400">Pivot</div><div id="m-p">0</div></div>
                        <div class="text-center"><div class="text-green-400">S1</div><div id="m-s1">0</div></div>
                    </div>
                    <!-- Logs -->
                    <div class="overflow-hidden rounded border border-slate-700">
                        <table class="w-full text-xs">
                            <thead class="bg-slate-900 text-slate-300"><tr><th class="p-2 text-left">Date</th><th class="p-2 text-right">Entry</th><th class="p-2 text-right">Result</th><th class="p-2 text-right">P&L</th></tr></thead>
                            <tbody id="m-logs" class="bg-slate-800 text-slate-300 divide-y divide-slate-700/50"></tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // INJECT PYTHON DATA HERE
            const RAW = {json_data};
            const data = RAW.stocks;
            let currentFilter = 'ALL';
            
            // --- MAIN RENDER LOGIC ---
            function initDashboard() {{
                // 1. Render Stats
                document.getElementById('stat-adv').innerText = RAW.stats.adv;
                document.getElementById('stat-dec').innerText = RAW.stats.dec;
                document.getElementById('stat-nifty').innerText = RAW.stats.nifty;

                // 2. Render Sectors
                const sectorContainer = document.getElementById('sector-container');
                let sectorHTML = '';
                RAW.sectors.forEach(s => {{
                   const colorClass = s.avg >= 0 ? 'text-green-400' : 'text-red-400';
                   sectorHTML += `
                    <div class="bg-slate-800 px-3 py-1.5 rounded border border-slate-700 whitespace-nowrap">
                        <span class="text-[10px] text-slate-400 mr-2">${{s.name}}</span>
                        <span class="font-bold text-xs ${{colorClass}}">${{s.avg}}%</span>
                    </div>`;
                }});
                sectorContainer.innerHTML = sectorHTML;

                // 3. Render Grid
                renderGrid();
                
                // 4. Initialize Icons
                lucide.createIcons();
            }}

            function setFilter(type) {{
                currentFilter = type;
                ['btn-all','btn-fno','btn-rs','btn-prime'].forEach(id => document.getElementById(id).className = "px-4 py-1.5 rounded text-xs font-bold text-slate-400 hover:text-white transition");
                document.getElementById('btn-'+type.toLowerCase()).className = "px-4 py-1.5 rounded text-xs font-bold bg-blue-600 text-white transition";
                renderGrid();
            }}

            function renderGrid() {{
                const grid = document.getElementById('grid');
                grid.innerHTML = '';
                
                const filtered = data.filter(s => {{
                    if(currentFilter === 'FNO') return s.is_fno;
                    if(currentFilter === 'RS') return s.rs > 1.5;
                    if(currentFilter === 'PRIME') return s.verdict === 'PRIME BUY';
                    return true;
                }});

                if(filtered.length === 0) grid.innerHTML = '<div class="col-span-3 text-center text-slate-500 p-10">No results found.</div>';

                let gridHTML = '';
                filtered.forEach((s) => {{
                    // Dots
                    const mDot = s.trends.M === 'UP' ? 'dot-up' : 'dot-down';
                    const wDot = s.trends.W === 'UP' ? 'dot-up' : 'dot-down';
                    const dDot = s.trends.D === 'UP' ? 'dot-up' : 'dot-down';
                    
                    // Verdict Class
                    let vClass = 'v-orange';
                    if(s.v_color === 'green') vClass = 'v-green';
                    if(s.v_color === 'blue') vClass = 'v-blue';
                    if(s.v_color === 'red') vClass = 'text-red-500 border border-red-500 bg-red-900/20';

                    // Badges
                    let badges = '';
                    if(s.is_fno) badges += `<span class="bg-blue-900/50 text-blue-300 border border-blue-500/30 px-1.5 py-0.5 rounded text-[10px] font-bold mr-1">F&O</span>`;
                    
                    s.signals.forEach(sig => {{
                        let cls = 'b-std';
                        if(sig.includes('NR7')) cls = 'b-nr7';
                        if(sig.includes('Divergence')) cls = 'b-div';
                        if(sig.includes('Golden')) cls = 'b-gold';
                        badges += `<span class="${{cls}} px-1.5 py-0.5 rounded text-[10px] font-bold uppercase mr-1 mb-1 inline-block">${{sig}}</span>`;
                    }});

                    // Mini Chart Points
                    const pts = s.history.map((d, j) => {{
                        const min = Math.min(...s.history); 
                        const max = Math.max(...s.history);
                        const range = max - min || 1;
                        const x = (j / (s.history.length - 1)) * 100;
                        const y = 30 - ((d - min) / range) * 30;
                        return `${{x}},${{y}}`;
                    }}).join(' ');
                    
                    const strokeColor = s.change >= 0 ? '#4ade80' : '#f87171';
                    const changeColor = s.change >= 0 ? 'text-green-400' : 'text-red-400';

                    gridHTML += `
                        <div class="card bg-slate-800 p-4 rounded-xl border border-slate-700 shadow-lg relative group cursor-pointer hover:border-slate-500 transition" onclick="openModal('${{s.symbol}}')">
                            <div class="flex justify-between items-start mb-2">
                                <div>
                                    <div class="flex items-center gap-2">
                                        <div class="font-bold text-lg text-white group-hover:text-blue-400 transition">${{s.symbol}}</div>
                                        <span class="text-[10px] text-slate-500 border border-slate-600 px-1 rounded">${{s.sector.substring(0,10)}}</span>
                                    </div>
                                    <div class="text-xs text-slate-400 font-mono mt-1">₹${{s.price}} <span class="${{changeColor}} ml-1">${{s.change}}%</span></div>
                                </div>
                                <div class="text-right">
                                    <div class="px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wider ${{vClass}}">${{s.verdict}}</div>
                                    <div class="mt-1"><span class="dot ${{mDot}}"></span><span class="dot ${{wDot}}"></span><span class="dot ${{dDot}}"></span></div>
                                </div>
                            </div>
                            
                            <div class="flex flex-wrap gap-1 mb-3 min-h-[1.5rem] content-start">${{badges}}</div>
                            
                            <div class="h-8 w-full opacity-60">
                                <svg width="100%" height="100%" preserveAspectRatio="none" class="overflow-visible"><polyline points="${{pts}}" fill="none" stroke="${{strokeColor}}" stroke-width="2" /></svg>
                            </div>
                        </div>
                    `;
                }});
                
                grid.innerHTML = gridHTML;
                lucide.createIcons();
            }}

            function openModal(sym) {{
                const s = data.find(x => x.symbol === sym);
                document.getElementById('m-sym').innerText = s.symbol;
                document.getElementById('m-tgt').innerText = s.levels.TGT;
                document.getElementById('m-sl').innerText = s.levels.SL;
                document.getElementById('m-r1').innerText = s.levels.R1;
                document.getElementById('m-p').innerText = s.levels.P;
                document.getElementById('m-s1').innerText = s.levels.S1;
                document.getElementById('m-score').innerText = s.score + '/100';
                
                const list = document.getElementById('m-reasons');
                list.innerHTML = '';
                s.reasons.forEach(r => list.innerHTML += `<li>${{r}}</li>`);
                
                let rows = '';
                const bt = s.backtest;
                if(bt.log.length === 0) rows = '<tr><td colspan="4" class="p-4 text-center italic text-slate-500">No recent signals.</td></tr>';
                else {{
                    [...bt.log].reverse().forEach(l => {{
                        let c = l.result.includes('WIN') ? 'text-green-400 font-bold' : (l.result.includes('LOSS') ? 'text-red-400' : 'text-yellow-400');
                        rows += `<tr class="hover:bg-slate-700/50"><td class="p-2">${{l.date}}</td><td class="p-2 text-right font-mono">${{l.entry}}</td><td class="p-2 text-right ${{c}} text-[10px]">${{l.result}}</td><td class="p-2 text-right font-mono ${{c}}">${{l.pnl}}%</td></tr>`;
                    }});
                }}
                document.getElementById('m-logs').innerHTML = rows;
                document.getElementById('modal').style.display = 'block';
            }}
            
            function closeModal() {{ document.getElementById('modal').style.display = 'none'; }}
            window.onclick = function(e) {{ if(e.target == document.getElementById('modal')) closeModal(); }}
            
            // Start
            initDashboard();
        </script>
    </body>
    </html>
    """
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    with open(FILE_PATH, "w", encoding="utf-8") as f: f.write(html)
    print(f"Generated Ultra Dashboard at: {FILE_PATH}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    session = get_session()
    
    # 1. Get List
    tickers, sector_map = get_nifty500_data(session)
    fno_list = get_fno_list(session)
    
    # 2. Get Market Context
    nifty_chg = 0
    try:
        nifty = yf.Ticker("^NSEI")
        h = nifty.history(period="2d")
        nifty_chg = ((h['Close'].iloc[-1] - h['Close'].iloc[-2])/h['Close'].iloc[-2])*100
    except: pass
    
    print(f"Scanning {len(tickers)} stocks with ThreadPool...")
    results = []

    # 3. Multi-threaded Scanning
    # We use ThreadPool to speed up YFinance blocking calls
    def scan_task(ticker):
        is_fno = ticker in fno_list
        sec = sector_map.get(ticker, "Unknown")
        return analyze_stock(ticker, is_fno, sec, nifty_chg)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = list(executor.map(scan_task, tickers))
        
    results = [r for r in futures if r is not None]

    # 4. Generate
    generate_html(results, nifty_chg)

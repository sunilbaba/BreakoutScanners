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

# --- 2. DATA FETCHING ---
def get_nifty500_list(session):
    print("Fetching Nifty 500 List...")
    try:
        if os.path.exists(CSV_FILENAME):
            df = pd.read_csv(CSV_FILENAME)
        else:
            r = session.get(NSE_URL, timeout=10)
            df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
        return [f"{x}.NS" for x in df['Symbol'].dropna().tolist()]
    except:
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]

def get_fno_list(session):
    try:
        r = session.get(FNO_URL, timeout=10)
        df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
        df.columns = [c.strip() for c in df.columns]
        if 'SYMBOL' in df.columns:
            return [f"{x.strip()}.NS" for x in df['SYMBOL'].dropna().unique().tolist()]
    except: pass
    return []

# --- 3. HELPER: PIVOT POINTS (Institutional Levels) ---
def get_pivots(high, low, close):
    """Calculates Standard Floor Pivots for Support/Resistance"""
    p = (high + low + close) / 3
    r1 = (2 * p) - low
    s1 = (2 * p) - high
    r2 = p + (high - low)
    s2 = p - (high - low)
    return round(r1, 2), round(s1, 2), round(p, 2)

# --- 4. HELPER: RSI DIVERGENCE ---
def check_divergence(close, rsi):
    """
    Detects Bullish Divergence: Price makes Lower Low, RSI makes Higher Low.
    Lookback: Last 20 candles.
    """
    if len(close) < 20: return False
    
    # Get recent low and previous low in a window
    curr_price_low = close.iloc[-5:].min()
    prev_price_low = close.iloc[-20:-5].min()
    
    curr_rsi_low = rsi.iloc[-5:].min()
    prev_rsi_low = rsi.iloc[-20:-5].min()
    
    # Logic: Price is lower, but RSI is higher (Strength building)
    if curr_price_low < prev_price_low and curr_rsi_low > prev_rsi_low:
        return True
    return False

# --- 5. BACKTESTER ---
def run_backtest(df):
    trades = []
    wins, losses = 0, 0
    if len(df) < 200: return {"win_rate": 0, "total": 0, "log": []}

    # Indicators
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['Trend'] = df['Close'] > df['SMA200']
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    df['Signal_MACD'] = (macd > sig) & (macd.shift(1) < sig.shift(1))

    in_pos = False
    entry_price = 0
    entry_date = None
    days = 0
    
    start = len(df) - 250
    if start < 200: start = 200
    
    for i in range(start, len(df)):
        price = df['Close'].iloc[i]
        date = df.index[i].strftime('%Y-%m-%d')
        
        if in_pos:
            days += 1
            pnl = ((price - entry_price) / entry_price) * 100
            res = None
            if pnl >= 6.0: res = "WIN"
            elif pnl <= -3.0: res = "LOSS"
            elif days >= 20: res = "TIMEOUT"
            
            if res:
                trades.append({"date": entry_date, "entry": round(entry_price, 2), "exit": round(price, 2), "result": res, "pnl": round(pnl, 2)})
                if pnl > 0: wins += 1
                else: losses += 1
                in_pos = False
        else:
            if df['Trend'].iloc[i] and df['Signal_MACD'].iloc[i]:
                in_pos = True
                entry_price = price
                entry_date = date
                days = 0

    total = wins + losses
    rate = round((wins / total * 100), 0) if total > 0 else 0
    return {"win_rate": rate, "total": total, "wins": wins, "log": trades[-5:]}

# --- 6. MASTER ANALYSIS ---
def analyze_stock(ticker, is_fno):
    try:
        stock = yf.Ticker(ticker)
        df_d = stock.history(period="5y", interval="1d")
        if df_d.empty or len(df_d) < 300: return None

        # A. TRENDS
        agg = {'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}
        df_w = df_d.resample('W').agg(agg).dropna()
        df_m = df_d.resample('ME').agg(agg).dropna()
        
        def get_trend(df): return "UP" if df['Close'].iloc[-1] > df['Close'].ewm(span=20).mean().iloc[-1] else "DOWN"
        m_trend = get_trend(df_m)
        w_trend = get_trend(df_w)
        
        # B. DAILY DATA
        close = df_d['Close']
        high = df_d['High']
        low = df_d['Low']
        vol = df_d['Volume']
        curr = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        change = round(((curr - prev) / prev) * 100, 2)
        
        # C. PIVOT POINTS (Institutional Levels)
        # Using yesterday's High/Low/Close to project Today's Support/Resistance
        y_high = float(high.iloc[-2])
        y_low = float(low.iloc[-2])
        y_close = float(close.iloc[-2])
        r1, s1, pivot = get_pivots(y_high, y_low, y_close)
        
        # D. INDICATORS
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        curr_sma200 = float(sma200.iloc[-1])
        d_trend = "UP" if curr > curr_sma200 else "DOWN"
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        curr_rsi = float(rsi.iloc[-1])
        
        # VOLUME
        vol_avg = float(vol.rolling(20).mean().iloc[-1])
        vol_mult = round(float(vol.iloc[-1]) / vol_avg, 1) if vol_avg > 0 else 0
        
        # E. BUZZ
        buzz = 0
        try:
            if stock.news: buzz = len([n for n in stock.news if (time.time() - n['providerPublishTime']) < 86400])
        except: pass

        # F. SIGNALS
        signals = []
        
        # 1. RSI Divergence (NEW)
        if check_divergence(close, rsi): signals.append("Bull Divergence")
        
        # 2. Pivot Breakout (NEW)
        if prev < r1 and curr > r1: signals.append("R1 Breakout")
        if prev > s1 and curr < s1: signals.append("S1 Breakdown")
        
        # 3. Standard Signals
        if float(sma50.iloc[-2]) < float(sma200.iloc[-2]) and float(sma50.iloc[-1]) > float(sma200.iloc[-1]): signals.append("Golden Cross")
        if curr_rsi < 30: signals.append("Oversold")
        if vol_mult >= 2.5: signals.append(f"Vol {vol_mult}x")
        
        # Filter
        confluence = 0
        if m_trend == "UP": confluence += 1
        if w_trend == "UP": confluence += 1
        if d_trend == "UP": confluence += 1
        
        if not signals and not (is_fno and confluence == 3): return None
        
        bt = run_backtest(df_d)

        return {
            "symbol": ticker.replace(".NS", ""),
            "price": round(curr, 2),
            "change": change,
            "trends": {"M": m_trend, "W": w_trend, "D": d_trend},
            "signals": signals,
            "buzz": buzz,
            "is_fno": is_fno,
            "backtest": bt,
            "levels": {"R1": r1, "S1": s1, "P": pivot}, # Send Pivots to UI
            "rsi": round(curr_rsi, 1),
            "history": [x if not math.isnan(x) else 0 for x in close.tail(30).tolist()]
        }

    except Exception:
        return None

# --- 7. GENERATE HTML ---
def generate_html(results):
    json_data = json.dumps(results)
    
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
            .card {{ transition: all 0.2s; cursor: pointer; }}
            .card:hover {{ transform: translateY(-3px); border-color: #60a5fa; }}
            .modal {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.9); z-index: 50; backdrop-filter: blur(5px); }}
            .modal-content {{ background: #1e293b; margin: 5vh auto; width: 95%; max-width: 600px; max-height: 90vh; overflow-y: auto; border-radius: 12px; border: 1px solid #334155; }}
            
            .dot {{ height: 8px; width: 8px; border-radius: 50%; display: inline-block; margin-right: 2px; }}
            .dot-up {{ background-color: #4ade80; box-shadow: 0 0 5px #4ade80; }}
            .dot-down {{ background-color: #f87171; opacity: 0.4; }}
            
            .b-std {{ background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }}
            .b-gold {{ background: rgba(250, 204, 21, 0.15); color: #facc15; border: 1px solid rgba(250, 204, 21, 0.3); }}
            .b-div {{ background: #ec4899; color: white; box-shadow: 0 0 8px #ec4899; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="max-w-6xl mx-auto p-4">
            <header class="flex flex-col md:flex-row justify-between items-center mb-6 border-b border-slate-700 pb-4 gap-4">
                <div>
                    <h1 class="text-2xl font-bold text-blue-400 flex items-center gap-2">
                        <i data-lucide="landmark"></i> Institutional Scanner
                    </h1>
                    <p class="text-xs text-slate-500 mt-1">Divergence + Pivot Points + F&O</p>
                </div>
                
                <div class="flex bg-slate-800 p-1 rounded-lg overflow-x-auto">
                    <button onclick="setFilter('ALL')" id="btn-all" class="px-4 py-1.5 rounded text-xs font-bold bg-blue-600 text-white">All</button>
                    <button onclick="setFilter('FNO')" id="btn-fno" class="px-4 py-1.5 rounded text-xs font-bold text-slate-400">F&O</button>
                    <button onclick="setFilter('DIV')" id="btn-div" class="px-4 py-1.5 rounded text-xs font-bold text-slate-400">Divergence</button>
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
                        <div class="text-xs text-slate-400">Institutional Levels</div>
                    </div>
                    <button onclick="closeModal()" class="p-2 bg-slate-700 rounded-full hover:bg-slate-600"><i data-lucide="x"></i></button>
                </div>
                
                <div class="p-6 space-y-6">
                    <!-- PIVOT POINTS -->
                    <div class="bg-slate-800 p-4 rounded-xl border border-slate-700">
                        <h3 class="text-xs text-slate-500 uppercase mb-3">Support & Resistance (Pivot Points)</h3>
                        <div class="flex justify-between items-center text-sm">
                            <div class="text-center">
                                <div class="text-red-400 font-bold">R1 (Res)</div>
                                <div id="m-r1" class="text-white text-lg">0</div>
                            </div>
                            <div class="text-center border-l border-r border-slate-600 px-4">
                                <div class="text-blue-400 font-bold">Pivot</div>
                                <div id="m-p" class="text-white text-lg">0</div>
                            </div>
                            <div class="text-center">
                                <div class="text-green-400 font-bold">S1 (Supp)</div>
                                <div id="m-s1" class="text-white text-lg">0</div>
                            </div>
                        </div>
                    </div>

                    <!-- Backtest -->
                    <div>
                        <h3 class="text-sm font-bold text-slate-300 mb-2">Backtest (1Y)</h3>
                        <div class="grid grid-cols-3 gap-2 mb-4">
                            <div class="bg-slate-800 p-2 rounded text-center"><div class="text-[10px] text-slate-500">WIN RATE</div><div id="m-rate" class="text-lg font-bold text-white">0%</div></div>
                            <div class="bg-slate-800 p-2 rounded text-center"><div class="text-[10px] text-slate-500">TRADES</div><div id="m-total" class="text-lg font-bold text-white">0</div></div>
                            <div class="bg-slate-800 p-2 rounded text-center"><div class="text-[10px] text-slate-500">WINS</div><div id="m-wins" class="text-lg font-bold text-green-400">0</div></div>
                        </div>
                        <div class="overflow-hidden rounded border border-slate-700">
                            <table class="w-full text-xs">
                                <thead class="bg-slate-900 text-slate-300"><tr><th class="p-2 text-left">Date</th><th class="p-2 text-right">Entry</th><th class="p-2 text-right">Result</th><th class="p-2 text-right">P&L</th></tr></thead>
                                <tbody id="m-logs" class="bg-slate-800 text-slate-300 divide-y divide-slate-700/50"></tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const data = {json_data};
            let currentFilter = 'ALL';
            lucide.createIcons();
            
            function setFilter(type) {{
                currentFilter = type;
                ['btn-all','btn-fno','btn-div'].forEach(id => document.getElementById(id).className = "px-4 py-1.5 rounded text-xs font-bold text-slate-400 hover:text-white transition");
                document.getElementById('btn-'+type.toLowerCase()).className = "px-4 py-1.5 rounded text-xs font-bold bg-blue-600 text-white transition";
                render();
            }}

            function render() {{
                const grid = document.getElementById('grid');
                grid.innerHTML = '';
                
                const filtered = data.filter(s => {{
                    if(currentFilter === 'FNO') return s.is_fno;
                    if(currentFilter === 'DIV') return s.signals.some(x => x.includes('Divergence'));
                    return true;
                }});

                if(filtered.length === 0) grid.innerHTML = '<div class="col-span-3 text-center text-slate-500 p-10">No stocks match this filter.</div>';

                filtered.forEach((s, i) => {{
                    // Trends
                    const mDot = s.trends.M === 'UP' ? 'dot-up' : 'dot-down';
                    const wDot = s.trends.W === 'UP' ? 'dot-up' : 'dot-down';
                    const dDot = s.trends.D === 'UP' ? 'dot-up' : 'dot-down';
                    
                    // Badges
                    let badges = '';
                    if(s.is_fno) badges += `<span class="b-std px-2 py-1 rounded text-[10px] font-bold mr-1 mb-1 inline-block text-blue-300">F&O</span>`;
                    if(s.buzz > 0) badges += `<span class="b-std px-2 py-1 rounded text-[10px] font-bold mr-1 mb-1 inline-block flex items-center gap-1"><i data-lucide="flame" class="w-3"></i> ${{s.buzz}} News</span>`;
                    
                    s.signals.forEach(sig => {{
                        let cls = 'b-std';
                        if(sig.includes('Divergence')) cls = 'b-div';
                        if(sig.includes('Golden')) cls = 'b-gold';
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
                                    <div class="font-bold text-lg text-white group-hover:text-blue-400 transition">${{s.symbol}}</div>
                                    <div class="flex items-center gap-2">
                                        <span class="text-slate-400 font-mono text-sm">₹${{s.price}}</span>
                                        <span class="text-xs font-bold ${{s.change >= 0 ? 'text-green-400' : 'text-red-400'}}">${{s.change}}%</span>
                                    </div>
                                </div>
                                <div class="text-right">
                                    <div class="text-[10px] text-slate-500 uppercase mb-1">M / W / D</div>
                                    <div><span class="dot ${{mDot}}"></span><span class="dot ${{wDot}}"></span><span class="dot ${{dDot}}"></span></div>
                                </div>
                            </div>
                            
                            <div class="flex flex-wrap gap-1 mb-3 min-h-[1.5rem] content-start">${{badges}}</div>
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
                const bt = s.backtest;
                document.getElementById('m-sym').innerText = s.symbol;
                document.getElementById('m-r1').innerText = s.levels.R1;
                document.getElementById('m-p').innerText = s.levels.P;
                document.getElementById('m-s1').innerText = s.levels.S1;
                
                document.getElementById('m-rate').innerText = bt.win_rate + '%';
                document.getElementById('m-total').innerText = bt.total;
                document.getElementById('m-wins').innerText = bt.wins;
                
                let rows = '';
                if(bt.log.length === 0) rows = '<tr><td colspan="4" class="p-4 text-center italic text-slate-500">No signals in last 12 months.</td></tr>';
                else {{
                    [...bt.log].reverse().forEach(l => {{
                        let c = l.result.includes('WIN') ? 'text-green-400 font-bold' : (l.result.includes('LOSS') ? 'text-red-400' : 'text-yellow-400');
                        rows += `<tr class="hover:bg-slate-700/50"><td class="p-2">${{l.date}}</td><td class="p-2 text-right font-mono">${{l.entry}}</td><td class="p-2 text-right ${{c}} text-[10px]">${{l.result}}</td><td class="p-2 text-right font-mono ${{c}}">${{l.pnl}}%</td></tr>`;
                    }});
                }}
                document.getElementById('m-logs').innerHTML = rows;
                document.getElementById('modal').style.display = 'block';
                lucide.createIcons();
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
    tickers = get_nifty500_list(session)
    fno_list = get_fno_list(session)
    
    results = []
    print(f"Scanning {len(tickers)} stocks...")
    for i, t in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] {t}...", end="\r")
        time.sleep(0.2)
        is_fno = t in fno_list
        res = analyze_stock(t, is_fno)
        if res: results.append(res)
    
    generate_html(results)

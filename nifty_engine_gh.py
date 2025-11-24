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

# --- 1. GET LIST ---
def get_nifty500_list():
    if os.path.exists(CSV_FILENAME):
        try:
            df = pd.read_csv(CSV_FILENAME)
            return [f"{x}.NS" for x in df['Symbol'].dropna().tolist()]
        except: pass
    
    print("Downloading Nifty 500 list...")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(NSE_URL, headers=headers, timeout=10)
        df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
        return [f"{x}.NS" for x in df['Symbol'].dropna().tolist()]
    except:
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]

# --- 2. BACKTESTER ---
def run_backtest(df):
    trades = []
    wins = 0
    losses = 0
    
    # Need enough data
    if len(df) < 200: return {"win_rate": 0, "total": 0, "log": []}

    # Calculate Logic for Backtest (Vectorized for speed)
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['Trend'] = df['Close'] > df['SMA200']
    
    # Triggers
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['Signal_MACD'] = (macd > signal) & (macd.shift(1) < signal.shift(1))
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    df['Signal_RSI'] = (rsi < 40) & (rsi.shift(1) > rsi) # Dip
    
    # SIMULATION
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
            if pnl >= 6.0: res = "WIN (Target)"
            elif pnl <= -3.0: res = "LOSS (Stop)"
            elif days >= 20: res = "TIMEOUT"
            
            if res:
                trades.append({"date": entry_date, "entry": round(entry_price, 2), "exit": round(price, 2), "result": res, "pnl": round(pnl, 2)})
                if pnl > 0: wins += 1
                else: losses += 1
                in_pos = False
        else:
            if df['Trend'].iloc[i] and (df['Signal_MACD'].iloc[i] or df['Signal_RSI'].iloc[i]):
                in_pos = True
                entry_price = price
                entry_date = date
                days = 0

    total = wins + losses
    rate = round((wins / total * 100), 0) if total > 0 else 0
    return {"win_rate": rate, "total": total, "wins": wins, "log": trades[-5:]} # Keep 5 logs

# --- 3. MULTI-TIMEFRAME HELPER ---
def get_trend_status(df, tf_name):
    if len(df) < 20: return "NEUTRAL"
    ema20 = df['Close'].ewm(span=20, adjust=False).mean()
    return "UP" if df['Close'].iloc[-1] > ema20.iloc[-1] else "DOWN"

# --- 4. MASTER ANALYZER ---
def analyze_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Fetch 5 years for Weekly/Monthly charts
        df_d = stock.history(period="5y", interval="1d")
        
        if df_d.empty or len(df_d) < 300: return None

        # --- A. MULTI-TIMEFRAME TRENDS ---
        agg = {'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}
        df_w = df_d.resample('W').agg(agg).dropna()
        df_m = df_d.resample('ME').agg(agg).dropna()
        
        m_trend = get_trend_status(df_m, "Monthly")
        w_trend = get_trend_status(df_w, "Weekly")
        
        # --- B. DAILY ANALYSIS (Fresh Breakouts) ---
        close = df_d['Close']
        high = df_d['High']
        low = df_d['Low']
        vol = df_d['Volume']
        opn = df_d['Open']
        
        curr = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        change = round(((curr - prev) / prev) * 100, 2)
        
        # Indicators
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        curr_sma200 = float(sma200.iloc[-1])
        prev_sma200 = float(sma200.iloc[-2])
        curr_sma50 = float(sma50.iloc[-1])
        prev_sma50 = float(sma50.iloc[-2])
        
        # Daily Trend
        d_trend = "UP" if curr > curr_sma200 else "DOWN"
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        curr_rsi = float(rsi.iloc[-1])
        prev_rsi = float(rsi.iloc[-2])
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        sig = macd.ewm(span=9, adjust=False).mean()
        curr_macd = float(macd.iloc[-1])
        curr_sig = float(sig.iloc[-1])
        prev_macd = float(macd.iloc[-2])
        prev_sig = float(sig.iloc[-2])
        
        # Bollinger
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = sma20 + (2 * std20)
        curr_upper = float(upper.iloc[-1])
        prev_upper = float(upper.iloc[-2])
        
        # ATR Target
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        target = round(curr + (2 * atr), 1)
        stop = round(curr - (1 * atr), 1)

        # --- C. BUZZ ---
        buzz = 0
        try:
            if stock.news: buzz = len([n for n in stock.news if (time.time() - n['providerPublishTime']) < 86400])
        except: pass

        # --- D. SIGNALS ---
        signals = []
        if prev_sma50 < prev_sma200 and curr_sma50 > curr_sma200: signals.append("Golden Cross")
        if prev_macd < prev_sig and curr_macd > curr_sig: signals.append("MACD Buy")
        if prev < curr_upper and curr > curr_upper: signals.append("BB Blast")
        if curr_rsi < 15: signals.append("RSI < 15")
        elif curr_rsi < 30: signals.append("Oversold")
        
        # Engulfing
        prev_open = float(opn.iloc[-2])
        curr_open = float(opn.iloc[-1])
        if (prev < prev_open) and (curr > curr_open) and (curr_open < prev) and (curr > prev_open):
            signals.append("Bull Engulfing")

        # Filter: Show if Signal OR Trend is Strong (3/3 UP)
        confluence = 0
        if m_trend == "UP": confluence += 1
        if w_trend == "UP": confluence += 1
        if d_trend == "UP": confluence += 1
        
        if not signals and confluence < 3: return None
        
        # --- E. BACKTEST ---
        bt = run_backtest(df_d)

        return {
            "symbol": ticker.replace(".NS", ""),
            "price": round(curr, 2),
            "change": change,
            "trends": {"M": m_trend, "W": w_trend, "D": d_trend},
            "signals": signals,
            "buzz": buzz,
            "backtest": bt,
            "target": target,
            "stop": stop,
            "rsi": round(curr_rsi, 1),
            "history": [x if not math.isnan(x) else 0 for x in close.tail(30).tolist()]
        }

    except Exception:
        return None

# --- 5. GENERATE HTML ---
def generate_html(results):
    json_data = json.dumps(results)
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Nifty Ultimate Scanner</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            body {{ background-color: #0f172a; color: #e2e8f0; font-family: sans-serif; }}
            .card {{ transition: all 0.2s; cursor: pointer; }}
            .card:hover {{ transform: translateY(-3px); border-color: #60a5fa; }}
            .modal {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.9); z-index: 50; backdrop-filter: blur(5px); }}
            .modal-content {{ background: #1e293b; margin: 5vh auto; width: 95%; max-width: 600px; max-height: 90vh; overflow-y: auto; border-radius: 12px; border: 1px solid #334155; }}
            
            /* Dot Indicators */
            .dot {{ height: 8px; width: 8px; border-radius: 50%; display: inline-block; margin-right: 2px; }}
            .dot-up {{ background-color: #4ade80; box-shadow: 0 0 5px #4ade80; }}
            .dot-down {{ background-color: #f87171; opacity: 0.4; }}
            
            /* Badges */
            .b-std {{ background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }}
            .b-gold {{ background: rgba(250, 204, 21, 0.15); color: #facc15; border: 1px solid rgba(250, 204, 21, 0.3); }}
            .b-panic {{ background: #9333ea; color: white; border: 1px solid #d8b4fe; box-shadow: 0 0 8px #a855f7; }}
        </style>
    </head>
    <body>
        <div class="max-w-6xl mx-auto p-4">
            <header class="flex justify-between items-center mb-6 border-b border-slate-700 pb-4">
                <div>
                    <h1 class="text-2xl font-bold text-blue-400 flex items-center gap-2">
                        <i data-lucide="layers"></i> Nifty Ultimate
                    </h1>
                    <p class="text-xs text-slate-500 mt-1">Multi-Timeframe + Backtest + Targets</p>
                </div>
                <div class="text-right">
                    <div class="text-xs text-slate-500">Opportunities</div>
                    <div class="text-xl font-bold text-white">{len(results)}</div>
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
                        <div class="text-xs text-slate-400">Deep Dive Analysis</div>
                    </div>
                    <button onclick="closeModal()" class="p-2 bg-slate-700 rounded-full hover:bg-slate-600"><i data-lucide="x"></i></button>
                </div>
                
                <div class="p-6 space-y-6">
                    <!-- Targets -->
                    <div class="grid grid-cols-2 gap-4">
                        <div class="bg-green-900/20 border border-green-500/30 p-3 rounded-lg text-center">
                            <div class="text-xs text-green-400 uppercase font-bold">Target (ATR)</div>
                            <div id="m-target" class="text-2xl font-bold text-white">0</div>
                        </div>
                        <div class="bg-red-900/20 border border-red-500/30 p-3 rounded-lg text-center">
                            <div class="text-xs text-red-400 uppercase font-bold">Stop Loss</div>
                            <div id="m-stop" class="text-2xl font-bold text-white">0</div>
                        </div>
                    </div>

                    <!-- Backtest -->
                    <div>
                        <h3 class="text-sm font-bold text-slate-300 mb-2 flex items-center gap-2"><i data-lucide="history" class="w-4"></i> Backtest (1 Year)</h3>
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
            lucide.createIcons();
            const grid = document.getElementById('grid');
            
            if(data.length === 0) grid.innerHTML = '<div class="col-span-3 text-center text-slate-500 p-10">No signals found.</div>';
            
            data.forEach((s, i) => {{
                // Trends
                const mDot = s.trends.M === 'UP' ? 'dot-up' : 'dot-down';
                const wDot = s.trends.W === 'UP' ? 'dot-up' : 'dot-down';
                const dDot = s.trends.D === 'UP' ? 'dot-up' : 'dot-down';
                
                // Badges
                let badges = '';
                if(s.buzz > 0) badges += `<span class="b-std px-2 py-1 rounded text-[10px] font-bold mr-1 mb-1 inline-block flex items-center gap-1"><i data-lucide="flame" class="w-3"></i> ${{s.buzz}} News</span>`;
                
                s.signals.forEach(sig => {{
                    let cls = 'b-std';
                    if(sig.includes('Golden')) cls = 'b-gold';
                    if(sig.includes('15')) cls = 'b-panic';
                    badges += `<span class="${{cls}} px-2 py-1 rounded text-[10px] font-bold uppercase mr-1 mb-1 inline-block">${{sig}}</span>`;
                }});

                // Sparkline
                const pts = s.history.map((d, j) => {{
                    const min = Math.min(...s.history); const max = Math.max(...s.history);
                    const x = (j / (s.history.length - 1)) * 100;
                    const y = 30 - ((d - min) / (max - min || 1)) * 30;
                    return `${{x}},${{y}}`;
                }}).join(' ');

                grid.innerHTML += `
                    <div class="card bg-slate-800 p-4 rounded-xl border border-slate-700 shadow-lg relative group" onclick="openModal(${{i}})">
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

            function openModal(i) {{
                const s = data[i];
                const bt = s.backtest;
                document.getElementById('m-sym').innerText = s.symbol;
                document.getElementById('m-target').innerText = s.target;
                document.getElementById('m-stop').innerText = s.stop;
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
        </script>
    </body>
    </html>
    """
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    with open(FILE_PATH, "w", encoding="utf-8") as f: f.write(html)
    print("Generated Ultimate Dashboard.")

if __name__ == "__main__":
    print("Starting Master Scan...")
    tickers = get_nifty500_list()
    results = []
    for i, t in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] {t}...", end="\r")
        time.sleep(0.2)
        res = analyze_stock(t)
        if res: results.append(res)
    generate_html(results)

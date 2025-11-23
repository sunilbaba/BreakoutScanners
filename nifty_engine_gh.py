import yfinance as yf
import pandas as pd
import os
import time
import requests
import io
import json
import warnings

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
    
    # Auto-Download
    print("Downloading Nifty 500 list...")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(NSE_URL, headers=headers, timeout=10)
        df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
        return [f"{x}.NS" for x in df['Symbol'].dropna().tolist()]
    except:
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]

# --- 2. EXTREME RSI BACKTESTER ---
def run_backtest(df):
    """
    Strategy: 
    1. EXTREME DIP: RSI < 15 (Panic Buy)
    2. MOMENTUM: MACD Crossover (Standard Buy)
    """
    trades = []
    wins = 0
    losses = 0
    
    if len(df) < 200: return {"win_rate": 0, "total": 0, "log": []}

    # Indicators
    # 1. Trend (SMA 200) - Optional for RSI 15 strategy? 
    # Usually RSI < 15 happens in downtrends, so we might IGNORE trend for this specific signal to catch bounces.
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['In_Uptrend'] = df['Close'] > df['SMA200']
    
    # 2. RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # TRIGGER: RSI Drops below 15 (Extreme)
    # We use .shift(1) >= 15 to ensure we buy exactly when it crosses into the zone
    df['Signal_RSI_15'] = (rsi < 15) & (rsi.shift(1) >= 15)
    
    # 3. MACD (Standard)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['Signal_MACD'] = (macd > signal) & (macd.shift(1) < signal.shift(1))

    # SIMULATION
    in_position = False
    entry_price = 0
    entry_reason = ""
    entry_date = None
    days_held = 0
    
    # Iterate last 1 year
    start_idx = len(df) - 250
    if start_idx < 200: start_idx = 200
    
    for i in range(start_idx, len(df)):
        price = df['Close'].iloc[i]
        date = df.index[i].strftime('%Y-%m-%d')
        
        if in_position:
            days_held += 1
            pct_change = ((price - entry_price) / entry_price) * 100
            
            # EXIT LOGIC
            # For RSI 15, we expect a sharp bounce, so we target higher (7-8%)
            target_pct = 8.0 if "RSI < 15" in entry_reason else 6.0
            stop_pct = -4.0 # Give it room to breathe
            
            result = None
            if pct_change >= target_pct: result = "WIN (Target)"
            elif pct_change <= stop_pct: result = "LOSS (Stop)"
            elif days_held >= 15: result = "WIN (Time)" if pct_change > 0 else "LOSS (Time)"
            
            if result:
                trades.append({
                    "date": entry_date, 
                    "entry": round(entry_price, 2), 
                    "exit": round(price, 2), 
                    "reason": entry_reason,
                    "result": result,
                    "pnl": round(pct_change, 2)
                })
                if pct_change > 0: wins += 1
                else: losses += 1
                in_position = False
        
        else:
            # ENTRY LOGIC
            trigger = []
            
            # STRATEGY 1: EXTREME RSI (Aggressive - Ignores Trend)
            if df['Signal_RSI_15'].iloc[i]: 
                trigger.append("RSI < 15")
            
            # STRATEGY 2: MACD (Conservative - Needs Uptrend)
            if df['Signal_MACD'].iloc[i] and df['In_Uptrend'].iloc[i]:
                trigger.append("MACD")
                
            if len(trigger) > 0:
                in_position = True
                entry_price = price
                entry_date = date
                entry_reason = "+".join(trigger)
                days_held = 0

    total = wins + losses
    win_rate = round((wins / total * 100), 0) if total > 0 else 0
    
    return {
        "win_rate": win_rate,
        "total": total,
        "wins": wins,
        "losses": losses,
        "log": trades[-15:]
    }

# --- 3. ANALYZE STOCK ---
def analyze_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y", interval="1d") 
        if df.empty or len(df) < 300: return None
        
        close = df['Close']
        curr_price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])
        change = round(((curr_price - prev_price) / prev_price) * 100, 2)
        
        # Backtest
        bt = run_backtest(df)
        
        # Trend
        sma200 = close.rolling(200).mean().iloc[-1]
        trend = "UP" if curr_price > sma200 else "DOWN"
        
        # RSI Calculation (Current)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        curr_rsi = float(rsi.iloc[-1])

        # Rating Logic
        rating = "WAIT"
        if curr_rsi < 15: 
            rating = "EXTREME OVERSOLD" # Priority Alert
        elif trend == "UP":
            if bt['win_rate'] >= 60: rating = "HIGH PROB BUY"
            else: rating = "BUY"
        elif trend == "DOWN":
            rating = "SELL"

        return {
            "symbol": ticker.replace(".NS", ""),
            "price": round(curr_price, 2),
            "change": change,
            "rsi": round(curr_rsi, 1),
            "trend": trend,
            "rating": rating,
            "backtest": bt
        }
    except Exception as e:
        return None

# --- 4. GENERATE HTML ---
def generate_html(results):
    json_data = json.dumps(results)
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Nifty RSI Scanner</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            body {{ background-color: #0f172a; color: #cbd5e1; font-family: sans-serif; }}
            .modal {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); z-index: 50; backdrop-filter: blur(4px); }}
            .modal-content {{ background: #1e293b; margin: 5% auto; padding: 0; width: 95%; max-width: 650px; border-radius: 12px; position: relative; max-height: 90vh; overflow-y: auto; border: 1px solid #334155; }}
        </style>
    </head>
    <body>
        <div class="max-w-5xl mx-auto p-4">
            <header class="flex justify-between items-center mb-6 border-b border-slate-700 pb-4">
                <div>
                    <h1 class="text-2xl font-bold text-blue-400 flex items-center gap-2">
                        <i data-lucide="zap" class="text-yellow-400"></i> Nifty Extreme RSI
                    </h1>
                    <p class="text-xs text-slate-500 mt-1">Alerts: RSI < 15 (Panic) | Strategy: Buy Dip & Trend</p>
                </div>
                <div class="text-right">
                    <div class="text-xs text-slate-500">Total Scanned</div>
                    <div class="text-xl font-bold text-white">{len(results)}</div>
                </div>
            </header>
            
            <div class="overflow-x-auto rounded-lg border border-slate-700 bg-slate-800/50">
                <table class="w-full text-left text-sm text-slate-400">
                    <thead class="bg-slate-900 text-xs uppercase text-slate-200 font-bold tracking-wider">
                        <tr>
                            <th class="px-4 py-3">Stock</th>
                            <th class="px-4 py-3 text-right">Price</th>
                            <th class="px-4 py-3 text-center">RSI (14)</th>
                            <th class="px-4 py-3 text-center">Signal</th>
                            <th class="px-4 py-3 text-center">Backtest Win%</th>
                            <th class="px-4 py-3 text-center">Analyze</th>
                        </tr>
                    </thead>
                    <tbody id="stock-table" class="divide-y divide-slate-700"></tbody>
                </table>
            </div>
        </div>

        <!-- MODAL -->
        <div id="modal" class="modal">
            <div class="modal-content">
                <div class="bg-slate-900 p-4 border-b border-slate-700 flex justify-between items-center sticky top-0">
                    <div>
                        <h2 id="m-symbol" class="text-2xl font-bold text-white">SYMBOL</h2>
                        <span class="text-xs text-slate-400">Strategy Performance Report</span>
                    </div>
                    <button onclick="closeModal()" class="bg-slate-800 hover:bg-slate-700 p-2 rounded-full"><i data-lucide="x" class="w-5 h-5 text-white"></i></button>
                </div>
                <div class="p-6">
                    <div class="grid grid-cols-3 gap-4 mb-6">
                        <div class="bg-slate-800 p-4 rounded-xl border border-slate-700 text-center">
                            <div class="text-xs text-slate-500 uppercase">Trades</div>
                            <div id="m-total" class="text-2xl font-bold text-white mt-1">0</div>
                        </div>
                        <div class="bg-slate-800 p-4 rounded-xl border border-slate-700 text-center">
                            <div class="text-xs text-slate-500 uppercase">Win Rate</div>
                            <div id="m-rate" class="text-2xl font-bold text-white mt-1">0%</div>
                        </div>
                        <div class="bg-slate-800 p-4 rounded-xl border border-slate-700 text-center">
                            <div class="text-xs text-slate-500 uppercase">Wins</div>
                            <div id="m-wins" class="text-2xl font-bold text-green-400 mt-1">0</div>
                        </div>
                    </div>
                    <div class="overflow-hidden rounded-lg border border-slate-700">
                        <table class="w-full text-xs">
                            <thead class="bg-slate-900 text-slate-300 font-bold">
                                <tr><th class="p-3 text-left">Date</th><th class="p-3 text-left">Reason</th><th class="p-3 text-right">Entry</th><th class="p-3 text-right">Result</th></tr>
                            </thead>
                            <tbody id="m-logs" class="bg-slate-800 text-slate-300 divide-y divide-slate-700/50"></tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const data = {json_data};
            lucide.createIcons();

            function renderTable() {{
                const tbody = document.getElementById('stock-table');
                let html = '';
                
                data.forEach((stock, index) => {{
                    // RSI Color Logic
                    let rsiClass = 'text-slate-300';
                    let rsiBg = '';
                    if(stock.rsi < 30) {{ rsiClass = 'text-red-300 font-bold'; }}
                    if(stock.rsi < 15) {{ rsiClass = 'text-white font-bold'; rsiBg = 'bg-purple-600 px-2 py-1 rounded shadow-lg shadow-purple-500/50'; }}
                    
                    // Rating Badge
                    let ratingBadge = `<span class="bg-slate-700 px-2 py-1 rounded text-xs">${{stock.rating}}</span>`;
                    if(stock.rating.includes('EXTREME')) ratingBadge = '<span class="bg-purple-600 text-white px-2 py-1 rounded text-xs font-bold animate-pulse">EXTREME OVERSOLD</span>';
                    else if(stock.rating.includes('HIGH')) ratingBadge = '<span class="bg-green-500/20 text-green-400 px-2 py-1 rounded text-xs font-bold border border-green-500/30">HIGH PROB</span>';

                    html += `
                        <tr class="hover:bg-slate-700/50 transition">
                            <td class="px-4 py-3 font-bold text-white">${{stock.symbol}}</td>
                            <td class="px-4 py-3 text-right">
                                <div class="font-mono text-slate-200">${{stock.price}}</div>
                                <div class="text-xs ${{stock.change >= 0 ? 'text-green-400' : 'text-red-400'}}">${{stock.change}}%</div>
                            </td>
                            <td class="px-4 py-3 text-center">
                                <span class="${{rsiClass}} ${{rsiBg}}">${{stock.rsi}}</span>
                            </td>
                            <td class="px-4 py-3 text-center">${{ratingBadge}}</td>
                            <td class="px-4 py-3 text-center font-mono ${{stock.backtest.win_rate >= 60 ? 'text-green-400' : 'text-slate-400'}}">${{stock.backtest.win_rate}}%</td>
                            <td class="px-4 py-3 text-center">
                                <button onclick="openModal(${{index}})" class="text-blue-400 hover:text-white hover:bg-blue-600 p-2 rounded transition"><i data-lucide="bar-chart-2" class="w-5 h-5"></i></button>
                            </td>
                        </tr>
                    `;
                }});
                tbody.innerHTML = html;
                lucide.createIcons();
            }}

            function openModal(index) {{
                const stock = data[index];
                const bt = stock.backtest;
                document.getElementById('m-symbol').innerText = stock.symbol;
                document.getElementById('m-total').innerText = bt.total;
                document.getElementById('m-rate').innerText = bt.win_rate + '%';
                document.getElementById('m-wins').innerText = bt.wins;
                
                const logsBody = document.getElementById('m-logs');
                let logHtml = '';
                if (bt.log.length === 0) logHtml = '<tr><td colspan="4" class="p-4 text-center italic text-slate-500">No signals in last 12 months.</td></tr>';
                else {{
                    [...bt.log].reverse().forEach(log => {{
                        let resClass = log.result.includes('WIN') ? 'text-green-400 font-bold' : 'text-red-400';
                        logHtml += `<tr><td class="p-3">${{log.date}}</td><td class="p-3 font-bold text-xs uppercase text-blue-300">${{log.reason}}</td><td class="p-3 text-right font-mono">${{log.entry}}</td><td class="p-3 text-right font-mono ${{resClass}}">${{log.result}} (${{log.pnl}}%)</td></tr>`;
                    }});
                }}
                logsBody.innerHTML = logHtml;
                document.getElementById('modal').style.display = 'block';
            }}
            function closeModal() {{ document.getElementById('modal').style.display = 'none'; }}
            window.onclick = function(e) {{ if(e.target == document.getElementById('modal')) closeModal(); }}
            renderTable();
        </script>
    </body>
    </html>
    """
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    with open(FILE_PATH, "w", encoding="utf-8") as f: f.write(html)
    print("Generated Extreme RSI Dashboard.")

if __name__ == "__main__":
    print("Starting Extreme RSI Scan...")
    tickers = get_nifty500_list()
    results = []
    for i, t in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] {t}...", end="\r")
        time.sleep(0.1) 
        res = analyze_stock(t)
        if res: results.append(res)
    generate_html(results)

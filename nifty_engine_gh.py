import yfinance as yf
import pandas as pd
import os
import time
import math
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
    
    # Auto-Download if missing
    print("Downloading Nifty 500 list...")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(NSE_URL, headers=headers, timeout=10)
        df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
        return [f"{x}.NS" for x in df['Symbol'].dropna().tolist()]
    except:
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]

# --- 2. BACKTEST ENGINE ---
def run_backtest(df):
    """
    Simulates trading the strategy over the past 1 year data.
    Strategy: BUY when Trend is UP and (MACD Buy OR RSI Dip).
    Exit: Take Profit +5% or Stop Loss -3% or Max Hold 20 Days.
    """
    trades = []
    wins = 0
    losses = 0
    
    # We need at least 50 days of data
    if len(df) < 50: return {"win_rate": 0, "total": 0, "log": []}

    # Pre-calculate Logic columns to speed up loop
    # Trend (SMA 200)
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['Trend'] = df['Close'] > df['SMA200']
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_Buy'] = (macd > signal) & (macd.shift(1) < signal.shift(1)) # Crossover
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI_Buy'] = (rsi < 40) & (rsi.shift(1) >= 40) # Dip below 40
    
    # SIMULATION LOOP
    in_position = False
    entry_price = 0
    entry_date = None
    days_held = 0
    
    # Iterate through history (skip first 200 days for SMA)
    for i in range(200, len(df)):
        price = df['Close'].iloc[i]
        date = df.index[i].strftime('%Y-%m-%d')
        
        if in_position:
            days_held += 1
            pct_change = ((price - entry_price) / entry_price) * 100
            
            # EXIT CONDITIONS
            result = None
            if pct_change >= 5.0: result = "WIN (Target)"
            elif pct_change <= -3.0: result = "LOSS (Stop)"
            elif days_held >= 20: result = "TIMEOUT" # Exit after 20 days regardless
            
            if result:
                trades.append({
                    "date": entry_date, 
                    "entry": round(entry_price, 2), 
                    "exit": round(price, 2), 
                    "days": days_held,
                    "result": result,
                    "pnl": round(pct_change, 2)
                })
                if pct_change > 0: wins += 1
                else: losses += 1
                in_position = False
        
        else:
            # ENTRY CONDITIONS (Trend UP + Signal)
            if df['Trend'].iloc[i]:
                if df['MACD_Buy'].iloc[i] or df['RSI_Buy'].iloc[i]:
                    in_position = True
                    entry_price = price
                    entry_date = date
                    days_held = 0

    total = wins + losses
    win_rate = round((wins / total * 100), 0) if total > 0 else 0
    
    return {
        "win_rate": win_rate,
        "total": total,
        "wins": wins,
        "losses": losses,
        "log": trades[-10:] # Keep only last 10 trades to save space
    }

# --- 3. ANALYZE STOCK ---
def analyze_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y", interval="1d") # Need 2y for valid 1y backtest (200d SMA buffer)
        
        if df.empty or len(df) < 250: return None
        
        # --- CURRENT STATUS ---
        close = df['Close']
        curr_price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])
        change = round(((curr_price - prev_price) / prev_price) * 100, 2)
        
        # Run Backtest
        backtest_results = run_backtest(df)
        
        # Trend
        sma200 = close.rolling(200).mean().iloc[-1]
        trend = "UP" if curr_price > sma200 else "DOWN"
        
        # Recommendations
        rec = "WAIT"
        if trend == "UP":
            if backtest_results['win_rate'] > 60: rec = "HIGH PROB BUY"
            else: rec = "BUY"
        elif trend == "DOWN":
            rec = "SELL"

        return {
            "symbol": ticker.replace(".NS", ""),
            "price": round(curr_price, 2),
            "change": change,
            "trend": trend,
            "rec": rec,
            "backtest": backtest_results
        }
    except Exception as e:
        return None

# --- 4. GENERATE HTML (With Clickable Modals) ---
def generate_html(results):
    # Convert data to JSON for JavaScript to use
    json_data = json.dumps(results)
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Nifty Backtest Scanner</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body {{ background-color: #0f172a; color: #e2e8f0; }}
            .modal {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 50; }}
            .modal-content {{ background: #1e293b; margin: 10% auto; padding: 20px; width: 90%; max-width: 600px; border-radius: 8px; position: relative; max-height: 80vh; overflow-y: auto; }}
        </style>
    </head>
    <body>
        <div class="max-w-4xl mx-auto p-4">
            <h1 class="text-3xl font-bold text-blue-400 mb-2">Nifty 500 Strategy Backtester</h1>
            <p class="text-sm text-slate-400 mb-6">Strategy: Trend Following (Buy on Dip/Momentum in Uptrend). Exit: +5% Profit or -3% Stop.</p>
            
            <div class="overflow-x-auto rounded-lg border border-slate-700">
                <table class="w-full text-left text-sm text-slate-400">
                    <thead class="bg-slate-800 text-xs uppercase text-slate-200">
                        <tr>
                            <th class="px-4 py-3">Symbol</th>
                            <th class="px-4 py-3">Price</th>
                            <th class="px-4 py-3">Trend</th>
                            <th class="px-4 py-3">Signal</th>
                            <th class="px-4 py-3">Win Rate (1Y)</th>
                            <th class="px-4 py-3">Action</th>
                        </tr>
                    </thead>
                    <tbody id="stock-table">
                        <!-- JS Will Inject Rows Here -->
                    </tbody>
                </table>
            </div>
        </div>

        <!-- MODAL -->
        <div id="modal" class="modal">
            <div class="modal-content border border-slate-600 shadow-2xl">
                <button onclick="closeModal()" class="absolute top-4 right-4 text-slate-400 hover:text-white">✕</button>
                <h2 id="m-symbol" class="text-2xl font-bold text-white mb-1">RELIANCE</h2>
                <div class="text-xs text-slate-400 mb-4">Historical Performance Report (Last 12 Months)</div>
                
                <div class="grid grid-cols-3 gap-4 mb-6">
                    <div class="bg-slate-800 p-3 rounded border border-slate-700 text-center">
                        <div class="text-xs text-slate-500">Total Trades</div>
                        <div id="m-total" class="text-xl font-bold text-white">0</div>
                    </div>
                    <div class="bg-slate-800 p-3 rounded border border-slate-700 text-center">
                        <div class="text-xs text-slate-500">Win Rate</div>
                        <div id="m-rate" class="text-xl font-bold text-green-400">0%</div>
                    </div>
                    <div class="bg-slate-800 p-3 rounded border border-slate-700 text-center">
                        <div class="text-xs text-slate-500">Net Logic</div>
                        <div class="text-sm font-bold text-blue-400">Trend + Mom</div>
                    </div>
                </div>

                <h3 class="text-sm font-bold text-white mb-2">Recent Trade Logs</h3>
                <div class="overflow-hidden rounded border border-slate-700">
                    <table class="w-full text-xs">
                        <thead class="bg-slate-900 text-slate-300">
                            <tr>
                                <th class="p-2">Date</th>
                                <th class="p-2">Entry</th>
                                <th class="p-2">Result</th>
                                <th class="p-2 text-right">P&L</th>
                            </tr>
                        </thead>
                        <tbody id="m-logs" class="bg-slate-800 text-slate-300 divide-y divide-slate-700">
                            <!-- Logs go here -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <script>
            const data = {json_data};

            function renderTable() {{
                const tbody = document.getElementById('stock-table');
                let html = '';
                
                data.forEach((stock, index) => {{
                    const winColor = stock.backtest.win_rate >= 60 ? 'text-green-400' : (stock.backtest.win_rate < 40 ? 'text-red-400' : 'text-slate-300');
                    const trendColor = stock.trend === 'UP' ? 'text-green-500' : 'text-red-500';
                    const changeColor = stock.change >= 0 ? 'text-green-400' : 'text-red-400';
                    
                    html += `
                        <tr class="border-b border-slate-800 hover:bg-slate-800/50 transition">
                            <td class="px-4 py-3 font-bold text-white">${{stock.symbol}}</td>
                            <td class="px-4 py-3">
                                <div>${{stock.price}}</div>
                                <div class="text-xs ${{changeColor}}">${{stock.change}}%</div>
                            </td>
                            <td class="px-4 py-3 font-bold text-xs ${{trendColor}}">${{stock.trend}}</td>
                            <td class="px-4 py-3 text-xs"><span class="bg-slate-700 px-2 py-1 rounded text-white">${{stock.rec}}</span></td>
                            <td class="px-4 py-3 font-bold ${{winColor}}">${{stock.backtest.win_rate}}%</td>
                            <td class="px-4 py-3">
                                <button onclick="openModal(${{index}})" class="bg-blue-600 hover:bg-blue-500 text-white text-xs px-3 py-1.5 rounded font-bold shadow-lg transition">
                                    Analyze
                                </button>
                            </td>
                        </tr>
                    `;
                }});
                tbody.innerHTML = html;
            }}

            function openModal(index) {{
                const stock = data[index];
                const bt = stock.backtest;
                
                document.getElementById('m-symbol').innerText = stock.symbol;
                document.getElementById('m-total').innerText = bt.total;
                document.getElementById('m-rate').innerText = bt.win_rate + '%';
                
                // Populate Logs
                const logsBody = document.getElementById('m-logs');
                let logHtml = '';
                
                if (bt.log.length === 0) {{
                    logHtml = '<tr><td colspan="4" class="p-3 text-center text-slate-500">No signals generated in last 12 months.</td></tr>';
                }} else {{
                    // Show reverse (newest first)
                    [...bt.log].reverse().forEach(log => {{
                        const resColor = log.result.includes('WIN') ? 'text-green-400' : (log.result.includes('LOSS') ? 'text-red-400' : 'text-yellow-400');
                        logHtml += `
                            <tr>
                                <td class="p-2">${{log.date}}</td>
                                <td class="p-2">${{log.entry}}</td>
                                <td class="p-2 font-bold ${{resColor}}">${{log.result}}</td>
                                <td class="p-2 text-right ${{resColor}}">${{log.pnl}}%</td>
                            </tr>
                        `;
                    }});
                }}
                
                logsBody.innerHTML = logHtml;
                document.getElementById('modal').style.display = 'block';
            }}

            function closeModal() {{
                document.getElementById('modal').style.display = 'none';
            }}

            // Close on outside click
            window.onclick = function(event) {{
                const modal = document.getElementById('modal');
                if (event.target == modal) {{
                    modal.style.display = 'none';
                }}
            }}

            renderTable();
        </script>
    </body>
    </html>
    """
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    with open(FILE_PATH, "w", encoding="utf-8") as f: f.write(html)
    print(f"Generated Interactive Dashboard with Backtesting.")

if __name__ == "__main__":
    print("Starting Backtest Scan...")
    tickers = get_nifty500_list()
    
    results = []
    # Using enumerate to track progress
    for i, t in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] Backtesting {t}...", end="\r")
        time.sleep(0.1) # Prevent blocking
        res = analyze_stock(t)
        if res: results.append(res)
        
    generate_html(results)

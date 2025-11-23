import yfinance as yf
import pandas as pd
import os
import time
import requests
import io
import json
import warnings
import math

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

# --- 2. MULTI-SCENARIO ANALYSIS ---
def analyze_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Fetch 2 years to calculate 52-week high correctly
        df = stock.history(period="2y", interval="1d")
        
        if df.empty or len(df) < 260: return None

        # Current & Previous Data Points
        close = df['Close']
        open_price = df['Open']
        high = df['High']
        volume = df['Volume']
        
        curr_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        curr_open = float(open_price.iloc[-1])
        prev_open = float(open_price.iloc[-2])
        
        change = round(((curr_close - prev_close) / prev_close) * 100, 2)
        
        # --- INDICATOR CALCULATIONS ---
        
        # SMA 50 & 200
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        curr_sma50 = float(sma50.iloc[-1])
        prev_sma50 = float(sma50.iloc[-2])
        curr_sma200 = float(sma200.iloc[-1])
        prev_sma200 = float(sma200.iloc[-2])

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        curr_rsi = float(rsi.iloc[-1])
        prev_rsi = float(rsi.iloc[-2])
        
        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = sma20 + (2 * std20)
        curr_upper = float(upper.iloc[-1])
        prev_upper = float(upper.iloc[-2])
        
        # Volume Average
        vol_avg = float(volume.rolling(20).mean().iloc[-1])
        curr_vol = float(volume.iloc[-1])
        
        # 52-Week High (Max of last 252 days EXCLUDING today)
        high_52w = float(high.iloc[-253:-1].max())

        # --- FRESH BREAKOUT LOGIC (Triggers) ---
        signals = []
        
        # 1. GOLDEN CROSS (50 crosses 200)
        if prev_sma50 < prev_sma200 and curr_sma50 > curr_sma200:
            signals.append("Golden Cross")
            
        # 2. 52-WEEK HIGH BREAKOUT
        if prev_close < high_52w and curr_close > high_52w:
            signals.append("52-Wk High")
            
        # 3. VOLUME SHOCK (>3x Avg)
        if curr_vol > (vol_avg * 3):
            signals.append("Vol Shock (3x)")
            
        # 4. BULLISH ENGULFING (Candlestick Pattern)
        # Prev was Red, Curr is Green, Curr Body engulfs Prev Body
        if (prev_close < prev_open) and (curr_close > curr_open):
            if (curr_open < prev_close) and (curr_close > prev_open):
                signals.append("Bull Engulfing")
                
        # 5. RSI MOMENTUM SHIFT (Crosses 60)
        if prev_rsi < 60 and curr_rsi > 60:
            signals.append("RSI > 60")
            
        # 6. BOLLINGER BLAST (Close outside Upper Band)
        if prev_close < prev_upper and curr_close > curr_upper:
            signals.append("BB Blast")
            
        # 7. GAP UP & GO (>1% Gap and held green)
        if (curr_open > prev_close * 1.01) and (curr_close > curr_open):
            signals.append("Gap Up")

        # FILTER: Return only if a signal exists
        if len(signals) == 0:
            return None

        return {
            "symbol": ticker.replace(".NS", ""),
            "price": round(curr_close, 2),
            "change": change,
            "rsi": round(curr_rsi, 1),
            "vol_mult": round(curr_vol / vol_avg, 1) if vol_avg > 0 else 0,
            "signals": signals,
            "history": [x if not math.isnan(x) else 0 for x in close.tail(30).tolist()]
        }

    except Exception as e:
        return None

# --- 3. GENERATE DASHBOARD ---
def generate_html(results):
    json_data = json.dumps(results)
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Nifty Multi-Breakout</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            body {{ background-color: #0f172a; color: #e2e8f0; font-family: sans-serif; }}
            .badge-gold {{ background: rgba(250, 204, 21, 0.15); color: #facc15; border: 1px solid rgba(250, 204, 21, 0.3); }}
            .badge-purple {{ background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }}
            .badge-blue {{ background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }}
            .badge-green {{ background: rgba(74, 222, 128, 0.15); color: #4ade80; border: 1px solid rgba(74, 222, 128, 0.3); }}
            .badge-pink {{ background: rgba(236, 72, 153, 0.15); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.3); }}
        </style>
    </head>
    <body>
        <div class="max-w-6xl mx-auto p-4">
            <header class="flex justify-between items-center mb-6 border-b border-slate-700 pb-4">
                <div>
                    <h1 class="text-2xl font-bold text-blue-400 flex items-center gap-2">
                        <i data-lucide="layers"></i> Nifty Multi-Breakout
                    </h1>
                    <p class="text-xs text-slate-500 mt-1">Scenarios: Golden Cross | 52W High | Vol Shock | Engulfing | RSI 60 | Gap Up</p>
                </div>
                <div class="text-right">
                    <div class="text-xs text-slate-500">Opportunities</div>
                    <div class="text-xl font-bold text-white">{len(results)}</div>
                </div>
            </header>
            
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4" id="stock-grid">
                <!-- JS Injected -->
            </div>
        </div>

        <script>
            const data = {json_data};
            lucide.createIcons();
            const grid = document.getElementById('stock-grid');
            
            if(data.length === 0) {{
                grid.innerHTML = '<div class="col-span-3 text-center text-slate-500 p-10">No fresh breakouts detected today across any scenario.</div>';
            }} else {{
                data.forEach(stock => {{
                    let badges = '';
                    stock.signals.forEach(s => {{
                        let cls = 'badge-blue';
                        if(s.includes('Golden')) cls = 'badge-gold';
                        if(s.includes('52-Wk')) cls = 'badge-green';
                        if(s.includes('Vol')) cls = 'badge-pink';
                        if(s.includes('Engulfing')) cls = 'badge-purple';
                        badges += `<span class="${{cls}} px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wider mr-1 mb-1 inline-block">${{s}}</span>`;
                    }});

                    // Sparkline
                    const pts = stock.history.map((d, i) => {{
                        const min = Math.min(...stock.history);
                        const max = Math.max(...stock.history);
                        const x = (i / (stock.history.length - 1)) * 100;
                        const y = 40 - ((d - min) / (max - min || 1)) * 40;
                        return `${{x}},${{y}}`;
                    }}).join(' ');

                    const html = `
                        <div class="bg-slate-800 p-4 rounded-xl border border-slate-700 shadow-lg hover:bg-slate-750 hover:border-slate-600 transition group">
                            <div class="flex justify-between items-start mb-3">
                                <div>
                                    <div class="font-bold text-lg text-white group-hover:text-blue-400 transition">${{stock.symbol}}</div>
                                    <div class="flex items-center gap-2 mt-1">
                                        <span class="text-slate-400 font-mono text-sm">₹${{stock.price}}</span>
                                        <span class="text-xs font-bold ${{stock.change >= 0 ? 'text-green-400' : 'text-red-400'}}">${{stock.change}}%</span>
                                    </div>
                                </div>
                                <div class="text-right">
                                    <div class="text-[10px] text-slate-500 uppercase font-bold">Vol Mult</div>
                                    <div class="text-sm font-mono text-pink-400">${{stock.vol_mult}}x</div>
                                </div>
                            </div>
                            
                            <div class="flex flex-wrap gap-y-1 mb-4 h-12 content-start">
                                ${{badges}}
                            </div>
                            
                            <div class="relative h-10 w-full opacity-70 group-hover:opacity-100 transition">
                                <svg width="100%" height="100%" preserveAspectRatio="none" class="overflow-visible">
                                    <polyline points="${{pts}}" fill="none" stroke="${{stock.change >= 0 ? '#4ade80' : '#f87171'}}" stroke-width="2" vector-effect="non-scaling-stroke" />
                                </svg>
                            </div>
                        </div>
                    `;
                    grid.innerHTML += html;
                }});
            }}
        </script>
    </body>
    </html>
    """
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    with open(FILE_PATH, "w", encoding="utf-8") as f: f.write(html)
    print(f"Generated Multi-Breakout Dashboard with {len(results)} stocks.")

if __name__ == "__main__":
    print("Starting Multi-Scenario Scan...")
    tickers = get_nifty500_list()
    
    results = []
    for i, t in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] {t}...", end="\r")
        time.sleep(0.1) 
        res = analyze_stock(t)
        if res: results.append(res)
        
    generate_html(results)

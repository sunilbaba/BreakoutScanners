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

# --- 2. HELPER: CALCULATE TRENDS ---
def get_trend_data(df, period_name):
    """
    Calculates Trend (EMA20) and Momentum (MACD) for a given timeframe dataframe.
    """
    if len(df) < 26: return "NEUTRAL", False # Not enough data
    
    # EMA 20 (Trend Filter)
    ema20 = df['Close'].ewm(span=20, adjust=False).mean()
    curr_price = df['Close'].iloc[-1]
    curr_ema = ema20.iloc[-1]
    prev_price = df['Close'].iloc[-2]
    prev_ema = ema20.iloc[-2]
    
    # Trend Status
    trend = "UP" if curr_price > curr_ema else "DOWN"
    
    # Check for Fresh Breakout (Price crossed EMA20 just now)
    breakout = (prev_price < prev_ema) and (curr_price > curr_ema)
    
    return trend, breakout

# --- 3. MULTI-TIMEFRAME ENGINE ---
def analyze_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Fetch 5 years to build valid Weekly/Monthly charts
        df_d = stock.history(period="5y", interval="1d")
        
        if df_d.empty or len(df_d) < 200: return None

        # --- A. RESAMPLE DATA (Create W and M Charts) ---
        agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
        
        # Weekly Chart
        df_w = df_d.resample('W').agg(agg_dict).dropna()
        # Monthly Chart
        df_m = df_d.resample('ME').agg(agg_dict).dropna() # 'ME' is Month End

        # --- B. ANALYZE TRENDS ---
        # 1. Monthly (Long Term)
        m_trend, m_breakout = get_trend_data(df_m, "Monthly")
        
        # 2. Weekly (Intermediate)
        w_trend, w_breakout = get_trend_data(df_w, "Weekly")
        
        # 3. Daily (Short Term Entry)
        curr_price = df_d['Close'].iloc[-1]
        prev_price = df_d['Close'].iloc[-2]
        change = round(((curr_price - prev_price) / prev_price) * 100, 2)
        
        # --- C. DAILY SIGNALS (Existing Mechanism) ---
        signals = []
        
        # RSI
        delta = df_d['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        curr_rsi = float(rsi.iloc[-1])
        
        if curr_rsi < 30: signals.append("RSI Oversold")
        if curr_rsi > 70: signals.append("Overbought")
        
        # Bollinger Blast (Volatility Breakout)
        sma20 = df_d['Close'].rolling(20).mean()
        std20 = df_d['Close'].rolling(20).std()
        upper = sma20 + (2 * std20)
        if curr_price > upper.iloc[-1] and prev_price < upper.iloc[-2]:
            signals.append("BB Blast")
            
        # Engulfing (Candlestick)
        open_p = df_d['Open']
        close_p = df_d['Close']
        if (close_p.iloc[-2] < open_p.iloc[-2]) and (close_p.iloc[-1] > open_p.iloc[-1]): # Red then Green
            if (open_p.iloc[-1] < close_p.iloc[-2]) and (close_p.iloc[-1] > open_p.iloc[-2]): # Engulfs
                signals.append("Bull Engulfing")

        # --- D. ADD TIMEFRAME SIGNALS ---
        if w_breakout: signals.append("W-Breakout")
        if m_breakout: signals.append("M-Breakout")

        # --- E. ALIGNMENT SCORE (Confluence) ---
        # 3/3 = Perfect alignment (All Up)
        # 0/3 = Avoid (All Down)
        alignment = 0
        if m_trend == "UP": alignment += 1
        if w_trend == "UP": alignment += 1
        # Daily trend check (Price > SMA50)
        sma50 = df_d['Close'].rolling(50).mean().iloc[-1]
        if curr_price > sma50: alignment += 1
        
        # FILTER: 
        # Show if there is a Signal OR if it's a Weekly/Monthly Breakout
        if not signals and not w_breakout and not m_breakout: return None

        return {
            "symbol": ticker.replace(".NS", ""),
            "price": round(curr_price, 2),
            "change": change,
            "m_trend": m_trend,
            "w_trend": w_trend,
            "signals": signals,
            "score": f"{alignment}/3",
            "rsi": round(curr_rsi, 1),
            "history": [x if not math.isnan(x) else 0 for x in df_d['Close'].tail(30).tolist()]
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
        <title>Nifty Multi-Timeframe</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            body {{ background-color: #0f172a; color: #e2e8f0; font-family: sans-serif; }}
            .card {{ transition: all 0.2s; }}
            .card:hover {{ transform: translateY(-3px); border-color: #60a5fa; }}
            
            /* Trend Dots */
            .dot {{ height: 8px; width: 8px; border-radius: 50%; display: inline-block; margin-right: 4px; }}
            .dot-up {{ background-color: #4ade80; box-shadow: 0 0 5px #4ade80; }}
            .dot-down {{ background-color: #f87171; opacity: 0.5; }}
            
            /* Badges */
            .badge {{ font-size: 10px; font-weight: bold; padding: 2px 6px; border-radius: 4px; border: 1px solid; display: inline-block; margin-right: 4px; margin-bottom: 4px; }}
            .b-gold {{ background: rgba(250, 204, 21, 0.1); color: #facc15; border-color: rgba(250, 204, 21, 0.3); }}
            .b-purple {{ background: rgba(192, 132, 252, 0.1); color: #c084fc; border-color: rgba(192, 132, 252, 0.3); }}
            .b-blue {{ background: rgba(96, 165, 250, 0.1); color: #60a5fa; border-color: rgba(96, 165, 250, 0.3); }}
        </style>
    </head>
    <body>
        <div class="max-w-6xl mx-auto p-4">
            <header class="flex justify-between items-center mb-6 border-b border-slate-700 pb-4">
                <div>
                    <h1 class="text-2xl font-bold text-blue-400 flex items-center gap-2">
                        <i data-lucide="bar-chart-2"></i> Nifty Triple-Trend
                    </h1>
                    <p class="text-xs text-slate-500 mt-1">Monthly + Weekly + Daily Confluence Scanner</p>
                </div>
                <div class="text-right">
                    <div class="text-xs text-slate-500">Stocks</div>
                    <div class="text-xl font-bold text-white">{len(results)}</div>
                </div>
            </header>
            
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4" id="grid"></div>
        </div>

        <script>
            const data = {json_data};
            lucide.createIcons();
            const grid = document.getElementById('grid');
            
            if(data.length === 0) grid.innerHTML = '<div class="col-span-3 text-center text-slate-500 p-10">No significant setups found today.</div>';
            
            data.forEach(s => {{
                // Trend Visualization (M | W | D)
                const mClass = s.m_trend === 'UP' ? 'dot-up' : 'dot-down';
                const wClass = s.w_trend === 'UP' ? 'dot-up' : 'dot-down';
                
                // Badges
                let badges = '';
                s.signals.forEach(sig => {{
                    let c = 'b-blue';
                    if(sig.includes('M-Breakout')) c = 'b-gold'; // Rare
                    if(sig.includes('W-Breakout')) c = 'b-purple'; // Strong
                    badges += `<span class="badge ${{c}}">${{sig}}</span>`;
                }});

                // Alignment Score Color
                let scoreColor = 'text-slate-500';
                if(s.score === '3/3') scoreColor = 'text-green-400 font-bold'; // Jackpot
                if(s.score === '0/3') scoreColor = 'text-red-500';

                // Sparkline
                const pts = s.history.map((d, i) => {{
                    const min = Math.min(...s.history); const max = Math.max(...s.history);
                    const x = (i / (s.history.length - 1)) * 100;
                    const y = 35 - ((d - min) / (max - min || 1)) * 35;
                    return `${{x}},${{y}}`;
                }}).join(' ');

                grid.innerHTML += `
                    <div class="card bg-slate-800 p-4 rounded-xl border border-slate-700 shadow-lg">
                        <div class="flex justify-between items-start mb-2">
                            <div>
                                <div class="font-bold text-lg text-white">${{s.symbol}}</div>
                                <div class="flex items-center gap-2 mt-0.5">
                                    <span class="text-slate-400 font-mono text-sm">₹${{s.price}}</span>
                                    <span class="text-xs font-bold ${{s.change >= 0 ? 'text-green-400' : 'text-red-400'}}">${{s.change}}%</span>
                                </div>
                            </div>
                            <div class="text-right">
                                <div class="text-[10px] text-slate-500 uppercase">M / W Trend</div>
                                <div class="flex items-center justify-end gap-1 mt-1">
                                    <span class="text-[10px] text-slate-400">M</span><span class="dot ${{mClass}}"></span>
                                    <span class="text-[10px] text-slate-400 ml-1">W</span><span class="dot ${{wClass}}"></span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="flex flex-wrap gap-1 mb-3 min-h-[24px] content-start">
                            ${{badges}}
                        </div>
                        
                        <div class="flex justify-between items-end">
                            <div class="text-xs text-slate-500">Score: <span class="${{scoreColor}}">${{s.score}}</span></div>
                            <div class="h-9 w-24 opacity-60">
                                <svg width="100%" height="100%" preserveAspectRatio="none" class="overflow-visible">
                                    <polyline points="${{pts}}" fill="none" stroke="${{s.change >= 0 ? '#4ade80' : '#f87171'}}" stroke-width="2" />
                                </svg>
                            </div>
                        </div>
                    </div>
                `;
            }});
        </script>
    </body>
    </html>
    """
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    with open(FILE_PATH, "w", encoding="utf-8") as f: f.write(html)
    print(f"Generated Multi-Timeframe Dashboard with {len(results)} stocks.")

if __name__ == "__main__":
    print("Starting Triple-Trend Scan...")
    tickers = get_nifty500_list()
    
    results = []
    for i, t in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] {t}...", end="\r")
        # Slightly longer sleep to handle the heavy 5-year data fetch
        time.sleep(0.25) 
        res = analyze_stock(t)
        if res: results.append(res)
        
    generate_html(results)

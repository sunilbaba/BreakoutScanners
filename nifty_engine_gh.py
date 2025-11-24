import yfinance as yf
import pandas as pd
import os
import time
import requests
import io
import json
import warnings
import math
import numpy as np

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

# --- 3. HELPER: PIVOTS & CANDLES ---
def get_pivots(high, low, close):
    p = (high + low + close) / 3
    r1 = (2 * p) - low
    s1 = (2 * p) - high
    return round(r1, 2), round(s1, 2), round(p, 2)

def is_hammer(open_p, high, low, close):
    body = abs(close - open_p)
    lower = min(open_p, close) - low
    upper = high - max(open_p, close)
    return (lower > 2 * body) and (upper < body)

# --- 4. DECISION LOGIC (THE BRAIN) ---
def calculate_verdict(trend_score, signals, is_fno, win_rate, rs_score):
    """
    Synthesizes all technical data into a single human-readable decision.
    """
    score = 0
    reasons = []
    
    # 1. Trend Weight (Max 40 pts)
    if trend_score == 3: 
        score += 40
        reasons.append("Market structure is Bullish (M/W/D).")
    elif trend_score == 2:
        score += 25
        reasons.append("Primary trend is Up.")
    else:
        score -= 20
        reasons.append("Trend is weak/down.")

    # 2. Institutional Weight (Max 20 pts)
    if is_fno:
        score += 20
        reasons.append("High Liquidity (F&O Stock).")
    
    # 3. Signal Strength (Max 20 pts)
    if "Golden Cross" in signals or "Bull Divergence" in signals:
        score += 20
        reasons.append("Strong institutional signal detected.")
    elif len(signals) > 0:
        score += 10
        reasons.append("Breakout signal present.")

    # 4. Probability Weight (Max 20 pts)
    if win_rate > 60:
        score += 20
        reasons.append(f"High Win Rate ({win_rate}%).")
    elif win_rate < 40:
        score -= 10
        reasons.append("Historical performance is poor.")

    # 5. FINAL DECISION
    verdict = "WATCH"
    color = "gray"
    
    if score >= 80:
        verdict = "PRIME BUY"
        color = "green"
    elif score >= 60:
        verdict = "MOMENTUM BUY" if not is_fno else "SAFE BUY"
        color = "blue"
    elif score >= 40:
        verdict = "RISKY BUY"
        color = "orange"
    else:
        verdict = "AVOID"
        color = "red"

    return verdict, color, score, reasons

# --- 5. BACKTESTER ---
def run_backtest(df):
    trades = []
    wins, losses = 0, 0
    if len(df) < 200: return {"win_rate": 0, "total": 0, "log": []}

    df['SMA200'] = df['Close'].rolling(200).mean()
    df['Trend'] = df['Close'] > df['SMA200']
    
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9).mean()
    df['Signal'] = (macd > sig) & (macd.shift(1) < sig.shift(1))

    in_pos = False
    entry_price = 0
    
    for i in range(len(df) - 250, len(df)):
        price = df['Close'].iloc[i]
        date = df.index[i].strftime('%Y-%m-%d')
        
        if in_pos:
            pnl = ((price - entry_price) / entry_price) * 100
            if pnl >= 6.0 or pnl <= -3.0:
                res = "WIN" if pnl > 0 else "LOSS"
                trades.append({"date": "", "entry": round(entry_price,2), "result": res, "pnl": round(pnl,2)})
                if pnl > 0: wins += 1
                else: losses += 1
                in_pos = False
        elif df['Trend'].iloc[i] and df['Signal'].iloc[i]:
            in_pos = True
            entry_price = price

    total = wins + losses
    rate = round((wins / total * 100), 0) if total > 0 else 0
    return {"win_rate": rate, "total": total, "log": trades[-5:]}

# --- 6. MASTER ANALYSIS ---
def analyze_stock(ticker, is_fno, nifty_chg):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="5y", interval="1d")
        if df.empty or len(df) < 300: return None

        # Trends
        agg = {'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}
        df_w = df.resample('W').agg(agg).dropna()
        df_m = df.resample('ME').agg(agg).dropna()
        
        def get_trend(d): return "UP" if d['Close'].iloc[-1] > d['Close'].ewm(span=20).mean().iloc[-1] else "DOWN"
        m_trend = get_trend(df_m)
        w_trend = get_trend(df_w)
        
        # Daily
        close = df['Close']
        curr = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        change = round(((curr - prev) / prev) * 100, 2)
        rs_score = round(change - nifty_change, 2)
        
        # Indicators
        sma200 = close.rolling(200).mean().iloc[-1]
        d_trend = "UP" if curr > sma200 else "DOWN"
        
        # Pivot
        r1, s1, pivot = get_pivots(float(df['High'].iloc[-2]), float(df['Low'].iloc[-2]), float(close.iloc[-2]))
        
        # Signals
        signals = []
        sma50 = close.rolling(50).mean()
        if float(sma50.iloc[-2]) < float(sma200) and float(sma50.iloc[-1]) > float(sma200): signals.append("Golden Cross")
        
        rsi_series = 100 - (100 / (1 + (close.diff().where(lambda x: x>0, 0).rolling(14).mean() / close.diff().where(lambda x: x<0, 0).abs().rolling(14).mean())))
        curr_rsi = float(rsi_series.iloc[-1])
        if curr_rsi < 30: signals.append("Oversold")
        
        vol_mult = round(float(df['Volume'].iloc[-1]) / float(df['Volume'].rolling(20).mean().iloc[-1]), 1)
        if vol_mult >= 2.0: signals.append(f"Vol {vol_mult}x")

        # Confluence Score (0-3)
        trend_score = 0
        if m_trend == "UP": trend_score += 1
        if w_trend == "UP": trend_score += 1
        if d_trend == "UP": trend_score += 1
        
        if not signals and trend_score < 3: return None
        
        bt = run_backtest(df)
        
        # *** GET FINAL VERDICT ***
        verdict, v_color, score, reasons = calculate_verdict(trend_score, signals, is_fno, bt['win_rate'], rs_score)

        # ATR Targets
        atr = float((df['High'] - df['Low']).rolling(14).mean().iloc[-1])
        
        return {
            "symbol": ticker.replace(".NS", ""),
            "price": round(curr, 2),
            "change": change,
            "verdict": verdict,
            "v_color": v_color,
            "score": score,
            "reasons": reasons,
            "signals": signals,
            "target": round(curr + (2*atr), 1),
            "stop": round(curr - (1*atr), 1),
            "history": [x if not math.isnan(x) else 0 for x in close.tail(30).tolist()]
        }

    except: return None

# --- 7. GENERATE HTML ---
def generate_html(results):
    json_data = json.dumps(results)
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Decision Engine</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            body {{ background-color: #0f172a; color: #e2e8f0; font-family: sans-serif; }}
            .modal {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.9); z-index: 50; backdrop-filter: blur(5px); }}
            .modal-content {{ background: #1e293b; margin: 5vh auto; width: 95%; max-width: 600px; max-height: 90vh; overflow-y: auto; border-radius: 12px; border: 1px solid #334155; }}
            
            /* Verdict Colors */
            .v-green {{ background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }}
            .v-blue {{ background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }}
            .v-orange {{ background: rgba(249, 115, 22, 0.2); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.4); }}
            .v-red {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}
        </style>
    </head>
    <body>
        <div class="max-w-6xl mx-auto p-4">
            <header class="mb-6 border-b border-slate-700 pb-4">
                <h1 class="text-2xl font-bold text-blue-400 flex items-center gap-2">
                    <i data-lucide="brain-circuit"></i> Decision Engine
                </h1>
                <p class="text-xs text-slate-500 mt-1">Artificial Intelligence Decision Matrix</p>
            </header>
            
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4" id="grid"></div>
        </div>

        <!-- MODAL -->
        <div id="modal" class="modal">
            <div class="modal-content">
                <div class="sticky top-0 bg-slate-800 p-4 border-b border-slate-700 flex justify-between items-center">
                    <div>
                        <h2 id="m-sym" class="text-2xl font-bold text-white">SYMBOL</h2>
                        <div class="text-xs text-slate-400">Analysis Report</div>
                    </div>
                    <button onclick="closeModal()" class="p-2 bg-slate-700 rounded-full"><i data-lucide="x"></i></button>
                </div>
                <div class="p-6 space-y-6">
                    
                    <!-- THE VERDICT BOX -->
                    <div class="bg-slate-800 p-4 rounded-xl border border-slate-700">
                        <h3 class="text-xs text-slate-500 uppercase mb-2">Why this decision?</h3>
                        <ul id="m-reasons" class="list-disc list-inside text-sm text-slate-300 space-y-1"></ul>
                        <div class="mt-4 pt-4 border-t border-slate-700 flex justify-between items-center">
                            <span class="text-xs text-slate-500">CONFIDENCE SCORE</span>
                            <span id="m-score" class="text-xl font-bold text-white">0/100</span>
                        </div>
                    </div>

                    <!-- TARGETS -->
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
                </div>
            </div>
        </div>

        <script>
            const data = {json_data};
            lucide.createIcons();
            
            const grid = document.getElementById('grid');
            if(data.length === 0) grid.innerHTML = '<div class="text-center p-10 text-slate-500">No actionable signals today.</div>';

            data.forEach((s, i) => {{
                const pts = s.history.map((d, j) => {{
                    const min = Math.min(...s.history); const max = Math.max(...s.history);
                    const x = (j / (s.history.length - 1)) * 100;
                    const y = 30 - ((d - min) / (max - min || 1)) * 30;
                    return `${{x}},${{y}}`;
                }}).join(' ');

                let vClass = 'v-red';
                if(s.v_color === 'green') vClass = 'v-green';
                if(s.v_color === 'blue') vClass = 'v-blue';
                if(s.v_color === 'orange') vClass = 'v-orange';

                grid.innerHTML += `
                    <div class="bg-slate-800 p-4 rounded-xl border border-slate-700 shadow-lg hover:border-blue-500/50 transition cursor-pointer relative" onclick="openModal('${{s.symbol}}')">
                        <div class="flex justify-between items-start mb-3">
                            <div>
                                <div class="font-bold text-lg text-white">${{s.symbol}}</div>
                                <div class="text-xs text-slate-400 font-mono mt-1">₹${{s.price}} <span class="${{s.change>=0?'text-green-400':'text-red-400'}} ml-1">${{s.change}}%</span></div>
                            </div>
                            <div class="px-3 py-1 rounded text-xs font-bold uppercase tracking-wider ${{vClass}}">${{s.verdict}}</div>
                        </div>
                        <div class="h-8 w-full opacity-60 mb-3">
                            <svg width="100%" height="100%" preserveAspectRatio="none" class="overflow-visible"><polyline points="${{pts}}" fill="none" stroke="${{s.change >= 0 ? '#4ade80' : '#f87171'}}" stroke-width="2" /></svg>
                        </div>
                        <div class="flex gap-2 text-[10px] text-slate-500">
                            <span>Signals: ${{s.signals.length}}</span>
                            <span>•</span>
                            <span>Score: ${{s.score}}</span>
                        </div>
                    </div>
                `;
            }});

            function openModal(sym) {{
                const s = data.find(x => x.symbol === sym);
                document.getElementById('m-sym').innerText = s.symbol;
                document.getElementById('m-target').innerText = s.target;
                document.getElementById('m-stop').innerText = s.stop;
                document.getElementById('m-score').innerText = s.score + '/100';
                
                const list = document.getElementById('m-reasons');
                list.innerHTML = '';
                s.reasons.forEach(r => {{
                    list.innerHTML += `<li>${{r}}</li>`;
                }});
                
                document.getElementById('modal').style.display = 'block';
            }}
            
            function closeModal() {{ document.getElementById('modal').style.display = 'none'; }}
            window.onclick = function(e) {{ if(e.target == document.getElementById('modal')) closeModal(); }}
        </script>
    </body>
    </html>
    """
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    with open(FILE_PATH, "w", encoding="utf-8") as f: f.write(html)
    print("Generated Decision Dashboard.")

if __name__ == "__main__":
    session = get_session()
    tickers = get_nifty500_list(session)
    fno_list = get_fno_list(session)
    nifty_change = 0 # Simplify
    
    results = []
    print(f"Scanning {{len(tickers)}} stocks...")
    for i, t in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] {t}...", end="\r")
        time.sleep(0.15)
        is_fno = t in fno_list
        res = analyze_stock(t, is_fno, nifty_change)
        if res: results.append(res)
    
    generate_html(results)

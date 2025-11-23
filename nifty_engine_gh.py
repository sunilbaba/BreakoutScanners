import yfinance as yf
import pandas as pd
import os
import time
import math
import requests
import io
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
    print("Downloading Nifty 500 list from Web...")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(NSE_URL, headers=headers, timeout=10)
        df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
        return [f"{x}.NS" for x in df['Symbol'].dropna().tolist()]
    except:
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]

# --- 2. HYBRID ANALYSIS ENGINE ---
def analyze_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        
        # A. TECHNICAL DATA (Fast)
        df = stock.history(period="1y", interval="1d")
        if df.empty or len(df) < 200: return None

        # Prices
        close = df['Close']
        high = df['High']
        low = df['Low']
        curr_price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])
        
        # B. FUNDAMENTAL DATA (Slow - Requires fetching info)
        # We use a try-except block here so if fundamentals fail, technicals still work
        fair_value = 0
        valuation_status = "-"
        try:
            info = stock.info
            eps = info.get('trailingEps', 0)
            book_value = info.get('bookValue', 0)
            
            # GRAHAM NUMBER CALCULATION: Sqrt(22.5 * EPS * BookValue)
            if eps > 0 and book_value > 0:
                graham_num = math.sqrt(22.5 * eps * book_value)
                fair_value = round(graham_num, 1)
                
                # Valuation Logic
                if curr_price < fair_value:
                    diff = round(((fair_value - curr_price) / curr_price) * 100, 0)
                    valuation_status = f"Undervalued (+{diff}%)"
                else:
                    valuation_status = "Overvalued"
            else:
                fair_value = 0 # Cannot calculate for loss-making companies
                valuation_status = "N/A (Neg Earnings)"
        except:
            valuation_status = "Data Error"

        # --- C. INDICATORS ---
        
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
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_signal = "BUY" if float(macd_line.iloc[-1]) > float(signal_line.iloc[-1]) else "SELL"
        
        # Trend
        ema200 = close.ewm(span=200, adjust=False).mean()
        trend = "UP" if curr_price > float(ema200.iloc[-1]) else "DOWN"

        # ATR & Target
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        # Recommendation Logic
        rec = "WAIT"
        tech_target = 0
        
        if trend == "UP":
            if macd_signal == "BUY":
                rec = "STRONG BUY"
                tech_target = curr_price + (2 * atr)
            elif curr_rsi < 40:
                rec = "BUY DIP"
                tech_target = curr_price + (1.5 * atr)
            else:
                rec = "HOLD"
                tech_target = curr_price + atr
        elif trend == "DOWN":
            rec = "SELL" if macd_signal == "SELL" else "AVOID"
            tech_target = curr_price - atr

        return {
            "symbol": ticker.replace(".NS", ""),
            "price": round(curr_price, 2),
            "change": round(((curr_price - prev_price) / prev_price) * 100, 2),
            "fair_value": fair_value,
            "val_status": valuation_status,
            "rec": rec,
            "target": round(tech_target, 1),
            "rsi": round(curr_rsi, 1)
        }

    except Exception as e:
        return None

# --- 3. GENERATE HTML ---
def generate_html(results):
    rows = ""
    for r in results:
        # Colors
        change_col = "#4ade80" if r['change'] >= 0 else "#f87171"
        rec_bg = "transparent"
        rec_col = "#94a3b8"
        
        if "BUY" in r['rec']: 
            rec_bg = "rgba(74, 222, 128, 0.1)"
            rec_col = "#4ade80"
        if "SELL" in r['rec']:
            rec_bg = "rgba(248, 113, 113, 0.1)"
            rec_col = "#f87171"
            
        # Valuation Color
        val_col = "#94a3b8"
        if "Undervalued" in r['val_status']: val_col = "#4ade80" # Green
        if "Overvalued" in r['val_status']: val_col = "#f87171" # Red

        rows += f"""
        <tr style="border-bottom: 1px solid #333;">
            <td style="padding: 12px; font-weight: bold;">{r['symbol']}</td>
            <td style="padding: 12px;">{r['price']}</td>
            <td style="padding: 12px; color: {change_col};">{r['change']}%</td>
            <td style="padding: 12px; color: {val_col}; font-size: 11px;">
                <div style="font-weight: bold; font-size: 13px;">{r['fair_value']}</div>
                <div>{r['val_status']}</div>
            </td>
            <td style="padding: 12px;">
                <span style="background: {rec_bg}; color: {rec_col}; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;">{r['rec']}</span>
            </td>
            <td style="padding: 12px; color: #38bdf8; font-weight: bold;">{r['target']}</td>
            <td style="padding: 12px; color: #64748b;">{r['rsi']}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Nifty Valuation Scanner</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ background: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', sans-serif; padding: 20px; }}
            h1 {{ color: #38bdf8; margin-bottom: 5px; }}
            p {{ color: #64748b; font-size: 12px; margin-bottom: 20px; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: #1e293b; border-radius: 8px; overflow: hidden; }}
            th {{ text-align: left; padding: 12px; background: #0f172a; color: #94a3b8; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; }}
            tr:hover {{ background: #334155; }}
        </style>
    </head>
    <body>
        <h1>Nifty 500 Value & Trend</h1>
        <p>Fair Value = Graham Number (Sqrt(22.5 * EPS * BV)). Target = Technical Projection (ATR).</p>
        
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Price</th>
                    <th>Chg%</th>
                    <th>Fair Value (Graham)</th>
                    <th>Recommendation</th>
                    <th>Tech Target</th>
                    <th>RSI</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </body>
    </html>
    """
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    with open(FILE_PATH, "w", encoding="utf-8") as f: f.write(html)
    print(f"Generated Dashboard with {len(results)} stocks.")

if __name__ == "__main__":
    print("Starting Hybrid Scan...")
    tickers = get_nifty500_list()
    
    results = []
    # SCANNING LOOP
    for i, t in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] {t}...", end="\r")
        # Need longer sleep because we are fetching fundamental INFO now
        time.sleep(0.3) 
        res = analyze_stock(t)
        if res: results.append(res)
        
    generate_html(results)

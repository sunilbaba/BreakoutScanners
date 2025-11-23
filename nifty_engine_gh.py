import yfinance as yf
import pandas as pd
import os
import time
import math
import requests
import io

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
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

# --- 2. TECHNICAL ANALYSIS ENGINE ---
def analyze_stock(ticker):
    try:
        # 1. Fetch Data (1 Year for 200 EMA)
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y", interval="1d")
        
        if df.empty or len(df) < 200: return None

        # 2. Prepare Series
        close = df['Close']
        curr_price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])
        
        # 3. CALCULATE INDICATORS
        
        # --- A. RSI (14) ---
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        curr_rsi = float(rsi.iloc[-1])

        # --- B. MACD (12, 26, 9) ---
        # Fast EMA (12) and Slow EMA (26)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        # MACD Crossover Logic (Bullish if MACD crosses ABOVE Signal)
        macd_val = float(macd_line.iloc[-1])
        sig_val = float(signal_line.iloc[-1])
        prev_macd = float(macd_line.iloc[-2])
        prev_sig = float(signal_line.iloc[-2])
        
        # --- C. BOLLINGER BANDS (20, 2) ---
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper_band = sma20 + (2 * std20)
        lower_band = sma20 - (2 * std20)
        curr_upper = float(upper_band.iloc[-1])
        
        # --- D. EMA 200 (Long Term Trend) ---
        ema200 = close.ewm(span=200, adjust=False).mean()
        curr_ema200 = float(ema200.iloc[-1])
        trend_status = "UP" if curr_price > curr_ema200 else "DOWN"

        # 4. PATTERN DETECTION LOGIC
        signals = []
        
        # RSI Logic
        if curr_rsi < 30: signals.append("RSI Oversold")
        elif curr_rsi < 40 and curr_rsi > float(rsi.iloc[-2]): signals.append("RSI Recovery")
        
        # MACD Logic
        if prev_macd < prev_sig and macd_val > sig_val:
            signals.append("MACD Buy Signal")
            
        # Bollinger Logic
        if curr_price > curr_upper:
            signals.append("Bollinger Breakout")
            
        # EMA Logic
        if prev_price < curr_ema200 and curr_price > curr_ema200:
            signals.append("Trend Reversal (Crossed 200 EMA)")

        # Only return if we have valid data
        return {
            "symbol": ticker.replace(".NS", ""),
            "price": round(curr_price, 2),
            "change": round(((curr_price - prev_price) / prev_price) * 100, 2),
            "rsi": round(curr_rsi, 1),
            "macd_signal": "BUY" if macd_val > sig_val else "SELL",
            "trend": trend_status,
            "signals": ", ".join(signals) if signals else "-"
        }

    except Exception as e:
        return None

# --- 3. GENERATE PRO HTML ---
def generate_html(results):
    rows = ""
    for r in results:
        # Color coding
        change_color = "#4ade80" if r['change'] >= 0 else "#f87171" # Green/Red
        trend_color = "#4ade80" if r['trend'] == "UP" else "#f87171"
        
        # RSI Color
        rsi_bg = "transparent"
        if r['rsi'] < 30: rsi_bg = "#b91c1c" # Deep Red (Oversold)
        if r['rsi'] > 70: rsi_bg = "#15803d" # Deep Green (Overbought)

        # Highlight if Buy Signals exist
        signal_style = "color: #aaa;"
        if r['signals'] != "-": signal_style = "color: #fbbf24; font-weight: bold;" # Amber color

        rows += f"""
        <tr style="border-bottom: 1px solid #333;">
            <td style="padding: 12px; font-weight: bold;">{r['symbol']}</td>
            <td style="padding: 12px;">{r['price']}</td>
            <td style="padding: 12px; color: {change_color};">{r['change']}%</td>
            <td style="padding: 12px;">
                <span style="background: {rsi_bg}; padding: 2px 6px; rounded: 4px;">{r['rsi']}</span>
            </td>
            <td style="padding: 12px; color: {trend_color}; font-size: 11px;">{r['trend']}</td>
            <td style="padding: 12px; font-size: 11px;">{r['macd_signal']}</td>
            <td style="padding: 12px; font-size: 12px; {signal_style}">{r['signals']}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pro Nifty Scanner</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ background: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; }}
            h1 {{ color: #38bdf8; border-bottom: 2px solid #1e293b; padding-bottom: 10px; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 14px; background: #1e293b; }}
            th {{ text-align: left; padding: 12px; background: #0f172a; color: #94a3b8; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; border-bottom: 2px solid #334155; }}
            tr:hover {{ background: #334155; }}
        </style>
    </head>
    <body>
        <h1>Nifty 500 Pro Scanner</h1>
        <p>Total Scanned: {len(results)} | Time: {pd.Timestamp.now()}</p>
        
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Price</th>
                    <th>% Chg</th>
                    <th>RSI (14)</th>
                    <th>Trend (200 EMA)</th>
                    <th>MACD</th>
                    <th>Technical Signals</th>
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
    print(f"Generated Pro Dashboard with {len(results)} stocks.")

if __name__ == "__main__":
    print("Starting Pro Scan...")
    tickers = get_nifty500_list()
    print(f"Stocks found: {len(tickers)}")
    
    results = []
    for i, t in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] {t}...", end="\r")
        # Rate limiting to prevent 0 results
        time.sleep(0.2) 
        res = analyze_stock(t)
        if res: results.append(res)
        
    generate_html(results)

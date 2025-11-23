import yfinance as yf
import pandas as pd
import json
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

# --- 1. GET LIST (Auto-Download if missing) ---
def get_nifty500_list():
    tickers = []
    
    # Try reading local file
    if os.path.exists(CSV_FILENAME):
        try:
            df = pd.read_csv(CSV_FILENAME)
            tickers = [f"{x}.NS" for x in df['Symbol'].dropna().tolist()]
            print(f"Loaded {len(tickers)} stocks from local CSV.")
            return tickers
        except Exception as e:
            print(f"Error reading local CSV: {e}")

    # If local failed, download from Web
    print("Downloading CSV from NSE...")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        s = requests.Session()
        r = s.get(NSE_URL, headers=headers, timeout=10)
        df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
        tickers = [f"{x}.NS" for x in df['Symbol'].dropna().tolist()]
        print(f"Downloaded {len(tickers)} stocks from Web.")
        return tickers
    except Exception as e:
        print(f"Download failed: {e}")
        # Fallback list so script doesn't crash
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

# --- 2. ANALYSIS ENGINE (No Custom Session) ---
def analyze_stock(ticker):
    try:
        # PAUSE to prevent rate limiting (Crucial for 500 stocks)
        time.sleep(0.25) 
        
        # FIX: Do not pass session. Let yfinance handle it.
        stock = yf.Ticker(ticker)
        
        # Fetch Data
        df = stock.history(period="6mo", interval="1d")
        
        if df.empty: 
            return {"symbol": ticker, "status": "FAIL", "msg": "No Data"}
        
        if len(df) < 50:
            return {"symbol": ticker, "status": "FAIL", "msg": "Insufficient Data"}

        # Extract Data
        close = df['Close']
        volume = df['Volume']
        
        curr = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        
        # Math safety
        if math.isnan(curr) or math.isnan(prev): return None
        
        change = round(((curr - prev) / prev) * 100, 2)
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        curr_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

        # SMA
        sma50_series = close.rolling(50).mean()
        sma50 = float(sma50_series.iloc[-1]) if not pd.isna(sma50_series.iloc[-1]) else 0.0
        
        # Volume
        vol_avg = float(volume.rolling(20).mean().iloc[-1])
        curr_vol = float(volume.iloc[-1])

        patterns = []
        if curr_rsi < 30: patterns.append("Oversold")
        if curr_rsi > 70: patterns.append("Overbought")
        if curr > sma50 and prev < sma50: patterns.append("SMA Breakout")
        if curr_vol > (vol_avg * 2): patterns.append("Vol Spike")
        
        return {
            "symbol": ticker.replace(".NS", ""),
            "status": "OK",
            "price": round(curr, 2),
            "change": change,
            "rsi": round(curr_rsi, 1),
            "patterns": ", ".join(patterns) if patterns else "-"
        }
    except Exception as e:
        return {"symbol": ticker, "status": "FAIL", "msg": str(e)}

# --- 3. GENERATE HTML ---
def generate_html(results):
    rows = ""
    success_count = 0
    
    for r in results:
        if r['status'] == "OK":
            success_count += 1
            color = "#4ade80" if r['change'] >= 0 else "#f87171"
            rows += f"""
            <tr style="border-bottom: 1px solid #333;">
                <td style="padding: 12px; font-weight: bold;">{r['symbol']}</td>
                <td style="padding: 12px;">{r['price']}</td>
                <td style="padding: 12px; color: {color};">{r['change']}%</td>
                <td style="padding: 12px;">{r['rsi']}</td>
                <td style="padding: 12px; font-size: 12px; color: #aaa;">{r['patterns']}</td>
            </tr>
            """
        else:
            # Error Row
            rows += f"""
            <tr style="border-bottom: 1px solid #333; background: #2a1111;">
                <td style="padding: 12px; color: #f87171;">{r['symbol']}</td>
                <td colspan="4" style="padding: 12px; color: #f87171;">FAILED: {r['msg']}</td>
            </tr>
            """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Nifty 500 Scanner</title>
        <style>
            body {{ background: #111; color: white; font-family: sans-serif; padding: 20px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th {{ text-align: left; background: #222; padding: 10px; color: #888; }}
        </style>
    </head>
    <body>
        <h1>Scanner Results</h1>
        <p>Total: {len(results)} | Success: {success_count}</p>
        <table>
            <thead><tr><th>SYM</th><th>PRICE</th><th>%</th><th>RSI</th><th>NOTE</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </body>
    </html>
    """
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    with open(FILE_PATH, "w", encoding="utf-8") as f: f.write(html)
    print(f"Dashboard generated with {len(results)} rows.")

if __name__ == "__main__":
    print("Starting Scan...")
    tickers = get_nifty500_list()
    print(f"Stocks found: {len(tickers)}")
    
    results = []
    # Removed the limit! Will scan all.
    for i, t in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] Scanning {t}...", end="\r")
        res = analyze_stock(t)
        if res: results.append(res)
        
    generate_html(results)

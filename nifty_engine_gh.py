import yfinance as yf
import pandas as pd
import os
import time
import requests
import warnings
import traceback

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURATION ---
OUTPUT_DIR = "public"
FILE_PATH = os.path.join(OUTPUT_DIR, "index.html")
CSV_FILENAME = "ind_nifty500list.csv"

# Global Log Buffer
SYSTEM_LOGS = []

def log(message):
    """Prints to GitHub Console AND saves to Website Log"""
    timestamp = time.strftime("%H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg)
    SYSTEM_LOGS.append(msg)

def get_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    return session

def get_nifty500_list():
    log("--- STEP 1: LOADING SYMBOL LIST ---")
    
    # DEBUG: Check file system
    cwd = os.getcwd()
    log(f"Current Directory: {cwd}")
    try:
        files = os.listdir(cwd)
        log(f"Files found in folder: {files}")
    except Exception as e:
        log(f"Error listing files: {e}")

    # Try loading CSV
    if os.path.exists(CSV_FILENAME):
        log(f"SUCCESS: Found {CSV_FILENAME}")
        try:
            df = pd.read_csv(CSV_FILENAME)
            if 'Symbol' in df.columns:
                symbols = [f"{x}.NS" for x in df['Symbol'].dropna().tolist()]
                log(f"Parsed {len(symbols)} symbols from CSV.")
                return symbols
            else:
                log("ERROR: CSV found but no 'Symbol' column!")
        except Exception as e:
            log(f"CRITICAL: Error reading CSV: {str(e)}")
    else:
        log(f"FAILURE: {CSV_FILENAME} NOT FOUND in {cwd}")

    # Fallback
    log("WARNING: Using Fallback List (Top 5 Stocks Only)")
    return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

def analyze_stock(ticker, session):
    try:
        ticker_obj = yf.Ticker(ticker, session=session)
        
        # Try fetching just 1 month first (faster)
        df = ticker_obj.history(period="1mo", interval="1d")
        
        if df.empty:
            return {"symbol": ticker, "status": "FAIL", "reason": "No Data (Blocked?)"}
        
        # Data Exists
        close = df['Close']
        curr = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        change = round(((curr - prev) / prev) * 100, 2)
        
        # RSI Calculation
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        curr_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

        patterns = []
        if curr_rsi < 30: patterns.append("Oversold")
        if change > 3: patterns.append("Big Move Up")
        
        return {
            "symbol": ticker.replace(".NS", ""),
            "status": "OK",
            "price": round(curr, 2),
            "change": change,
            "rsi": round(curr_rsi, 1),
            "patterns": ", ".join(patterns) if patterns else "-"
        }

    except Exception as e:
        return {"symbol": ticker, "status": "FAIL", "reason": str(e)}

def generate_html(results):
    log("--- STEP 3: GENERATING HTML ---")
    
    rows = ""
    for r in results:
        if r['status'] == "OK":
            color = "#4ade80" if r['change'] >= 0 else "#f87171"
            rows += f"""
            <tr style="border-bottom: 1px solid #333;">
                <td style="padding: 10px; font-weight: bold;">{r['symbol']}</td>
                <td style="padding: 10px;">{r['price']}</td>
                <td style="padding: 10px; color: {color};">{r['change']}%</td>
                <td style="padding: 10px;">{r['rsi']}</td>
                <td style="padding: 10px; color: #888; font-size: 12px;">{r['patterns']}</td>
            </tr>"""
        else:
            rows += f"""
            <tr style="border-bottom: 1px solid #333; background: #2a1111;">
                <td style="padding: 10px; color: #f87171;">{r['symbol']}</td>
                <td colspan="4" style="padding: 10px; color: #f87171;">FAILED: {r['reason']}</td>
            </tr>"""

    # EMBED LOGS INTO HTML
    log_html = "<br>".join(SYSTEM_LOGS)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Scanner Diagnostic</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ background: #111; color: white; font-family: monospace; padding: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th {{ text-align: left; background: #333; padding: 10px; }}
            .logs {{ background: #000; color: #0f0; padding: 15px; border: 1px solid #333; margin-top: 30px; height: 300px; overflow-y: scroll; }}
        </style>
    </head>
    <body>
        <h1>Market Scanner</h1>
        <p>Total Scanned: {len(results)}</p>
        
        <table>
            <thead><tr><th>Symbol</th><th>Price</th><th>Change</th><th>RSI</th><th>Pattern</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>

        <h3>SYSTEM DEBUG LOGS (Check here for errors)</h3>
        <div class="logs">{log_html}</div>
    </body>
    </html>
    """
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    with open(FILE_PATH, "w", encoding="utf-8") as f: f.write(html)
    log(f"HTML Generated at {FILE_PATH}")

if __name__ == "__main__":
    session = get_session()
    
    # 1. Get Symbols
    tickers = get_nifty500_list()
    
    # 2. Scan
    results = []
    log(f"--- STEP 2: SCANNING {len(tickers)} STOCKS ---")
    
    # Limit to 50 for testing to avoid timeout, remove limit later
    scan_limit = tickers[:50] 
    
    for i, t in enumerate(scan_limit):
        log(f"Scanning {i+1}/{len(scan_limit)}: {t}")
        res = analyze_stock(t, session)
        results.append(res)
        time.sleep(0.1) 
        
    generate_html(results)

import yfinance as yf
import pandas as pd
import os
import time
import math
import csv

# --- CONFIGURATION ---
OUTPUT_DIR = "public"
FILE_PATH = os.path.join(OUTPUT_DIR, "index.html")
CSV_FILENAME = "ind_nifty500list.csv"

def get_nifty500_list():
    if os.path.exists(CSV_FILENAME):
        try:
            df = pd.read_csv(CSV_FILENAME)
            return [f"{x}.NS" for x in df['Symbol'].dropna().tolist()]
        except:
            pass
    # Fallback list if CSV fails
    return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS"]

def analyze_stock(ticker):
    try:
        time.sleep(0.1) 
        stock = yf.Ticker(ticker)
        df = stock.history(period="6mo", interval="1d")
        
        if df.empty or len(df) < 50: return None

        close = df['Close']
        curr = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        change = round(((curr - prev) / prev) * 100, 2)
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        curr_rsi = float(rsi.iloc[-1])

        # SMA
        sma50 = float(close.rolling(50).mean().iloc[-1])

        patterns = []
        if curr_rsi < 30: patterns.append("RSI Oversold")
        if curr > sma50 and prev < sma50: patterns.append("SMA Breakout")

        if not patterns: return None # Only show stocks with patterns? 
        # Actually, let's return everything for now so you see DATA.
        
        return {
            "symbol": ticker.replace(".NS", ""),
            "price": round(curr, 2),
            "change": change,
            "rsi": round(curr_rsi, 1),
            "patterns": ", ".join(patterns) if patterns else "-"
        }
    except:
        return None

def generate_simple_html(results):
    rows = ""
    for r in results:
        color = "green" if r['change'] >= 0 else "red"
        rows += f"""
        <tr style="border-bottom: 1px solid #333;">
            <td style="padding: 10px; font-weight: bold;">{r['symbol']}</td>
            <td style="padding: 10px;">{r['price']}</td>
            <td style="padding: 10px; color: {color};">{r['change']}%</td>
            <td style="padding: 10px;">{r['rsi']}</td>
            <td style="padding: 10px; color: #aaa;">{r['patterns']}</td>
        </tr>
        """
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <body style="background: #111; color: white; font-family: sans-serif; padding: 20px;">
        <h1>Nifty 500 Scan Results</h1>
        <p>Last Updated: {pd.Timestamp.now()}</p>
        <p>Total Stocks Scanned: {len(results)}</p>
        
        <table style="width: 100%; border-collapse: collapse; text-align: left;">
            <thead style="background: #222;">
                <tr>
                    <th style="padding: 10px;">Symbol</th>
                    <th style="padding: 10px;">Price</th>
                    <th style="padding: 10px;">Change</th>
                    <th style="padding: 10px;">RSI</th>
                    <th style="padding: 10px;">Patterns</th>
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
    print("Generated Simple HTML.")

if __name__ == "__main__":
    print("Starting Scan...")
    tickers = get_nifty500_list()
    results = []
    
    for t in tickers:
        data = analyze_stock(t)
        if data: results.append(data)
        
    generate_simple_html(results)

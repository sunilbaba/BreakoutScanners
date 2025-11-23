import yfinance as yf
import pandas as pd
import json
import os
import shutil

# --- CONFIGURATION ---
OUTPUT_DIR = "public"
FILE_PATH = os.path.join(OUTPUT_DIR, "index.html")

STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS",
    "TATAMOTORS.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "TITAN.NS", "BAJFINANCE.NS", "SUNPHARMA.NS", "HCLTECH.NS",
    "WIPRO.NS", "ULTRACEMCO.NS", "NTPC.NS", "POWERGRID.NS",
    "ONGC.NS", "M&M.NS", "ADANIENT.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
    "COALINDIA.NS", "HINDUNILVR.NS", "GRASIM.NS", "HEROMOTOCO.NS",
    "EICHERMOT.NS", "DIVISLAB.NS", "DRREDDY.NS", "CIPLA.NS", "BPCL.NS"
]

def analyze_stock(ticker):
    try:
        # User-Agent trick to prevent Yahoo from blocking GitHub Servers
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y", interval="1d")
        
        if df.empty or len(df) < 50: return None
        
        # Simple calculations to avoid pandas errors
        close = df['Close']
        curr_price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])
        change_pct = round(((curr_price - prev_price) / prev_price) * 100, 2)
        
        # Calculate RSI manually
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        curr_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

        # Patterns
        sma50 = close.rolling(50).mean()
        curr_sma50 = float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else 0.0
        
        patterns = []
        if curr_rsi < 30: patterns.append("RSI Oversold")
        if curr_price > curr_sma50 and prev_price < curr_sma50: patterns.append("SMA Breakout")

        return {
            "symbol": ticker.replace(".NS", ""),
            "price": round(curr_price, 2),
            "change": change_pct,
            "rsi": round(curr_rsi, 1),
            "patterns": patterns,
            # Convert NaN to None for JSON safety
            "history": [x if not pd.isna(x) else 0 for x in close.tail(30).tolist()] 
        }
    except Exception as e:
        print(f"Error scanning {ticker}: {e}")
        return None

def generate_dashboard(stock_data):
    # FALLBACK: If data is empty, provide dummy data so the app doesn't crash white
    if not stock_data:
        stock_data = [{"symbol": "NO DATA", "price": 0, "change": 0, "rsi": 0, "patterns": [], "history": [0]*30}]
        print("WARNING: No stock data fetched. Using dummy data.")

    json_data = json.dumps(stock_data)

    # We use {{ }} for CSS/JS to escape them in Python f-strings
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nifty Scanner</title>
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
</head>
<body class="bg-slate-900 text-white">
    <div id="root" class="h-screen flex items-center justify-center">
        <div class="animate-pulse text-blue-400 font-bold">Loading Scanner Data...</div>
    </div>

    <script type="text/babel">
        // SAFE DATA INJECTION
        const REAL_DATA = {json_data};
        
        const {{ useState, useEffect }} = React;
        const {{ TrendingUp, Activity, Menu, X, Play, Filter }} = lucide;

        const App = () => {{
            const [stocks, setStocks] = useState(REAL_DATA);
            
            useEffect(() => {{
                if (window.lucide) window.lucide.createIcons();
            }});

            return (
                <div className="max-w-md mx-auto h-screen bg-slate-900 flex flex-col font-sans">
                    <header className="p-4 bg-slate-800 border-b border-slate-700 flex justify-between items-center">
                        <h1 className="font-bold text-lg flex items-center gap-2">
                            <Activity size={{20}} className="text-blue-500" /> NiftyScan
                        </h1>
                        <span className="text-xs bg-slate-700 px-2 py-1 rounded text-slate-300">
                            {{stocks.length}} Stocks
                        </span>
                    </header>
                    
                    <div className="flex-1 overflow-y-auto p-4 space-y-3">
                        {{stocks.map((stock, i) => (
                            <div key={{i}} className="bg-slate-800 p-4 rounded-xl border border-slate-700 shadow-sm">
                                <div className="flex justify-between items-start mb-2">
                                    <div>
                                        <div className="font-bold text-lg">{{stock.symbol}}</div>
                                        <div className="flex items-center gap-2">
                                            <span className="text-slate-400 text-sm">₹{{stock.price}}</span>
                                            <span className={{`text-xs font-bold px-1 rounded ${{stock.change >= 0 ? 'text-green-400 bg-green-900/30' : 'text-red-400 bg-red-900/30'}}`}}>
                                                {{stock.change}}%
                                            </span>
                                        </div>
                                    </div>
                                    <div className={{`px-2 py-1 rounded text-xs font-bold ${{stock.rsi < 30 ? 'bg-red-500 text-white' : 'bg-slate-700 text-slate-400'}}`}}>
                                        RSI {{stock.rsi}}
                                    </div>
                                </div>
                                <div className="flex flex-wrap gap-1">
                                    {{stock.patterns.length > 0 ? stock.patterns.map(p => (
                                        <span key={{p}} className="text-[10px] bg-blue-900 text-blue-200 px-1 rounded border border-blue-700">
                                            {{p}}
                                        </span>
                                    )) : <span className="text-[10px] text-slate-600">No Signals</span>}}
                                </div>
                            </div>
                        ))}}
                    </div>
                </div>
            );
        }};

        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(<App />);
    </script>
</body>
</html>
    """

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    with open(FILE_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"SUCCESS: Dashboard generated at {FILE_PATH}")

if __name__ == "__main__":
    print("Starting Scan...")
    results = []
    # Fetch sequentially to be safer against blocking
    for symbol in STOCKS:
        data = analyze_stock(symbol)
        if data:
            results.append(data)
            print(f"Scanned {symbol}")
            
    generate_dashboard(results)

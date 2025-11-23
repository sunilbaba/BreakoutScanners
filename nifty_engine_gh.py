import yfinance as yf
import pandas as pd
import json
import os
import time
import math

# --- CONFIGURATION ---
OUTPUT_DIR = "public"
FILE_PATH = os.path.join(OUTPUT_DIR, "index.html")
CSV_FILENAME = "ind_nifty500list.csv"

# --- HELPER: CLEAN DATA ---
def clean_float(val):
    """Converts NaN/Infinity to 0 to prevent Browser Crash"""
    if val is None: return 0.0
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val): return 0.0
    return val

def get_nifty500_list():
    try:
        if os.path.exists(CSV_FILENAME):
            print(f"Loading symbols from {CSV_FILENAME}...")
            df = pd.read_csv(CSV_FILENAME)
            # Filter valid symbols and add .NS
            symbols = [f"{x}.NS" for x in df['Symbol'].dropna().tolist() if isinstance(x, str)]
            return symbols
        else:
            print(f"WARNING: {CSV_FILENAME} not found. Using fallback list.")
            return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS", "ICICIBANK.NS", "ITC.NS"]
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return ["RELIANCE.NS"]

def analyze_stock(ticker):
    try:
        time.sleep(0.1) # Tiny pause for rate limiting
        
        stock = yf.Ticker(ticker)
        # Fetch data
        df = stock.history(period="1y", interval="1d")
        
        if df.empty: return {"symbol": ticker, "status": "failed", "error": "No Data"}
        if len(df) < 50: return {"symbol": ticker, "status": "failed", "error": "Insufficient Data"}

        # Safe Calculations
        close = df['Close']
        volume = df['Volume']
        
        curr_price = clean_float(float(close.iloc[-1]))
        prev_price = clean_float(float(close.iloc[-2]))
        
        # Avoid division by zero
        if prev_price == 0: change_pct = 0.0
        else: change_pct = clean_float(round(((curr_price - prev_price) / prev_price) * 100, 2))
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        curr_rsi = clean_float(float(rsi.iloc[-1])) if not pd.isna(rsi.iloc[-1]) else 50.0

        # SMA
        sma50 = close.rolling(50).mean()
        curr_sma50 = clean_float(float(sma50.iloc[-1])) if not pd.isna(sma50.iloc[-1]) else 0.0
        
        # Patterns
        patterns = []
        if curr_rsi < 30: patterns.append("RSI Oversold")
        if curr_price > curr_sma50 and prev_price < curr_sma50: patterns.append("SMA Breakout")

        # History for Sparkline (Clean every single point)
        history_raw = close.tail(30).tolist()
        history_clean = [clean_float(x) for x in history_raw]

        return {
            "symbol": ticker.replace(".NS", ""),
            "status": "success",
            "price": curr_price,
            "change": change_pct,
            "rsi": round(curr_rsi, 1),
            "patterns": patterns,
            "history": history_clean
        }

    except Exception as e:
        return {"symbol": ticker, "status": "failed", "error": str(e)}

def generate_dashboard(results):
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    meta = {
        "total": len(results),
        "success": len(successful),
        "failed": len(failed),
        "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
    }
    
    # Dump JSON (ensure no NaN leaks)
    data_json = json.dumps({"stocks": successful, "logs": failed, "meta": meta}, default=str)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nifty 500 Scanner</title>
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
</head>
<body class="bg-slate-900 text-white">
    <div id="root" class="min-h-screen flex items-center justify-center">
        <div class="text-center">
             <div class="text-blue-500 font-bold text-xl animate-pulse">Loading Nifty 500 Data...</div>
             <div class="text-slate-500 text-sm mt-2">If this takes too long, check console (F12)</div>
        </div>
    </div>

    <script type="text/babel">
        // INJECT DATA
        const RAW = {data_json};
        
        const {{ useState, useEffect }} = React;
        const {{ Activity, AlertCircle, Search }} = lucide;

        const App = () => {{
            const [view, setView] = useState('stocks');
            const [filter, setFilter] = useState('ALL');
            const [search, setSearch] = useState('');
            const [stocks] = useState(RAW.stocks || []);
            const [logs] = useState(RAW.logs || []);
            
            useEffect(() => {{ if(window.lucide) window.lucide.createIcons(); }}, [view, stocks]);

            const displayed = stocks.filter(s => {{
                const matchF = filter === 'ALL' ? true : s.patterns.length > 0;
                const matchS = s.symbol.toLowerCase().includes(search.toLowerCase());
                return matchF && matchS;
            }});

            const Sparkline = ({{ data, color }}) => {{
                if(!data || data.length < 2) return null;
                const min = Math.min(...data); const max = Math.max(...data);
                if (min === max) return null;
                const points = data.map((d, i) => {{
                    const x = (i / (data.length - 1)) * 100;
                    const y = 30 - ((d - min) / (max - min)) * 30;
                    return `${{x}},${{y}}`;
                }}).join(' ');
                return <svg width="100%" height="30" viewBox="0 0 100 30" preserveAspectRatio="none"><polyline points={{points}} fill="none" stroke={{color}} strokeWidth="2" /></svg>;
            }};

            return (
                <div className="max-w-4xl mx-auto min-h-screen bg-slate-900 flex flex-col">
                    <div className="p-4 bg-slate-800 border-b border-slate-700 sticky top-0 z-20 shadow-lg">
                        <div className="flex justify-between items-center mb-3">
                            <h1 className="font-bold text-xl flex items-center gap-2"><Activity className="text-blue-500" /> NiftyScan</h1>
                            <div className="flex gap-2 text-xs">
                                <button onClick={{() => setView('stocks')}} className={{`px-3 py-1 rounded ${{view==='stocks' ? 'bg-blue-600' : 'bg-slate-700'}}`}}>Stocks ({{RAW.meta.success}})</button>
                                <button onClick={{() => setView('logs')}} className={{`px-3 py-1 rounded text-red-300 ${{view==='logs' ? 'bg-red-900/50' : 'bg-slate-700'}}`}}>Failed ({{RAW.meta.failed}})</button>
                            </div>
                        </div>
                        {{view === 'stocks' && (
                            <div className="flex gap-2">
                                <input type="text" placeholder="Search..." className="flex-1 bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500" value={{search}} onChange={{e => setSearch(e.target.value)}} />
                                <button onClick={{() => setFilter(filter === 'ALL' ? 'BREAKOUT' : 'ALL')}} className={{`px-4 rounded text-sm font-bold ${{filter==='BREAKOUT' ? 'bg-blue-600' : 'bg-slate-700 text-slate-400'}}`}}>
                                    {{filter === 'ALL' ? 'Breakouts' : 'All'}}
                                </button>
                            </div>
                        )}}
                    </div>

                    <div className="flex-1 p-3 space-y-3">
                        {{view === 'stocks' ? displayed.map((stock, i) => (
                            <div key={{i}} className="bg-slate-800 p-4 rounded-xl border border-slate-700 shadow-sm">
                                <div className="flex justify-between items-start">
                                    <div>
                                        <div className="font-bold text-lg">{{stock.symbol}}</div>
                                        <div className="flex items-center gap-2">
                                            <span className="text-slate-400 text-sm">₹{{stock.price}}</span>
                                            <span className={{`text-xs font-bold px-1 rounded ${{stock.change >= 0 ? 'text-green-400 bg-green-900/20' : 'text-red-400 bg-red-900/20'}}`}}>{{stock.change}}%</span>
                                        </div>
                                    </div>
                                    <div className={{`px-2 py-1 rounded text-xs font-bold ${{stock.rsi < 30 ? 'bg-red-500 text-white' : 'bg-slate-700 text-slate-400'}}`}}>RSI {{stock.rsi}}</div>
                                </div>
                                <div className="mt-3 grid grid-cols-2 gap-4 items-center">
                                    <div className="flex flex-wrap gap-1">
                                        {{stock.patterns.length > 0 ? stock.patterns.map(p => (<span key={{p}} className="text-[10px] uppercase font-bold bg-blue-900/40 text-blue-200 border border-blue-500/30 px-1.5 py-0.5 rounded">{{p}}</span>)) : <span className="text-[10px] text-slate-600 italic">No patterns</span>}}
                                    </div>
                                    <div className="h-8 w-full opacity-80"><Sparkline data={{stock.history}} color={{stock.change >= 0 ? '#4ade80' : '#f87171'}} /></div>
                                </div>
                            </div>
                        )) : (
                            <div className="space-y-2">
                                <div className="text-xs text-slate-500 p-2">Showing failed attempts from CSV list:</div>
                                {{logs.map((log, i) => (
                                    <div key={{i}} className="bg-slate-800/50 p-3 rounded border border-red-900/30 flex justify-between items-center">
                                        <span className="font-mono text-sm text-red-200">{{log.symbol}}</span>
                                        <span className="text-xs text-red-400">{{log.error}}</span>
                                    </div>
                                ))}}
                            </div>
                        )}}
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
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    with open(FILE_PATH, "w", encoding="utf-8") as f: f.write(html_content)
    print(f"SUCCESS: Dashboard generated at {FILE_PATH}")

if __name__ == "__main__":
    print("Starting Nifty 500 Scan...")
    tickers = get_nifty500_list()
    print(f"Found {len(tickers)} stocks.")
    
    results = []
    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] {ticker}", end="\r")
        results.append(analyze_stock(ticker))
        
    generate_dashboard(results)

import yfinance as yf
import pandas as pd
import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
OUTPUT_DIR = "public"
DASHBOARD_FILENAME = os.path.join(OUTPUT_DIR, "index.html")

# Target Stocks (Add more as needed)
STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS",
    "TATAMOTORS.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "TITAN.NS", "BAJFINANCE.NS", "SUNPHARMA.NS", "HCLTECH.NS",
    "WIPRO.NS", "ULTRACEMCO.NS", "NTPC.NS", "POWERGRID.NS",
    "ONGC.NS", "M&M.NS", "ADANIENT.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
    "COALINDIA.NS", "HINDUNILVR.NS", "GRASIM.NS", "HEROMOTOCO.NS",
    "EICHERMOT.NS", "DIVISLAB.NS", "DRREDDY.NS", "CIPLA.NS", "BPCL.NS",
    "ZOMATO.NS", "JIOFIN.NS", "DMART.NS", "HAL.NS", "BEL.NS", "VBL.NS"
]

def analyze_stock(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty or len(df) < 200: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        close = df['Close']
        volume = df['Volume']
        
        # Indicators
        sma50 = close.rolling(window=50).mean()
        sma200 = close.rolling(window=200).mean()
        delta = close.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        vol_avg = volume.rolling(window=20).mean()

        # Latest Values
        curr_price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])
        curr_rsi = float(rsi.iloc[-1])
        curr_vol = float(volume.iloc[-1])
        curr_vol_avg = float(vol_avg.iloc[-1])
        curr_sma50 = float(sma50.iloc[-1])
        curr_sma200 = float(sma200.iloc[-1])
        prev_sma50 = float(sma50.iloc[-2])
        prev_sma200 = float(sma200.iloc[-2])
        change_pct = round(((curr_price - prev_price) / prev_price) * 100, 2)

        patterns = []
        details = {}

        # Logic
        if prev_price < prev_sma50 and curr_price > curr_sma50:
            patterns.append("SMA 50 Breakout")
            details["SMA Breakout"] = f"Price crossed SMA50 ({round(curr_sma50, 2)})"

        if prev_sma50 < prev_sma200 and curr_sma50 > curr_sma200:
            patterns.append("Golden Cross")
            details["Golden Cross"] = "50 SMA crossed above 200 SMA"

        if curr_rsi < 30:
            patterns.append("RSI Oversold")
            details["RSI Status"] = f"Oversold at {round(curr_rsi, 1)}"
        elif 30 <= curr_rsi < 40 and float(rsi.iloc[-2]) < 30:
            patterns.append("RSI Bounce")
            details["RSI Status"] = f"Bouncing up from oversold ({round(curr_rsi, 1)})"

        if curr_vol > (curr_vol_avg * 2):
            patterns.append("Volume Spike")
            details["Volume"] = f"{int(curr_vol)} (2x Average)"

        return {
            "symbol": ticker.replace(".NS", ""),
            "price": round(curr_price, 2),
            "change": change_pct,
            "rsi": round(curr_rsi, 1),
            "vol": int(curr_vol),
            "sma50": round(curr_sma50, 2),
            "sma200": round(curr_sma200, 2),
            "patterns": patterns,
            "details": details,
            "history": close.tail(30).tolist()
        }
    except Exception:
        return None

def generate_dashboard(stock_data):
    json_data = json.dumps(stock_data)
    
    # HTML Template
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Nifty 500 Live Scanner</title>
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        body {{ background-color: #0f172a; color: white; -webkit-tap-highlight-color: transparent; }}
        .no-scrollbar::-webkit-scrollbar {{ display: none; }}
        .no-scrollbar {{ -ms-overflow-style: none;  scrollbar-width: none; }}
    </style>
</head>
<body>
    <div id="root"></div>
    <script type="text/babel">
        const REAL_DATA = {json_data};
        const {{ useState, useEffect }} = React;
        const {{ TrendingUp, Activity, Menu, X, Play, Filter, CheckCircle, Eye }} = lucide;

        const App = () => {{
            const [stocks, setStocks] = useState(REAL_DATA);
            const [filtered, setFiltered] = useState(REAL_DATA);
            const [selectedStock, setSelectedStock] = useState(null);
            const [filter, setFilter] = useState('ALL');
            const [sidebarOpen, setSidebarOpen] = useState(false);

            useEffect(() => {{ lucide.createIcons(); }}, []);

            useEffect(() => {{
                if (filter === 'ALL') setFiltered(stocks);
                else setFiltered(stocks.filter(s => s.patterns.length > 0));
            }}, [filter, stocks]);

            const Sparkline = ({{ data, color }}) => {{
                if(!data || data.length === 0) return null;
                const min = Math.min(...data); const max = Math.max(...data); const range = max - min || 1; 
                const points = data.map((d, i) => {{
                    const x = (i / (data.length - 1)) * 100;
                    const y = 30 - ((d - min) / range) * 30;
                    return `${{x}},${{y}}`;
                }}).join(' ');
                return <svg width="100%" height="30" viewBox="0 0 100 30" preserveAspectRatio="none" className="overflow-visible"><polyline points={{points}} fill="none" stroke={{color}} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>;
            }};

            return (
                <div className="flex flex-col h-screen bg-slate-900 text-slate-100 font-sans overflow-hidden">
                    <header className="flex items-center justify-between px-4 py-3 bg-slate-800 border-b border-slate-700 shadow-md z-30 shrink-0">
                        <div className="flex items-center gap-2">
                            <div className="bg-blue-600 p-1.5 rounded-lg"><Activity size={{20}} className="text-white" /></div>
                            <div><h1 className="text-lg font-bold tracking-tight leading-tight">NiftyScan</h1><p className="text-[10px] text-slate-400 leading-none">Daily Update</p></div>
                        </div>
                        <button onClick={{() => setSidebarOpen(!sidebarOpen)}} className="p-2 bg-slate-700 rounded-md md:hidden"><Menu size={{20}}/></button>
                    </header>
                    <div className="flex flex-1 overflow-hidden relative">
                        <div className={{`absolute inset-y-0 left-0 w-64 bg-slate-800 border-r border-slate-700 p-4 transform transition-transform duration-300 z-20 md:relative md:translate-x-0 ${{sidebarOpen ? 'translate-x-0' : '-translate-x-full'}}`}}>
                            <div className="bg-slate-700/50 p-4 rounded-xl border border-slate-600 mb-4">
                                <div className="text-2xl font-bold text-white">{{stocks.length}}</div>
                                <div className="text-xs text-slate-400">Stocks Scanned Today</div>
                            </div>
                            <div className="flex flex-col gap-2">
                                <button onClick={{() => {{setFilter('ALL'); setSidebarOpen(false)}}}} className={{`p-3 rounded-lg text-left text-sm font-bold ${{filter === 'ALL' ? 'bg-blue-600 text-white' : 'bg-slate-700 text-slate-400'}}`}}>All Stocks</button>
                                <button onClick={{() => {{setFilter('BREAKOUTS'); setSidebarOpen(false)}}}} className={{`p-3 rounded-lg text-left text-sm font-bold ${{filter === 'BREAKOUTS' ? 'bg-blue-600 text-white' : 'bg-slate-700 text-slate-400'}}`}}>Breakouts Only</button>
                            </div>
                        </div>
                        {{sidebarOpen && <div className="absolute inset-0 bg-black/60 z-10 md:hidden" onClick={{() => setSidebarOpen(false)}}></div>}}
                        <div className="flex-1 bg-slate-900 overflow-y-auto p-3 space-y-3">
                            {{filtered.length === 0 && <div className="text-center text-slate-500 mt-10">No stocks found with current filter.</div>}}
                            {{filtered.map((stock, idx) => (
                                <div key={{idx}} onClick={{() => setSelectedStock(stock)}} className="bg-slate-800 rounded-xl p-4 border border-slate-700 active:bg-slate-700 flex flex-col gap-3 shadow-sm cursor-pointer">
                                    <div className="flex justify-between items-start">
                                        <div className="flex items-center gap-3">
                                            <div className="w-10 h-10 rounded-lg bg-slate-700 flex items-center justify-center text-xs font-bold text-blue-300">{{stock.symbol.substring(0, 2)}}</div>
                                            <div>
                                                <div className="font-bold text-white">{{stock.symbol}}</div>
                                                <div className="flex items-center gap-2">
                                                    <span className="text-xs text-slate-400">₹{{stock.price}}</span>
                                                    <span className={{`text-xs font-bold px-1.5 py-0.5 rounded ${{parseFloat(stock.change) >= 0 ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'}}`}}>{{parseFloat(stock.change) > 0 ? '+' : ''}}{{stock.change}}%</span>
                                                </div>
                                            </div>
                                        </div>
                                        <div className={{`px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wide ${{stock.rsi < 30 ? 'bg-red-500/20 text-red-400' : 'bg-slate-700 text-slate-400'}}`}}>RSI {{stock.rsi}}</div>
                                    </div>
                                    <div className="grid grid-cols-2 gap-4 items-center">
                                        <div className="flex flex-wrap gap-1.5">
                                            {{stock.patterns.length > 0 ? stock.patterns.map((p, i) => (<span key={{i}} className="text-[10px] uppercase font-bold tracking-wide bg-blue-900/30 text-blue-300 border border-blue-500/20 px-1.5 py-0.5 rounded">{{p}}</span>)) : <span className="text-[10px] text-slate-600">No Signal</span>}}
                                        </div>
                                        <div className="h-8 w-full"><Sparkline data={{stock.history}} color={{parseFloat(stock.change) >= 0 ? '#4ade80' : '#f87171'}} /></div>
                                    </div>
                                </div>
                            ))}}
                        </div>
                    </div>
                    {{selectedStock && (
                        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
                            <div className="bg-slate-900 w-full max-w-lg rounded-2xl flex flex-col overflow-hidden shadow-2xl border border-slate-700 max-h-[90vh]">
                                <div className="p-4 border-b border-slate-700 bg-slate-800 flex justify-between items-center">
                                    <h2 className="text-xl font-bold text-white">{{selectedStock.symbol}}</h2>
                                    <button onClick={{() => setSelectedStock(null)}} className="p-2 bg-slate-700 rounded-full"><X size={{20}} /></button>
                                </div>
                                <div className="flex-1 overflow-y-auto p-5 space-y-5">
                                    <div className="flex justify-between items-end">
                                        <div className="text-4xl font-mono text-white font-light">₹{{selectedStock.price}}</div>
                                        <div className={{`text-xl font-mono font-bold ${{parseFloat(selectedStock.change) >= 0 ? 'text-green-400' : 'text-red-400'}}`}}>{{selectedStock.change}}%</div>
                                    </div>
                                    <div className="bg-slate-800/40 p-4 rounded-xl border border-slate-700/50"><div className="h-32 flex items-end w-full"><Sparkline data={{selectedStock.history}} color="#3b82f6" /></div></div>
                                    <div className="grid grid-cols-2 gap-3">
                                        <div className="bg-slate-800 p-3 rounded-lg border border-slate-700"><div className="text-xs text-slate-500">RSI</div><div className="text-lg font-bold text-white">{{selectedStock.rsi}}</div></div>
                                        <div className="bg-slate-800 p-3 rounded-lg border border-slate-700"><div className="text-xs text-slate-500">SMA 50</div><div className="text-lg font-bold text-white">{{selectedStock.sma50}}</div></div>
                                    </div>
                                    <div className="bg-blue-900/10 border border-blue-500/30 rounded-xl p-4">
                                        <h3 className="text-sm font-bold text-blue-300 mb-3">Analysis</h3>
                                        <div className="space-y-2">{{Object.entries(selectedStock.details).map(([key, value], idx) => (<div key={{idx}} className="flex flex-col text-sm border-b border-blue-500/10 pb-2"><span className="font-bold text-blue-200 text-xs uppercase">{{key}}</span><span className="text-slate-300 text-xs">{{value}}</span></div>))}}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}}
                </div>
            );
        }};
        ReactDOM.createRoot(document.getElementById('root')).render(<App />);
    </script>
</body>
</html>
    """
    
    # Ensure directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with open(DASHBOARD_FILENAME, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Dashboard generated at: {DASHBOARD_FILENAME}")

def run_job():
    print("Starting GitHub Actions Scan...")
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(analyze_stock, sym): sym for sym in STOCKS}
        for future in futures:
            res = future.result()
            if res: results.append(res)
    generate_dashboard(results)

if __name__ == "__main__":
    run_job()

import yfinance as yf
import pandas as pd
import json
import os
import time
import requests
import io

# --- CONFIGURATION ---
OUTPUT_DIR = "public"
FILE_PATH = os.path.join(OUTPUT_DIR, "index.html")

# --- FETCH NIFTY 500 SYMBOLS ---
def get_nifty500_list():
    """
    Fetches the Nifty 500 list from a public CSV or uses a fallback list.
    """
    try:
        # Using a stable GitHub raw URL for Nifty 500 (Backup of NSE list)
        url = "https://raw.githubusercontent.com/sunilbaba/BreakoutScanners/main/ind_nifty500list.csv"
        # If you haven't uploaded the CSV yet, we use a hardcoded fallback for Top 50 to ensure it works
        
        # NOTE: For now, let's use a generated list of top 100 + common ones to ensure it runs immediately
        # without you needing to upload a CSV first.
        # You can expand this list or uncomment the CSV reading part later.
        
        tickers = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS",
            "LICI.NS", "HINDUNILVR.NS", "LT.NS", "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS",
            "ADANIENT.NS", "KOTAKBANK.NS", "TITAN.NS", "ONGC.NS", "TATAMOTORS.NS", "NTPC.NS", "AXISBANK.NS",
            "ULTRACEMCO.NS", "ADANIPORTS.NS", "POWERGRID.NS", "M&M.NS", "WIPRO.NS", "BAJAJFINSV.NS", "JIOFIN.NS",
            "HAL.NS", "DLF.NS", "ZOMATO.NS", "COALINDIA.NS", "SIEMENS.NS", "SBILIFE.NS", "IOC.NS", "VBL.NS",
            "TATASTEEL.NS", "GRASIM.NS", "HINDALCO.NS", "EICHERMOT.NS", "JSWSTEEL.NS", "BEL.NS", "INDUSINDBK.NS",
            "DRREDDY.NS", "BPCL.NS", "ADANIPOWER.NS", "DMART.NS", "ASIANPAINT.NS", "TRENT.NS", "RECLTD.NS",
            "LTIM.NS", "DIVISLAB.NS", "CIPLA.NS", "GAIL.NS", "AMBUJACEM.NS", "PFC.NS", "BANKBARODA.NS",
            "VEDL.NS", "TATAPOWER.NS", "GODREJCP.NS", "BRITANNIA.NS", "INDIGO.NS", "DABUR.NS", "ABB.NS",
            "LODHA.NS", "HAVELLS.NS", "PNB.NS", "UNIONBANK.NS", "CANBK.NS", "IDFCFIRSTB.NS", "IRFC.NS",
            "SHRIRAMFIN.NS", "CHOLAFIN.NS", "TVSMOTOR.NS", "MOTHERSON.NS", "HEROMOTOCO.NS", "BOSCHLTD.NS",
            "ASHOKLEY.NS", "MRF.NS", "APOLLOTYRES.NS", "BALKRISIND.NS", "CUMMINSIND.NS", "BHARATFORG.NS",
            "ASTRAL.NS", "POLYCAB.NS", "SUPREMEIND.NS", "PIIND.NS", "UPL.NS", "NAVINFLUOR.NS", "SRF.NS",
            "DEEPAKNTR.NS", "TATACHEM.NS", "COROMANDEL.NS", "LALPATHLAB.NS", "METROPOLIS.NS", "SYNGENE.NS",
            "APOLLOHOSP.NS", "MAXHEALTH.NS", "FORTIS.NS", "NHPC.NS", "SJVN.NS", "TORNTPOWER.NS", "SUZLON.NS"
        ]
        return list(set(tickers)) # Remove duplicates
    except Exception as e:
        print(f"Error fetching list: {e}")
        return ["RELIANCE.NS", "TCS.NS"] # Minimal fallback

# --- ANALYSIS ENGINE ---
def analyze_stock(ticker):
    """
    Fetches data and returns a dict with details OR error info.
    """
    try:
        # 1. Fetch Data with Rate Limiting
        # GitHub Actions IPs are often flagged, so we must be gentle.
        time.sleep(0.5) # Wait 0.5s between stocks
        
        stock = yf.Ticker(ticker)
        # Fetch 6 months of data
        df = stock.history(period="6mo", interval="1d")
        
        if df.empty:
            return {"symbol": ticker, "status": "failed", "error": "No data returned from Yahoo"}
        
        if len(df) < 50:
            return {"symbol": ticker, "status": "failed", "error": "Insufficient data points"}

        # 2. Calculate Indicators
        close = df['Close']
        volume = df['Volume']
        
        curr_price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])
        change_pct = round(((curr_price - prev_price) / prev_price) * 100, 2)
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        curr_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

        # Moving Averages
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        curr_sma50 = float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else 0.0
        curr_sma200 = float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else 0.0
        
        # Volume Avg
        vol_avg = volume.rolling(20).mean()
        curr_vol = float(volume.iloc[-1])
        curr_vol_avg = float(vol_avg.iloc[-1]) if not pd.isna(vol_avg.iloc[-1]) else 0.0

        # 3. Detect Patterns
        patterns = []
        details = {}
        
        # RSI Oversold
        if curr_rsi < 30: 
            patterns.append("RSI Oversold")
            details["RSI"] = f"{round(curr_rsi,1)} (Oversold)"
            
        # SMA Breakout (Price crossed above SMA 50)
        if prev_price < curr_sma50 and curr_price > curr_sma50:
            patterns.append("SMA 50 Breakout")
            details["SMA"] = "Crossed above 50-Day Avg"
            
        # Volume Spike
        if curr_vol > (curr_vol_avg * 2):
            patterns.append("Volume Spike")
            details["Volume"] = "High Buying Activity"

        return {
            "symbol": ticker.replace(".NS", ""),
            "status": "success",
            "price": round(curr_price, 2),
            "change": change_pct,
            "rsi": round(curr_rsi, 1),
            "patterns": patterns,
            "details": details,
            "history": [x if not pd.isna(x) else 0 for x in close.tail(30).tolist()]
        }

    except Exception as e:
        return {"symbol": ticker, "status": "failed", "error": str(e)}

# --- GENERATE HTML ---
def generate_dashboard(results):
    
    # Separate Success and Failures
    successful_stocks = [r for r in results if r['status'] == 'success']
    failed_stocks = [r for r in results if r['status'] == 'failed']
    
    scan_meta = {
        "total": len(results),
        "success": len(successful_stocks),
        "failed": len(failed_stocks),
        "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
    }

    print(f"Generating Dashboard: {len(successful_stocks)} Success, {len(failed_stocks)} Failed")

    # JSON Injection
    data_json = json.dumps({"stocks": successful_stocks, "logs": failed_stocks, "meta": scan_meta})

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
<body class="bg-slate-900 text-white overflow-hidden">
    <div id="root" class="h-screen w-full"></div>

    <script type="text/babel">
        const RAW_DATA = {data_json};
        const {{ useState, useEffect }} = React;
        const {{ Activity, AlertCircle, CheckCircle, Search, Menu, X, Filter }} = lucide;

        const App = () => {{
            const [view, setView] = useState('stocks'); // 'stocks' or 'logs'
            const [filter, setFilter] = useState('ALL');
            const [stocks, setStocks] = useState(RAW_DATA.stocks);
            const [filteredStocks, setFilteredStocks] = useState(RAW_DATA.stocks);
            
            useEffect(() => {{
                if(window.lucide) window.lucide.createIcons();
            }});

            useEffect(() => {{
                if(filter === 'ALL') setFilteredStocks(stocks);
                else setFilteredStocks(stocks.filter(s => s.patterns.length > 0));
            }}, [filter, stocks]);

            const Sparkline = ({{ data, color }}) => {{
                if(!data || data.length === 0) return null;
                const min = Math.min(...data); const max = Math.max(...data); 
                const points = data.map((d, i) => {{
                    const x = (i / (data.length - 1)) * 100;
                    const y = 30 - ((d - min) / (max - min || 1)) * 30;
                    return `${{x}},${{y}}`;
                }}).join(' ');
                return <svg width="100%" height="30" viewBox="0 0 100 30" preserveAspectRatio="none"><polyline points={{points}} fill="none" stroke={{color}} strokeWidth="2" /></svg>;
            }};

            return (
                <div className="flex flex-col h-full max-w-lg mx-auto md:max-w-4xl bg-slate-900">
                    {{/* HEADER */}}
                    <div className="p-4 bg-slate-800 border-b border-slate-700 flex justify-between items-center shadow-lg z-10">
                        <div>
                            <h1 className="font-bold text-xl flex items-center gap-2">
                                <Activity className="text-blue-500" /> NiftyScan
                            </h1>
                            <p className="text-[10px] text-slate-400">Updated: {{RAW_DATA.meta.timestamp}}</p>
                        </div>
                        <div className="flex gap-2 text-xs">
                             <button onClick={{() => setView('stocks')}} className={{`px-3 py-1 rounded ${{view==='stocks' ? 'bg-blue-600' : 'bg-slate-700'}}`}}>
                                Stocks ({{RAW_DATA.meta.success}})
                             </button>
                             <button onClick={{() => setView('logs')}} className={{`px-3 py-1 rounded flex items-center gap-1 ${{view==='logs' ? 'bg-red-600' : 'bg-slate-700'}}`}}>
                                <AlertCircle size={{12}} /> Failed ({{RAW_DATA.meta.failed}})
                             </button>
                        </div>
                    </div>

                    {{/* CONTROLS */}}
                    {{view === 'stocks' && (
                        <div className="p-2 bg-slate-800/50 flex gap-2 overflow-x-auto">
                            <button onClick={{() => setFilter('ALL')}} className={{`px-3 py-1 rounded text-xs font-bold whitespace-nowrap ${{filter==='ALL' ? 'bg-blue-600' : 'bg-slate-700 text-slate-400'}}`}}>All Stocks</button>
                            <button onClick={{() => setFilter('BREAKOUT')}} className={{`px-3 py-1 rounded text-xs font-bold whitespace-nowrap ${{filter==='BREAKOUT' ? 'bg-blue-600' : 'bg-slate-700 text-slate-400'}}`}}>Breakouts Only</button>
                        </div>
                    )}}

                    {{/* MAIN CONTENT AREA */}}
                    <div className="flex-1 overflow-y-auto p-3 space-y-3 bg-slate-900">
                        
                        {{/* STOCKS VIEW */}}
                        {{view === 'stocks' && filteredStocks.map((stock, i) => (
                            <div key={{i}} className="bg-slate-800 p-4 rounded-xl border border-slate-700 shadow-sm">
                                <div className="flex justify-between items-start">
                                    <div>
                                        <div className="font-bold text-lg text-white">{{stock.symbol}}</div>
                                        <div className="flex items-center gap-2">
                                            <span className="text-slate-400 text-sm">₹{{stock.price}}</span>
                                            <span className={{`text-xs font-bold px-1.5 py-0.5 rounded ${{stock.change >= 0 ? 'text-green-400 bg-green-900/20' : 'text-red-400 bg-red-900/20'}}`}}>
                                                {{stock.change}}%
                                            </span>
                                        </div>
                                    </div>
                                    <div className={{`px-2 py-1 rounded text-xs font-bold ${{stock.rsi < 30 ? 'bg-red-500 text-white' : 'bg-slate-700 text-slate-400'}}`}}>
                                        RSI {{stock.rsi}}
                                    </div>
                                </div>

                                {{/* Patterns & Chart */}}
                                <div className="mt-3 grid grid-cols-2 gap-4 items-center">
                                    <div className="flex flex-wrap gap-1">
                                        {{stock.patterns.length > 0 ? stock.patterns.map(p => (
                                            <span key={{p}} className="text-[10px] uppercase font-bold bg-blue-900/40 text-blue-200 border border-blue-500/30 px-1.5 py-0.5 rounded">
                                                {{p}}
                                            </span>
                                        )) : <span className="text-[10px] text-slate-600 italic">No patterns detected</span>}}
                                    </div>
                                    <div className="h-8 w-full opacity-80">
                                        <Sparkline data={{stock.history}} color={{stock.change >= 0 ? '#4ade80' : '#f87171'}} />
                                    </div>
                                </div>
                            </div>
                        ))}}

                        {{/* LOGS VIEW */}}
                        {{view === 'logs' && (
                            <div className="space-y-2">
                                <div className="text-xs text-slate-500 uppercase font-bold px-1">Error Report</div>
                                {{RAW_DATA.logs.map((log, i) => (
                                    <div key={{i}} className="bg-slate-800/50 p-3 rounded border border-red-900/30 flex justify-between items-center">
                                        <span className="font-mono text-sm text-red-200">{{log.symbol}}</span>
                                        <span className="text-xs text-red-400">{{log.error}}</span>
                                    </div>
                                ))}}
                                {{RAW_DATA.logs.length === 0 && <div className="text-center text-slate-500 p-4">No errors! All stocks scanned successfully.</div>}}
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

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    with open(FILE_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"SUCCESS: Dashboard generated at {FILE_PATH}")

if __name__ == "__main__":
    print("Starting Nifty 500 Scan...")
    
    # 1. Get List
    tickers = get_nifty500_list()
    print(f"Loaded {len(tickers)} tickers to scan.")
    
    # 2. Scan sequentially to prevent blocking
    results = []
    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] Scanning {ticker}...", end="\r")
        data = analyze_stock(ticker)
        results.append(data)
        
    print("\nScan Complete. Generating Dashboard...")
    generate_dashboard(results)

"""
nifty_engine_gh.py

Institutional-style swing scanner + simulated execution using JSON persistence.
Features:
- ADX Trend Strength Filter
- Realistic Gap-Down Execution Logic
- Dynamic Risk Scaling based on Market Regime
- Volatility Parity Position Sizing
- JSON State Persistence
- "Smart Filtering": Hides stocks you already own from scanner.
- "Duplicate Guard": Prevents buying the same stock twice.

Requirements:
    pip install yfinance pandas numpy
"""

import os
import time
import json
import math
import logging
from datetime import datetime, date
from functools import wraps

import yfinance as yf
import pandas as pd
import numpy as np

# -------------------------
# 1. CONFIGURATION
# -------------------------
CAPITAL = 100_000.0               # Total Portfolio Capital (₹)
RISK_PER_TRADE = 0.01             # Standard Risk: 1% per trade
MAX_POSITION_PERC = 0.25          # Max 25% capital in one stock

DATA_PERIOD = "1y"
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 2.0

MIN_ADV_VALUE_RS = 2_000_000      # Min Avg Daily Value (Liquidity Filter)

# Transaction Costs
BROKERAGE_PER_ORDER = 20.0
BROKERAGE_PCT = 0.0005
STT_PCT = 0.001
EXCHANGE_FEES_PCT = 0.0000345
STAMP_DUTY_PCT = 0.00015
GST_PCT = 0.18
SLIPPAGE_PCT = 0.001

# Strategy Parameters
ATR_MULTIPLIER_TARGET = 3.0
ATR_MULTIPLIER_STOP = 1.0
TSL_MOVE_TO_BE_AT = 0.5           # Move SL to Breakeven at 50% to target
ADX_THRESHOLD = 25.0              # Min ADX for Momentum setups

# File Paths
OUTPUT_DIR = "public"
HTML_FILE = os.path.join(OUTPUT_DIR, "index.html")
TRADE_HISTORY_FILE = "trade_history.json"

SECTOR_INDICES = {
    "NIFTY 50": "^NSEI",
    "BANK": "^NSEBANK", "AUTO": "^CNXAUTO", "IT": "^CNXIT",
    "METAL": "^CNXMETAL", "PHARMA": "^CNXPHARMA", "FMCG": "^CNXFMCG",
    "ENERGY": "^CNXENERGY", "REALTY": "^CNXREALTY", "PSU BANK": "^CNXPSUBANK"
}

# Fallback Ticker List
DEFAULT_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LTIM.NS", "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS", "BAJFINANCE.NS", "HCLTECH.NS",
    "ADANIENT.NS", "TATASTEEL.NS", "JIOFIN.NS", "ZOMATO.NS", "DLF.NS",
    "HAL.NS", "TRENT.NS", "BEL.NS", "POWERGRID.NS", "ONGC.NS",
    "NTPC.NS", "COALINDIA.NS", "BPCL.NS", "WIPRO.NS"
]

# -------------------------
# 2. LOGGING & UTILS
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PrimeEngine")

def retry_on_exception(max_tries=3, backoff=2.0):
    def deco(func):
        @wraps(func)
        def inner(*a, **kw):
            last_exc = None
            for i in range(max_tries):
                try:
                    return func(*a, **kw)
                except Exception as e:
                    last_exc = e
                    wait = backoff * (2 ** i)
                    logger.warning(f"Retry {i+1}/{max_tries} for {func.__name__}: {e}")
                    time.sleep(wait)
            logger.error(f"All retries failed for {func.__name__}")
            raise last_exc
        return inner
    return deco

# -------------------------
# 3. TECHNICAL INDICATORS
# -------------------------
def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def wilder_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.ewm(alpha=1/period, adjust=False).mean()
    loss = down.ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), -minus_dm, 0.0)
    tr = true_range(df)
    atr_series = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr_series)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr_series)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.ewm(alpha=1/period, adjust=False).mean().fillna(0)

# -------------------------
# 4. DATA MANAGEMENT
# -------------------------
@retry_on_exception(max_tries=RETRY_ATTEMPTS, backoff=RETRY_BACKOFF)
def robust_download(tickers, period=DATA_PERIOD):
    logger.info(f"Downloading {len(tickers)} symbols...")
    df = yf.download(tickers, period=period, group_by='ticker', threads=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance returned empty data")
    return df

def get_tickers():
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        df = pd.read_csv(url)
        symbols = [f"{x}.NS" for x in df['Symbol'].dropna().unique()]
        logger.info(f"Fetched {len(symbols)} tickers from NSE.")
        return symbols
    except Exception as e:
        logger.warning(f"NSE fetch failed ({e}). Using default list.")
        return DEFAULT_TICKERS.copy()

def extract_stock_df(bulk_data, ticker):
    try:
        if isinstance(bulk_data.columns, pd.MultiIndex):
            if ticker in bulk_data.columns.get_level_values(0):
                return bulk_data[ticker].copy().dropna()
    except Exception: pass
    return None

# -------------------------
# 5. RISK & EXECUTION MATH
# -------------------------
def estimate_transaction_costs(price: float, qty: int):
    val = price * qty
    brokerage = min(BROKERAGE_PER_ORDER, BROKERAGE_PCT * val) 
    stt = STT_PCT * val
    exchange = EXCHANGE_FEES_PCT * val
    stamp = STAMP_DUTY_PCT * val
    gst = GST_PCT * (brokerage + exchange)
    slippage = SLIPPAGE_PCT * val
    return round(brokerage + stt + exchange + stamp + gst + slippage, 2)

def calculate_qty(entry, stop_loss, risk_per_trade=RISK_PER_TRADE):
    risk_amount = CAPITAL * risk_per_trade
    risk_per_share = abs(entry - stop_loss)
    if risk_per_share <= 0: return 0
    qty = int(risk_amount / risk_per_share)
    max_alloc = CAPITAL * MAX_POSITION_PERC
    if (qty * entry) > max_alloc: qty = int(max_alloc / entry)
    return max(qty, 0)

# -------------------------
# 6. STRATEGY ENGINE
# -------------------------
def analyze_market_trend(bulk_data):
    try:
        nifty = extract_stock_df(bulk_data, "^NSEI")
        if nifty is None or len(nifty) < 200: return "UNKNOWN"
        curr = nifty['Close'].iloc[-1]
        sma200 = nifty['Close'].rolling(200).mean().iloc[-1]
        sma50 = nifty['Close'].rolling(50).mean().iloc[-1]
        if curr > sma50 and sma50 > sma200: return "BULL MARKET 🟢"
        if curr > sma200: return "UPTREND 🟡"
        return "BEAR MARKET 🔴"
    except: return "UNKNOWN"

def get_stock_sector(symbol):
    s = symbol.replace('.NS', '')
    if s in ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK"]: return "BANK"
    if s in ["TCS", "INFY", "HCLTECH", "WIPRO", "LTIM"]: return "IT"
    if s in ["MARUTI", "M&M", "TATAMOTORS", "BAJAJ-AUTO"]: return "AUTO"
    if s in ["TATASTEEL", "JINDALSTEL", "HINDALCO", "VEDL"]: return "METAL"
    if s in ["SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB"]: return "PHARMA"
    if s in ["ITC", "HINDUNILVR", "NESTLEIND"]: return "FMCG"
    return "Other"

def generate_signal(ticker, df, sector_changes, market_regime):
    if df is None or len(df) < 50: return None
    df = df.copy()
    
    # Indicators
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['ATR'] = atr(df, 14)
    df['RSI'] = wilder_rsi(df['Close'], 14)
    df['ADX'] = adx(df, 14)

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(curr['Close'])
    if math.isnan(close): return None

    sma200 = float(curr['SMA200'])
    trend = "UP" if close > sma200 else "DOWN"
    trend_strength = float(curr['ADX'])
    
    setups = []
    # Momentum
    if close > float(curr['EMA20']) and curr['RSI'] > 60 and trend_strength > ADX_THRESHOLD:
        setups.append("Momentum Burst")
    # Pullback
    sma50 = float(curr['SMA50'])
    if trend == "UP" and close > sma50 and abs(close - sma50)/close < 0.03 and curr['RSI'] < 60:
        setups.append("Pullback")

    # Liquidity Filter
    avg_vol = df['Volume'].rolling(30).mean().iloc[-1]
    if (avg_vol * close) < MIN_ADV_VALUE_RS: return None

    # Sector
    clean_sym = ticker.replace(".NS", "")
    my_sector = get_stock_sector(clean_sym)
    has_sector = sector_changes.get(my_sector, 0) > 0.5
    if has_sector: setups.append("Sector Support")

    # Targets
    atr_val = float(curr['ATR'])
    stop = round(close - ATR_MULTIPLIER_STOP * atr_val, 2)
    target = round(close + ATR_MULTIPLIER_TARGET * atr_val, 2)
    rr = round((target - close) / (close - stop), 2)

    # Risk Scale
    adjusted_risk = RISK_PER_TRADE
    if "BEAR" in market_regime:
        adjusted_risk = RISK_PER_TRADE / 2
        setups.append("⚠️ Half Size")

    qty = calculate_qty(close, stop, adjusted_risk)

    verdict = "WAIT"
    v_color = "gray"
    if trend == "UP" and len(setups) > 0:
        if rr >= 2.0:
            verdict = "PRIME BUY ⭐" if has_sector else "BUY"
            v_color = "purple" if has_sector else "green"
        else:
            verdict = "WATCH"

    if verdict == "WAIT" and abs((close - float(prev['Close']))/float(prev['Close'])) < 0.01:
        return None

    return {
        "symbol": clean_sym,
        "price": round(close, 2),
        "change": round(((close - float(prev['Close']))/float(prev['Close']))*100, 2),
        "sector": my_sector,
        "setups": setups,
        "verdict": verdict,
        "v_color": v_color,
        "rr": rr,
        "qty": qty,
        "adx": round(trend_strength, 1),
        "investment": round(qty * close, 0),
        "levels": {"TGT": target, "SL": stop},
        "history": df['Close'].tail(30).tolist()
    }

# -------------------------
# 7. PORTFOLIO & EXECUTION
# -------------------------
def load_json_file(path, default):
    if not os.path.exists(path): return default
    try:
        with open(path, 'r') as f: return json.load(f)
    except: return default

def save_json_file(path, data):
    with open(path, 'w') as f: json.dump(data, f, indent=2)

def update_open_trades_json(bulk_data):
    trades = load_json_file(TRADE_HISTORY_FILE, [])
    updated = False
    today_str = date.today().isoformat()
    
    for trade in trades:
        if trade['status'] != 'OPEN': continue
        
        ticker = trade['symbol'] + ".NS"
        df = extract_stock_df(bulk_data, ticker)
        if df is None: continue
        
        curr = df.iloc[-1]
        curr_open = float(curr['Open'])
        high = float(curr['High'])
        low = float(curr['Low'])
        close = float(curr['Close'])
        
        entry = float(trade['entry'])
        target = float(trade['target'])
        sl = float(trade['stop_loss'])
        
        # Stop Loss
        if low <= sl:
            trade['status'] = 'LOSS'
            exit_price = curr_open if curr_open < sl else sl
            trade['exit_price'] = exit_price
            trade['exit_date'] = today_str
            pnl = (exit_price - entry) * trade['qty']
            costs = estimate_transaction_costs(entry, trade['qty']) + estimate_transaction_costs(exit_price, trade['qty'])
            trade['net_pnl'] = round(pnl - costs, 2)
            trade['note'] = "SL Hit" if curr_open >= sl else "SL Gap Down"
            updated = True
            logger.info(f"Trade {trade['symbol']} stopped out.")

        # Target
        elif high >= target:
            trade['status'] = 'WIN'
            exit_price = curr_open if curr_open > target else target
            trade['exit_price'] = exit_price
            trade['exit_date'] = today_str
            pnl = (exit_price - entry) * trade['qty']
            costs = estimate_transaction_costs(entry, trade['qty']) + estimate_transaction_costs(exit_price, trade['qty'])
            trade['net_pnl'] = round(pnl - costs, 2)
            trade['note'] = "Target Hit"
            updated = True
            logger.info(f"Trade {trade['symbol']} won.")

        # Active Update
        else:
            # TSL Logic
            move_needed = (target - entry) * TSL_MOVE_TO_BE_AT
            if close > (entry + move_needed) and sl < entry:
                trade['stop_loss'] = entry
                trade['note'] = "TSL @ BE"
                updated = True
            
            m2m = (close - entry) * trade['qty']
            trade['pnl'] = round(m2m, 2)
            trade['pnl_pct'] = round(((close - entry)/entry)*100, 2)
            updated = True

    if updated: save_json_file(TRADE_HISTORY_FILE, trades)

def place_orders(signals):
    trades = load_json_file(TRADE_HISTORY_FILE, [])
    today_str = date.today().isoformat()
    
    # 1. IDENTIFY EXISTING OPEN SYMBOLS
    # This prevents buying TATASTEEL if we already own TATASTEEL
    open_symbols = {t['symbol'] for t in trades if t['status'] == 'OPEN'}
    
    prime_signals = [s for s in signals if "BUY" in s['verdict']]
    prime_signals.sort(key=lambda x: x['rr'], reverse=True)
    
    for s in prime_signals[:5]:
        # 2. CHECK IF ALREADY OWNED
        if s['symbol'] in open_symbols:
            logger.info(f"Skipping {s['symbol']} - Already in portfolio.")
            continue
            
        tid = f"{s['symbol']}-{today_str}"
        # Check if trade ID exists (prevent dupes same day)
        if not any(t['id'] == tid for t in trades):
            new_trade = {
                "id": tid, "date": today_str, "symbol": s['symbol'],
                "entry": s['price'], "qty": s['qty'],
                "target": s['levels']['TGT'], "stop_loss": s['levels']['SL'],
                "status": "OPEN", "pnl": 0.0, "pnl_pct": 0.0, "note": ""
            }
            trades.insert(0, new_trade)
            open_symbols.add(s['symbol']) # Update local set
            save_json_file(TRADE_HISTORY_FILE, trades)
            logger.info(f"Placed Order: {s['symbol']}")

# -------------------------
# 8. HTML REPORT
# -------------------------
def generate_html(signals, trades, market_regime, timestamp):
    closed_trades = [t for t in trades if t['status'] in ['WIN', 'LOSS']]
    wins = len([t for t in closed_trades if t['status'] == 'WIN'])
    total = len(closed_trades)
    acc = round((wins/total*100), 1) if total > 0 else 0
    total_pnl = sum([t.get('net_pnl', 0) for t in closed_trades])
    
    # FILTER: Don't show stocks in "Scanner" if they are already in "Portfolio"
    open_symbols = {t['symbol'] for t in trades if t['status'] == 'OPEN'}
    filtered_signals = [s for s in signals if s['symbol'] not in open_symbols]
    
    json_data = json.dumps({"stocks": filtered_signals, "pos": trades})
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Institutional PrimeTrade</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            body {{ background: #0f172a; color: #e2e8f0; font-family: sans-serif; }}
            .card {{ background: #1e293b; border: 1px solid #334155; padding: 16px; border-radius: 12px; }}
            .prime {{ border: 1px solid #a855f7; background: linear-gradient(to bottom right, #1e293b, #2e1065); }}
            .badge {{ font-size: 10px; padding: 2px 6px; border-radius: 4px; font-weight: bold; margin-right: 4px; }}
            .ledger-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
            .ledger-table th {{ text-align: left; padding: 8px; color: #64748b; border-bottom: 1px solid #334155; }}
            .ledger-table td {{ padding: 8px; border-bottom: 1px solid #1e293b; }}
        </style>
    </head>
    <body class="p-4 md:p-8">
        <div class="max-w-7xl mx-auto">
            <div class="flex justify-between items-center mb-6 bg-slate-900 p-4 rounded-xl border border-slate-800">
                <div>
                    <h1 class="text-2xl font-bold text-white flex items-center gap-2">
                        <i data-lucide="building-2" class="text-purple-500"></i> PrimeTrade Institutional
                    </h1>
                    <div class="text-xs text-slate-500 mt-1">{timestamp} • {market_regime}</div>
                </div>
                <div class="text-right">
                    <div class="text-[10px] text-slate-500 uppercase">Net PnL</div>
                    <div class="text-xl font-bold { 'text-green-400' if total_pnl >=0 else 'text-red-400' }">₹{total_pnl}</div>
                    <div class="text-[10px] text-slate-600">Win Rate: {acc}% ({total} Trades)</div>
                </div>
            </div>

            <h2 class="text-xs font-bold text-slate-500 mb-3 uppercase">Active Positions</h2>
            <div id="portfolio" class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8"></div>

            <h2 class="text-xs font-bold text-slate-500 mb-3 uppercase">New Opportunities</h2>
            <div id="scanner" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8"></div>
            
            <h2 class="text-xs font-bold text-slate-500 mb-3 uppercase">Trade Ledger (History)</h2>
            <div class="bg-slate-900 rounded-lg border border-slate-800 overflow-hidden mb-8">
                <div class="overflow-x-auto">
                    <table class="ledger-table">
                        <thead><tr><th>Date</th><th>Symbol</th><th>Entry</th><th>Exit</th><th>Qty</th><th>PnL</th><th>Status</th></tr></thead>
                        <tbody id="ledger-body"></tbody>
                    </table>
                </div>
            </div>
        </div>

        <script>
            const DATA = {json_data};
            
            // Portfolio
            const portRoot = document.getElementById('portfolio');
            const openTrades = DATA.pos.filter(t => t.status === 'OPEN');
            if(openTrades.length === 0) portRoot.innerHTML = '<div class="col-span-full text-center text-slate-600 py-4">No active trades.</div>';
            else {{
                portRoot.innerHTML = openTrades.map(p => {{
                    const pnlClass = p.pnl_pct >= 0 ? 'text-green-400' : 'text-red-400';
                    return `<div class="bg-slate-800 border border-slate-700 rounded-lg p-4 relative">
                        <div class="flex justify-between mb-2"><div class="font-bold text-white">${{p.symbol}}</div><div class="font-mono font-bold ${{pnlClass}}">${{p.pnl_pct}}%</div></div>
                        <div class="text-xs text-slate-400 flex justify-between"><span>Entry: ${{p.entry}}</span><span>Qty: ${{p.qty}}</span></div>
                        <div class="text-[10px] text-slate-500 mt-1">Date: ${{p.date}}</div>
                        <div class="flex justify-between text-[10px] mt-2 font-mono"><span class="text-red-400">${{p.stop_loss}} SL</span><span class="text-green-400">${{p.target}} TGT</span></div>
                        ${{p.note ? `<div class="absolute top-0 right-0 bg-blue-600 text-[9px] text-white px-2 py-0.5 rounded-bl">${{p.note}}</div>` : ''}}
                    </div>`;
                }}).join('');
            }}

            // Scanner
            const scanRoot = document.getElementById('scanner');
            if(DATA.stocks.length === 0) scanRoot.innerHTML = '<div class="col-span-full text-center text-slate-600 py-10">No new signals.</div>';
            else {{
                scanRoot.innerHTML = DATA.stocks.map(s => {{
                    const isPrime = s.verdict.includes('PRIME');
                    const badges = s.setups.map(b => `<span class="badge bg-slate-700 text-slate-300">${{b}}</span>`).join('');
                    return `<div class="card ${{isPrime ? 'prime' : ''}}">
                        <div class="flex justify-between mb-2"><div><div class="font-bold text-white">${{s.symbol}}</div><div class="text-[10px] text-slate-400">${{s.sector}}</div></div><div class="text-right"><div class="font-bold ${{s.change>=0?'text-green-400':'text-red-400'}}">${{s.change}}%</div><div class="text-[10px] text-slate-500">₹${{s.price}}</div></div></div>
                        <div class="mb-3 h-6 flex flex-wrap">${{badges}}</div>
                        <div class="flex justify-between text-[10px] bg-slate-900/50 p-2 rounded mb-2"><div class="text-center"><div>Qty</div><div class="text-blue-400 font-bold">${{s.qty}}</div></div><div class="text-center"><div>ADX</div><div class="text-white font-bold">${{s.adx}}</div></div><div class="text-center"><div>RR</div><div class="text-white font-bold">${{s.rr}}</div></div></div>
                        <div class="flex justify-between text-[10px] font-mono mt-1"><span class="text-red-400">SL: ${{s.levels.SL}}</span><span class="text-green-400">TGT: ${{s.levels.TGT}}</span></div>
                    </div>`;
                }}).join('');
            }}
            
            // Ledger
            const ledgerRoot = document.getElementById('ledger-body');
            const closedTrades = DATA.pos.filter(t => t.status !== 'OPEN');
            if(closedTrades.length === 0) ledgerRoot.innerHTML = '<tr><td colspan="7" class="text-center text-slate-600 py-4">No closed trades yet.</td></tr>';
            else {{
                ledgerRoot.innerHTML = closedTrades.map(t => {{
                    const statusClass = t.status === 'WIN' ? 'text-green-400' : 'text-red-400';
                    return `<tr><td class="text-slate-400">${{t.date}}</td><td class="font-bold text-white">${{t.symbol}}</td><td>${{t.entry}}</td><td>${{t.exit_price}}</td><td>${{t.qty}}</td><td class="font-mono font-bold ${{statusClass}}">₹${{t.net_pnl}}</td><td><span class="px-2 py-0.5 rounded text-[10px] font-bold bg-opacity-20 ${{t.status==='WIN'?'bg-green-500 text-green-400':'bg-red-500 text-red-400'}}">${{t.status}}</span></td></tr>`;
                }}).join('');
            }}
            lucide.createIcons();
        </script>
    </body>
    </html>
    """
    os.makedirs(os.path.dirname(HTML_FILE), exist_ok=True)
    with open(HTML_FILE, "w", encoding="utf-8") as f: f.write(html)

# -------------------------
# 9. MAIN EXECUTION
# -------------------------
def run_once():
    logger.info("--- INSTITUTIONAL ENGINE START ---")
    tickers = get_tickers()
    all_symbols = tickers + list(SECTOR_INDICES.values())
    
    try:
        bulk = robust_download(all_symbols)
    except Exception as e:
        logger.error(f"Critical Data Failure: {e}")
        return

    sector_changes = {}
    for name, t in SECTOR_INDICES.items():
        df = extract_stock_df(bulk, t)
        if df is not None and len(df)>1:
            s = df['Close']
            sector_changes[name] = round(((s.iloc[-1]-s.iloc[-2])/s.iloc[-2])*100, 2)

    market_regime = analyze_market_trend(bulk)
    logger.info(f"Market Regime: {market_regime}")

    results = []
    if isinstance(bulk.columns, pd.MultiIndex):
        cols = bulk.columns.get_level_values(0).unique()
    else:
        cols = bulk.columns

    for t in cols:
        if str(t).startswith('^'): continue
        df = extract_stock_df(bulk, t)
        try:
            res = generate_signal(t, df, sector_changes, market_regime)
            if res: results.append(res)
        except: continue

    place_orders(results) 
    update_open_trades_json(bulk) 
    
    trades = load_json_file(TRADE_HISTORY_FILE, [])
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    
    generate_html(results, trades, market_regime, timestamp)
    logger.info("Run Complete. Dashboard updated.")

if __name__ == "__main__":
    run_once()

#!/usr/bin/env python3
"""
backtest_runner.py — production thresholds edition
- Downloads 2y data for entire ticker list.
- Strategy filters: RSI >= 60, ADX >= 25, Close > SMA200.
- Realistic costs: brokerage + slippage, minimum order value.
- Robust ADX implementation and robust downloads.
- Outputs backtest_stats.json
"""
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime
from time import sleep

# --- CONFIG (production) ---
DATA_PERIOD = "2y"
CACHE_FILE = "backtest_stats.json"
CAPITAL = 100_000.0
RISK_PER_TRADE = 0.02          # fraction of equity risked per trade
BROKERAGE_PCT = 0.001          # brokerage % on traded value (round-trip modeled as entry+exit)
SLIPPAGE_PCT = 0.0005          # 0.05% slippage per trade
MAX_POSITION_PERC = 0.25       # max capital allocation per position (as a fraction of CAPITAL)
MIN_ORDER_VALUE = 2000         # minimum order value (INR) to avoid tiny micro-positions

# Data & performance
MIN_ROWS_REQUIRED = 150
BATCH_SIZE = 50                # download batches tuned for full Nifty500
DIAGNOSTIC = False             # set False for production runs

# Strategy thresholds (strict / production)
RSI_THRESHOLD = 60
ADX_THRESHOLD = 25
REQUIRE_SMA200 = True

# Other
INDICATOR_SAMPLE_TICKERS = 8

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Backtester")

SECTOR_INDICES = {
    "NIFTY 50": "^NSEI", "BANK": "^NSEBANK", "AUTO": "^CNXAUTO",
    "IT": "^CNXIT", "METAL": "^CNXMETAL", "PHARMA": "^CNXPHARMA",
    "FMCG": "^CNXFMCG", "ENERGY": "^CNXENERGY", "REALTY": "^CNXREALTY",
    "PSU BANK": "^CNXPSUBANK"
}

# -------------------------
# Data helpers
# -------------------------
def get_tickers():
    """
    Loads ticker list from ind_nifty500list.csv if present, else falls back to a reasonable default.
    CSV expected to have a 'Symbol' column with NSE symbols (without .NS)
    """
    if os.path.exists("ind_nifty500list.csv"):
        try:
            df = pd.read_csv("ind_nifty500list.csv")
            syms = [f"{x}.NS" for x in df['Symbol'].dropna().unique()]
            logger.info(f"Loaded {len(syms)} tickers from CSV")
            return syms
        except Exception as e:
            logger.warning("Failed to read CSV list, using defaults: %s", e)
    # safe defaults (will be replaced by CSV in production)
    return [
        "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "SBIN.NS",
        "HINDUNILVR.NS", "ICICIBANK.NS", "LT.NS", "AXISBANK.NS", "BHARTIARTL.NS"
    ]

def _tz_safe_index(df):
    df = df.copy()
    try:
        df.index = pd.to_datetime(df.index)
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_convert(None)
    except Exception:
        try:
            df.index = pd.to_datetime(df.index).tz_localize(None)
        except Exception:
            pass
    return df

def robust_download(tickers, period=DATA_PERIOD, batch_size=BATCH_SIZE):
    """
    Download tickers in batches. Fall back to per-symbol downloads if needed.
    Returns a combined DataFrame (multi-index columns when multiple tickers).
    """
    logger.info(f"📥 Downloading {len(tickers)} symbols ({period}) in batches of {batch_size}...")
    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        success = False
        for attempt in range(3):
            try:
                data = yf.download(batch, period=period, group_by='ticker', threads=False, progress=False, ignore_tz=True)
                if isinstance(data, pd.DataFrame) and not data.empty:
                    frames.append(data)
                    success = True
                    break
            except Exception as e:
                logger.debug("Batch download failed attempt %d for %s: %s", attempt+1, batch, e)
                sleep(1.0 * (attempt+1))
        if not success:
            logger.info("Falling back to per-symbol download for batch: %s", batch)
            for sym in batch:
                tries = 0
                while tries < 2:
                    try:
                        df = yf.download(sym, period=period, threads=False, progress=False, ignore_tz=True)
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            # single-symbol frames will have columns Open, High, Low, Close, Volume, Adj Close
                            # convert to multiindex-like structure by renaming columns for later concat consistency
                            df.columns = pd.Index(df.columns)
                            frames.append(df)
                            break
                    except Exception as e:
                        logger.debug("Single download fail for %s: %s", sym, e)
                        sleep(0.5)
                    tries += 1
    if not frames:
        logger.error("No data frames downloaded.")
        return pd.DataFrame()
    try:
        combined = pd.concat(frames, axis=1)
        return combined
    except Exception as e:
        logger.warning("Concat failed: %s - attempting to return first non-empty frame", e)
        for f in frames:
            if isinstance(f, pd.DataFrame) and not f.empty:
                return f
    return pd.DataFrame()

def extract_df(bulk, ticker):
    """
    Given bulk output (possibly multiindexed), return a cleaned df for 'ticker' (e.g., 'RELIANCE.NS' or '^NSEI').
    Ensures timezone-safe index and enforces MIN_ROWS_REQUIRED rows.
    """
    try:
        if bulk is None or bulk.empty:
            return None
        if isinstance(bulk.columns, pd.MultiIndex):
            level0 = list(bulk.columns.get_level_values(0).unique())
            if ticker in level0:
                df = bulk[ticker].copy().dropna()
            else:
                tbase = ticker.replace('.NS', '')
                if tbase in level0:
                    df = bulk[tbase].copy().dropna()
                else:
                    found = None
                    for cand in level0:
                        if cand.upper().startswith(tbase.upper()):
                            found = cand; break
                    if found:
                        df = bulk[found].copy().dropna()
                    else:
                        return None
            df = _tz_safe_index(df)
            if len(df) < MIN_ROWS_REQUIRED:
                logger.debug("Ticker %s has %d rows (<%d)", ticker, len(df), MIN_ROWS_REQUIRED)
                return None
            return df
        else:
            # single-frame (no multiindex) - maybe we downloaded per-ticker frames earlier
            if all(c in bulk.columns for c in ['Open', 'High', 'Low', 'Close']):
                df = bulk.copy().dropna()
                df = _tz_safe_index(df)
                if len(df) < MIN_ROWS_REQUIRED:
                    logger.debug("Single-frame data has insufficient rows: %d", len(df))
                    return None
                return df
    except Exception as e:
        logger.debug("extract_df error for %s: %s", ticker, e)
    return None

# -------------------------
# Indicators
# -------------------------
def true_range(df):
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df, period=14):
    return true_range(df).ewm(alpha=1/period, adjust=False).mean()

def wilder_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss)).fillna(50)

def adx(df, period=14):
    """
    Robust ADX implementation:
    - up_move = high.diff()
    - down_move = prev_low - low (positive when down)
    - plus_dm/minus_dm positive numbers only
    - uses ewm smoothing (Wilder-style)
    """
    high = df['High']
    low = df['Low']
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low   # positive when price moved down

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df)
    sm_tr = tr.ewm(alpha=1/period, adjust=False).mean()

    sm_plus = pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    sm_minus = pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()

    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * (sm_plus / sm_tr)
        minus_di = 100 * (sm_minus / sm_tr)

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = (abs(plus_di - minus_di) / denom * 100).fillna(0)
    adx_series = dx.ewm(alpha=1/period, adjust=False).mean().fillna(0)
    return adx_series

def prepare_df(df):
    df = df.copy()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['ATR'] = atr(df)
    df['RSI'] = wilder_rsi(df['Close'])
    df['ADX'] = adx(df)
    return df

# -------------------------
# Win-rate & portfolio sim (production)
# -------------------------
def calc_stock_win_rate(df):
    """
    Compute per-stock win-rate using production thresholds (last ~6 months window).
    """
    if df is None or len(df) < 150:
        return 0
    df = prepare_df(df)
    wins, total = 0, 0
    start = max(0, len(df) - 130)
    for i in range(start, len(df) - 10):
        row = df.iloc[i]
        if pd.isna(row.get('EMA20')) or pd.isna(row.get('RSI')) or pd.isna(row.get('ADX')):
            continue
        if row['Close'] > row['EMA20'] and row['RSI'] > RSI_THRESHOLD and row['ADX'] > ADX_THRESHOLD:
            sl = row['Close'] - row['ATR']
            tgt = row['Close'] + (3 * row['ATR'])
            outcome = "OPEN"
            for j in range(1, 15):
                if i + j >= len(df): break
                fut = df.iloc[i + j]
                if fut['Low'] <= sl:
                    outcome = "LOSS"; break
                if fut['High'] >= tgt:
                    outcome = "WIN"; break
            if outcome != "OPEN":
                total += 1
                if outcome == "WIN": wins += 1
    return round((wins / total) * 100, 1) if total > 0 else 0

def run_portfolio_sim(bulk_data, tickers):
    """
    Production-ready portfolio simulation with realistic fees/slippage and min order value check.
    """
    processed = {}
    for t in tickers:
        df = extract_df(bulk_data, t)
        if df is not None and len(df) >= MIN_ROWS_REQUIRED:
            processed[t] = prepare_df(df)
    logger.info("Processed tickers for sim: %d/%d", len(processed), len(tickers))
    if not processed:
        logger.error("No tickers processed - exiting sim.")
        return [], [], 0, 0, 0

    # unified date set (some tickers may miss dates)
    dates = sorted(list(set().union(*[d.index for d in processed.values()])))
    sim_dates = dates[150:]  # skip early rows where indicators are NaN

    cash = CAPITAL
    equity_curve = [CAPITAL]
    portfolio = []
    history = []

    for date in sim_dates:
        # --- EXITS ---
        active = []
        for trade in portfolio:
            sym = trade['symbol']
            if date not in processed[sym].index:
                active.append(trade); continue
            row = processed[sym].loc[date]
            exit_p = None
            # check open/low/high based exit logic
            if row['Open'] < trade['sl']:
                exit_p = row['Open']
            elif row['Low'] <= trade['sl']:
                exit_p = trade['sl']
            elif row['Open'] > trade['tgt']:
                exit_p = row['Open']
            elif row['High'] >= trade['tgt']:
                exit_p = trade['tgt']

            if exit_p:
                # apply slippage on exit price (worse for us): for sell we subtract slippage %
                exit_price_effective = exit_p * (1 - SLIPPAGE_PCT)
                revenue = exit_price_effective * trade['qty']
                cost = revenue * BROKERAGE_PCT
                # realized pnl considers entry price (which included slippage on entry) and entry_cost recorded
                pnl = revenue - cost - (trade['entry'] * trade['qty'] + trade['entry_cost'])
                self_pnl = pnl
                cash += (revenue - cost)
                history.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "symbol": sym,
                    "entry": round(trade['entry'], 2),
                    "exit": round(exit_price_effective, 2),
                    "pnl": round(self_pnl, 2),
                    "status": "WIN" if self_pnl > 0 else "LOSS"
                })
            else:
                active.append(trade)
        portfolio = active

        # --- ENTRIES ---
        if len(portfolio) < 5:
            # iterate symbols in deterministic order for reproducibility
            for sym in sorted(processed.keys()):
                if len(portfolio) >= 5: break
                df = processed[sym]
                if date not in df.index: continue
                row = df.loc[date]
                # require indicators to exist
                if pd.isna(row.get('EMA20')) or pd.isna(row.get('RSI')) or pd.isna(row.get('ADX')):
                    continue
                # strategy filters (production)
                if not (row['Close'] > row['EMA20'] and row['RSI'] > RSI_THRESHOLD and row['ADX'] > ADX_THRESHOLD):
                    continue
                if REQUIRE_SMA200 and (row['Close'] <= row['SMA200']):
                    continue
                # avoid duplicates
                if any(t['symbol'] == sym for t in portfolio): continue

                # position sizing based on ATR risk
                atr_val = row['ATR']
                if pd.isna(atr_val) or atr_val <= 0:
                    continue
                risk_per_share = atr_val
                risk_amount = equity_curve[-1] * RISK_PER_TRADE
                qty = int(risk_amount / risk_per_share)
                # enforce max position limit (based on initial CAPITAL)
                max_cost_allowed = CAPITAL * MAX_POSITION_PERC
                if qty * row['Close'] > max_cost_allowed:
                    qty = int(max_cost_allowed / row['Close'])
                # enforce minimum order value
                if qty * row['Close'] < MIN_ORDER_VALUE:
                    continue
                if qty <= 0:
                    continue

                # compute cost, fees and slippage for entry
                cost = qty * row['Close']
                est_fees = cost * BROKERAGE_PCT
                est_slippage = cost * SLIPPAGE_PCT
                total_required = cost + est_fees + est_slippage
                if cash < total_required:
                    continue

                # record entry effective price (including slippage worse for buyer)
                entry_price_effective = row['Close'] * (1 + SLIPPAGE_PCT)
                cash -= total_required
                portfolio.append({
                    "symbol": sym,
                    "entry": round(entry_price_effective, 4),
                    "qty": qty,
                    "sl": round(row['Close'] - atr_val, 4),
                    "tgt": round(row['Close'] + 3 * atr_val, 4),
                    "entry_cost": round(est_fees + est_slippage, 4)
                })

        # --- MTM / Equity ---
        m2m = 0.0
        for t in portfolio:
            sym = t['symbol']
            if date in processed[sym].index:
                price = processed[sym].loc[date]['Close']
            else:
                price = t['entry']
            m2m += (price * t['qty'])
        equity_curve.append(round(cash + m2m, 2))

    # summary
    wins = len([t for t in history if t['pnl'] > 0])
    win_rate = round((wins / len(history) * 100), 1) if history else 0
    profit = round(equity_curve[-1] - CAPITAL, 2)
    return equity_curve, history, win_rate, len(history), profit

# -------------------------
# Ledger/HTML helpers (kept minimal)
# -------------------------
def load_json_file(path):
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            pass
    return []

def save_json_file(path, data):
    with open(path, 'w') as f: json.dump(data, f, indent=2)

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    logger.info("Starting production backtest...")
    tickers = get_tickers()
    all_syms = tickers + list(SECTOR_INDICES.values())
    bulk = robust_download(all_syms, period=DATA_PERIOD, batch_size=BATCH_SIZE)

    # calculate per-ticker win rates
    logger.info("Calculating individual ticker win rates...")
    ticker_stats = {}
    for t in tickers:
        df = extract_df(bulk, t)
        if df is not None:
            try:
                ticker_stats[t.replace('.NS', '')] = calc_stock_win_rate(df)
            except Exception as e:
                logger.debug("Win rate calc failed for %s: %s", t, e)
                ticker_stats[t.replace('.NS', '')] = 0
        else:
            ticker_stats[t.replace('.NS', '')] = 0

    # run portfolio sim
    logger.info("Running portfolio simulation (production thresholds)...")
    curve, ledger, win_rate, trades, profit = run_portfolio_sim(bulk, tickers)

    output = {
        "updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "portfolio": {
            "curve": curve,
            "ledger": ledger[-50:],  # last 50 trades
            "win_rate": win_rate,
            "total_trades": trades,
            "profit": profit
        },
        "tickers": ticker_stats,
        "last_run": datetime.utcnow().strftime("%Y-%m-%d")
    }

    with open(CACHE_FILE, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"✅ Saved stats to {CACHE_FILE} (profit={profit}, trades={trades}, win_rate={win_rate}%)")

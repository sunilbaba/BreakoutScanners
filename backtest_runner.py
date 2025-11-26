#!/usr/bin/env python3
"""
backtest_runner.py
- Runs ONCE (on demand).
- Downloads 2 years of data for NIFTY500 (per-ticker robust download).
- Backtest: Smart-Strict (Option B) rules with A3 TSL.
- Output: Saves 'backtest_stats.json' with enhanced metrics.
"""

import os
import json
import logging
from datetime import datetime
import math
import time

import yfinance as yf
import pandas as pd
import numpy as np

# -------------------------
# CONFIG
# -------------------------
DATA_PERIOD = "2y"
CAPITAL = 100_000.0
RISK_PER_TRADE = 0.02
BROKERAGE_PCT = 0.001

# Strategy constants (Smart-Strict)
ADX_MIN = 18           # relaxed slightly from 20
RSI_MIN = 55           # relaxed from 60
VOLUME_SPIKE = 1.0     # allow >= 1.0 * VOL_SMA20 (breathing room)
BREAKOUT_LOOKBACK = 20
RR_MIN = 1.8

# A3 TSL
TSL_BE_R = 1.0
TSL_EMA20_R = 2.0
TSL_SWING_R = 3.0

MIN_HISTORY_ROWS = 250  # minimum rows for instrument to be considered

CACHE_FILE = "backtest_stats.json"

# Fallback default tickers if CSV not present (small sample)
DEFAULT_TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "SBIN.NS",
    "ICICIBANK.NS", "LT.NS", "AXISBANK.NS", "ITC.NS", "HINDUNILVR.NS"
]

# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BacktestA3")

# -------------------------
# INDICATORS
# -------------------------
def true_range(df):
    prev = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev).abs()
    tr3 = (df["Low"] - prev).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df, period=14):
    return true_range(df).ewm(alpha=1/period, adjust=False).mean()

def wilder_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_adx(df, period=14):
    up = df["High"].diff()
    dn = df["Low"].diff() * -1

    plus_dm = np.where((up > dn) & (up > 0), up, 0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0)

    tr = true_range(df)
    tr_smoothed = tr.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / tr_smoothed)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / tr_smoothed)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.ewm(alpha=1/period, adjust=False).mean().fillna(0)

def compute_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iat[i] > df["Close"].iat[i-1]:
            obv.append(obv[-1] + df["Volume"].iat[i])
        elif df["Close"].iat[i] < df["Close"].iat[i-1]:
            obv.append(obv[-1] - df["Volume"].iat[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

def prepare_df(df):
    """
    Expects df with Open/High/Low/Close/Volume (index = DatetimeIndex).
    Returns df with indicators and drops NaNs.
    """
    df = df.copy()
    # Price structure
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA20_SLOPE"] = df["EMA20"].diff()

    # Volatility
    df["ATR"] = atr(df)

    # Momentum
    df["RSI"] = wilder_rsi(df["Close"])
    df["ADX"] = compute_adx(df)

    # Volume & OBV
    df["VOL_SMA20"] = df["Volume"].rolling(20).mean()
    df["OBV"] = compute_obv(df)
    df["OBV_SMA"] = df["OBV"].rolling(20).mean()

    # Drop rows with NaNs produced by long rolling windows
    df.dropna(inplace=True)
    return df

# -------------------------
# WEEKLY
# -------------------------
def make_weekly(df):
    weekly = df.resample("W").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()
    if len(weekly) == 0:
        return weekly
    weekly["EMA20"] = weekly["Close"].ewm(span=20, adjust=False).mean()
    weekly["EMA20_SLOPE"] = weekly["EMA20"].diff()
    return weekly

# -------------------------
# BACKTEST ENGINE
# -------------------------
class BacktestEngine:
    def __init__(self, bulk_data, tickers):
        """
        bulk_data: MultiIndex DataFrame produced by pd.concat({ticker: df})
        tickers: list of tickers (with .NS) that we want to consider (keys of bulk_data)
        """
        self.bulk = bulk_data
        self.tickers = [t for t in tickers if not str(t).startswith("^")]
        self.cash = CAPITAL
        self.equity_curve = [CAPITAL]
        self.portfolio = []
        self.history = []  # finished trades

    def calc_qty(self, entry, sl, equity):
        risk_amt = equity * RISK_PER_TRADE
        per_share_risk = abs(entry - sl)
        if per_share_risk <= 0:
            return 0
        qty = int(risk_amt / per_share_risk)
        # limit position percent
        max_shares = int((equity * 0.25) / entry)  # 25% max position
        if qty * entry > equity * 0.25:
            qty = max_shares
        return max(qty, 0)

    def process_exit(self, trade, row, date):
        """
        Check whether trade should exit today based on row (series).
        Returns True if exited, else False.
        """
        exit_price = None
        # stop loss conditions (gap or intraday)
        if row["Low"] <= trade["sl"]:
            exit_price = trade["sl"]
        if row["Open"] < trade["sl"]:
            exit_price = row["Open"]

        # target conditions
        if row["High"] >= trade["target"]:
            exit_price = trade["target"]
        if row["Open"] > trade["target"]:
            exit_price = row["Open"]

        if exit_price is None:
            return False

        qty = trade["qty"]
        revenue = exit_price * qty
        cost = revenue * BROKERAGE_PCT
        entry_cost = trade["entry"] * qty * BROKERAGE_PCT
        pnl = revenue - cost - (trade["entry"] * qty + entry_cost)

        # record trade
        self.history.append({
            "symbol": trade["symbol"],
            "entry": round(trade["entry"], 2),
            "exit": round(exit_price, 2),
            "qty": qty,
            "entry_date": trade["entry_date"].strftime("%Y-%m-%d"),
            "exit_date": date.strftime("%Y-%m-%d"),
            "sl": round(trade["sl"], 2),
            "target": round(trade["target"], 2),
            "rr": round((trade["target"] - trade["entry"]) / (trade["entry"] - trade["sl"]) if trade["entry"] - trade["sl"] != 0 else 0, 2),
            "pnl": round(pnl, 2)
        })

        # cash update
        self.cash += (revenue - cost)
        return True

    def apply_tsl(self, trade, row, df, idx):
        """
        Hybrid A3 TSL
        trade: dict with keys entry, sl, target, qty, symbol
        row: today's row (Series)
        df: full dataframe of symbol
        idx: integer location of today's index in df
        """
        atr_v = row["ATR"]
        entry = trade["entry"]
        # +1R -> move to BE
        if row["High"] >= entry + (TSL_BE_R * atr_v):
            trade["sl"] = max(trade["sl"], entry)

        # +2R -> move to EMA20 (today's EMA20)
        if row["High"] >= entry + (TSL_EMA20_R * atr_v):
            if "EMA20" in row:
                trade["sl"] = max(trade["sl"], row["EMA20"])

        # +3R -> move to swing low of previous 5 candles
        if row["High"] >= entry + (TSL_SWING_R * atr_v):
            if idx >= 5:
                swing_low = df["Low"].iloc[idx-5:idx].min()
                trade["sl"] = max(trade["sl"], swing_low)

    def calc_recent_high(self, df, date, lookback=BREAKOUT_LOOKBACK):
        # highest high over lookback excluding current day (shifted)
        recent = df["High"].rolling(lookback).max().shift(1)
        try:
            return recent.loc[date]
        except Exception:
            return None

    def run(self):
        logger.info("Running backtest...")
        # Build processed map only for tickers present in bulk
        processed = {}
        weekly_map = {}
        for t in self.tickers:
            df = None
            try:
                if isinstance(self.bulk, dict):
                    df = self.bulk.get(t)
                else:
                    if isinstance(self.bulk.columns, pd.MultiIndex):
                        if t in self.bulk.columns.get_level_values(0):
                            df = self.bulk[t].copy()
            except Exception:
                df = None

            if df is None or len(df) < MIN_HISTORY_ROWS:
                continue

            # prepare indicators (if not already prepared)
            if "ATR" not in df.columns or "EMA20" not in df.columns:
                try:
                    df = prepare_df(df)
                except Exception:
                    continue

            if df.empty or len(df) < MIN_HISTORY_ROWS:
                continue

            processed[t] = df
            weekly_map[t] = make_weekly(df)

        if not processed:
            logger.info("No processed tickers available for backtest.")
            return {
                "curve": [CAPITAL],
                "profit": 0,
                "total_trades": 0,
                "win_rate": 0,
                "history": [],
                "winning_symbols": [],
                "losing_symbols": [],
                "avg_win": 0,
                "avg_loss": 0,
                "avg_rr": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "last_run": datetime.utcnow().strftime("%Y-%m-%d")
            }

        # union of dates
        all_dates = sorted(set().union(*[df.index for df in processed.values()]))
        for date in all_dates:
            # process exits first
            new_port = []
            for trade in self.portfolio:
                sym = trade["symbol"]
                df = processed.get(sym)
                if df is None or date not in df.index:
                    new_port.append(trade)
                    continue
                row = df.loc[date]
                idx = df.index.get_loc(date)
                # apply tsl before checking exits
                self.apply_tsl(trade, row, df, idx)
                exited = self.process_exit(trade, row, date)
                if not exited:
                    new_port.append(trade)
            self.portfolio = new_port

            # entries (limit portfolio size)
            if len(self.portfolio) < 5:
                for sym, df in processed.items():
                    if date not in df.index:
                        continue
                    if len(self.portfolio) >= 5:
                        break
                    row = df.loc[date]

                    # weekly filters
                    wdf = weekly_map.get(sym)
                    if wdf is None or len(wdf) < 10:
                        continue
                    last_week = wdf.iloc[-1]
                    if last_week["Close"] < last_week["EMA20"]:
                        continue
                    if last_week["EMA20_SLOPE"] <= 0:
                        continue

                    # daily trend
                    if not (row["Close"] > row["EMA20"] and row["EMA20_SLOPE"] > 0 and row["Close"] > row["SMA200"]):
                        continue

                    # momentum
                    if row["RSI"] < RSI_MIN or row["ADX"] < ADX_MIN:
                        continue

                    # volume filter (relaxed from strict)
                    if pd.isna(row.get("VOL_SMA20", None)) or row["Volume"] < VOLUME_SPIKE * row["VOL_SMA20"]:
                        continue

                    # breakout (allow 0.5% tolerance)
                    recent_high = self.calc_recent_high(df, date, lookback=BREAKOUT_LOOKBACK)
                    if recent_high is None:
                        continue
                    if not (row["Close"] >= recent_high * 0.995):
                        continue

                    # OBV filter (allow breathing room)
                    if pd.isna(row.get("OBV", None)) or pd.isna(row.get("OBV_SMA", None)):
                        continue
                    if row["OBV"] < row["OBV_SMA"] * 0.95:
                        continue

                    # sl, tgt, rr
                    sl = row["Close"] - row["ATR"]
                    target = row["Close"] + 2.5 * row["ATR"]
                    if sl >= row["Close"]:
                        continue
                    rr = (target - row["Close"]) / (row["Close"] - sl) if (row["Close"] - sl) != 0 else 0
                    if rr < RR_MIN:
                        continue

                    # position sizing
                    qty = self.calc_qty(row["Close"], sl, self.equity_curve[-1])
                    if qty <= 0 or qty * row["Close"] > self.equity_curve[-1]:
                        continue

                    # execute entry
                    entry_cost = row["Close"] * qty * BROKERAGE_PCT
                    self.cash -= (row["Close"] * qty + entry_cost)
                    self.portfolio.append({
                        "symbol": sym,
                        "entry": row["Close"],
                        "sl": sl,
                        "target": target,
                        "qty": qty,
                        "entry_date": date,
                    })

            # mark-to-market
            m2m = self.cash
            for t in self.portfolio:
                sym = t["symbol"]
                df = processed.get(sym)
                if df is None:
                    m2m += t["entry"] * t["qty"]
                else:
                    if df.index[-1] >= date and date in df.index:
                        m2m += df.loc[date]["Close"] * t["qty"]
                    else:
                        m2m += t["entry"] * t["qty"]
            self.equity_curve.append(round(m2m, 2))

        # build report
        if not self.history:
            return {
                "curve": [round(x, 2) for x in self.equity_curve],
                "profit": round(self.equity_curve[-1] - CAPITAL, 2),
                "total_trades": 0,
                "win_rate": 0,
                "history": [],
                "winning_symbols": [],
                "losing_symbols": [],
                "avg_win": 0,
                "avg_loss": 0,
                "avg_rr": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "last_run": datetime.utcnow().strftime("%Y-%m-%d")
            }

        # compute metrics
        wins = [t for t in self.history if t["pnl"] > 0]
        losses = [t for t in self.history if t["pnl"] <= 0]
        avg_win = float(round(np.mean([t["pnl"] for t in wins]), 2)) if wins else 0
        avg_loss = float(round(np.mean([t["pnl"] for t in losses]), 2)) if losses else 0
        avg_rr = float(round(np.mean([t["rr"] for t in self.history]), 2)) if self.history else 0

        curve_series = pd.Series(self.equity_curve)
        roll_max = curve_series.cummax()
        drawdown = (curve_series - roll_max) / roll_max
        max_dd = float(drawdown.min())

        ret = curve_series.pct_change().dropna()
        sharpe = float(round((ret.mean() / ret.std() * (252 ** 0.5)) if ret.std() != 0 else 0, 3))

        return {
            "curve": [round(x, 2) for x in self.equity_curve],
            "profit": round(self.equity_curve[-1] - CAPITAL, 2),
            "total_trades": len(self.history),
            "win_rate": round(100 * len(wins) / len(self.history), 2),
            "history": self.history,
            "winning_symbols": list({t["symbol"] for t in wins}),
            "losing_symbols": list({t["symbol"] for t in losses}),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_rr": avg_rr,
            "max_drawdown": round(max_dd, 4),
            "sharpe_ratio": sharpe,
            "last_run": datetime.utcnow().strftime("%Y-%m-%d")
        }

# -------------------------
# Robust download + prepare
# -------------------------
def download_and_prepare_tickers(tickers, min_rows=MIN_HISTORY_ROWS, period=DATA_PERIOD):
    """
    Downloads each ticker individually, prepares the dataframe (prepare_df),
    and returns:
      - bulk: a concatenated MultiIndex DataFrame (ticker -> Open/High/...)
      - prepared_map: dict[ticker] = prepared df (cleaned)
      - skipped: list of tickers skipped (delisted or insufficient data)
    """
    prepared_map = {}
    skipped = []
    logger.info(f"Downloading {len(tickers)} tickers individually (robust mode)...")

    for t in tickers:
        try:
            raw = yf.download(
                t,
                period=period,
                interval="1d",
                progress=False,
                threads=True,
                ignore_tz=True,
                auto_adjust=True
            )
            if raw is None or raw.empty:
                logger.warning(f"Skipping {t} — no data (may be delisted).")
                skipped.append(t)
                continue

            # Ensure columns exist
            req_cols = {"Open", "High", "Low", "Close", "Volume"}
            if not req_cols.issubset(set(raw.columns)):
                logger.warning(f"Skipping {t} — missing required columns. Found: {list(raw.columns)}")
                skipped.append(t)
                continue

            if len(raw) < min_rows:
                logger.info(f"Skipping {t} — insufficient history ({len(raw)} rows).")
                skipped.append(t)
                continue

            df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df = prepare_df(df)
            if df is None or df.empty or len(df) < min_rows:
                logger.info(f"Skipping {t} after prepare_df — insufficient rows.")
                skipped.append(t)
                continue

            prepared_map[t] = df
            logger.info(f"Prepared {t}: {len(df)} rows.")
        except Exception as e:
            logger.warning(f"Failed to fetch/prepare {t}: {repr(e)}")
            skipped.append(t)
            continue

    if not prepared_map:
        logger.error("No tickers prepared successfully. Aborting.")
        return None, {}, skipped

    # Build MultiIndex bulk DataFrame: keys = ticker
    try:
        bulk = pd.concat(prepared_map, axis=1)
    except Exception as e:
        logger.error(f"Failed to concat prepared frames: {e}")
        return None, prepared_map, skipped

    logger.info(f"Prepared {len(prepared_map)} tickers, skipped {len(skipped)} tickers.")
    return bulk, prepared_map, skipped

# -------------------------
# extract_df helper (works with dict or MultiIndex)
# -------------------------
def extract_df(bulk, ticker):
    try:
        if isinstance(bulk, dict):
            return bulk.get(ticker, None)
        if isinstance(bulk.columns, pd.MultiIndex):
            if ticker in bulk.columns.get_level_values(0):
                return bulk[ticker].copy().dropna()
        return None
    except Exception:
        return None

# -------------------------
# Main runner
# -------------------------
def run_backtest_main():
    # Load NIFTY500 list if available
    if os.path.exists("ind_nifty500list.csv"):
        try:
            df_symbols = pd.read_csv("ind_nifty500list.csv")
            tickers = [f"{s}.NS" for s in df_symbols["Symbol"].dropna().unique()]
        except Exception as e:
            logger.warning(f"Failed to read ind_nifty500list.csv: {e}. Falling back to default tickers.")
            tickers = DEFAULT_TICKERS.copy()
    else:
        tickers = DEFAULT_TICKERS.copy()

    # quick sanitize: filter out obvious placeholders
    cleaned = []
    bad_prefixes = ["DUMMY", "TMP", "TEST", "ZZ"]
    for t in tickers:
        base = t.replace(".NS", "").upper()
        if any(base.startswith(p) for p in bad_prefixes):
            logger.info(f"Filtering out placeholder ticker: {t}")
            continue
        cleaned.append(t)
    tickers = cleaned

    # Download + prepare
    bulk, prepared_map, skipped = download_and_prepare_tickers(tickers, min_rows=MIN_HISTORY_ROWS, period=DATA_PERIOD)
    if bulk is None:
        logger.error("No data prepared. Exiting.")
        return

    # Instantiate engine (BacktestEngine class must be defined above)
    try:
        engine = BacktestEngine(bulk, list(prepared_map.keys()))
    except NameError as e:
        logger.error("BacktestEngine not defined or not in scope.")
        raise

    report = engine.run()

    # Compose output JSON (enhanced)
    out = {
        "updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "portfolio": {
            "curve": report.get("curve", [CAPITAL]),
            "profit": report.get("profit", 0),
            "total_trades": report.get("total_trades", 0),
            "win_rate": report.get("win_rate", 0),
            "history": report.get("history", [])[:1000]
        },
        "tickers": {k.replace(".NS", ""): 0 for k in prepared_map.keys()},
        "winning_symbols": report.get("winning_symbols", []),
        "losing_symbols": report.get("losing_symbols", []),
        "avg_win": report.get("avg_win", 0),
        "avg_loss": report.get("avg_loss", 0),
        "avg_rr": report.get("avg_rr", 0),
        "max_drawdown": report.get("max_drawdown", 0),
        "sharpe_ratio": report.get("sharpe_ratio", 0),
        "skipped_tickers": skipped,
        "last_run": report.get("last_run", datetime.utcnow().strftime("%Y-%m-%d"))
    }

    with open(CACHE_FILE, "w") as f:
        json.dump(out, f, indent=2)

    logger.info("=========================================================")
    logger.info("BACKTEST COMPLETE — Summary:")
    logger.info(f"Prepared tickers: {len(prepared_map)}")
    logger.info(f"Skipped tickers: {len(skipped)}")
    logger.info(f"Trades: {out['portfolio']['total_trades']}")
    logger.info(f"Win Rate: {out['portfolio']['win_rate']}%")
    logger.info(f"Profit: {out['portfolio']['profit']}")
    logger.info("Saved backtest_stats.json")
    logger.info("=========================================================")

    return out

if __name__ == "__main__":
    run_backtest_main()

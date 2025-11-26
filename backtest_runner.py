import yfinance as yf
import pandas as pd
import numpy as np
import logging
import json
import time
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
DATA_PERIOD = "2y"        # your selection
CAPITAL = 100000
RISK_PER_TRADE = 0.02
MAX_POSITIONS = 5

BROKERAGE = 0.001

ADX_LOW = 20
ADX_HIGH = 35
RSI_LOW = 55
RSI_HIGH = 65

VOL_SPIKE = 1.5

BREAKOUT_LOOKBACK = 10

# TSL LOGIC (A3 Hybrid)
TSL_BE_R = 1       # breakeven at 1R
TSL_EMA20_R = 2    # EMA20 is SL at 2R
TSL_SWING_R = 3    # Swing low SL at 3R

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BacktestA3")

# ============================================================
# INDICATORS
# ============================================================
def true_range(df):
    prev = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev).abs()
    tr3 = (df["Low"] - prev).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df, period=14):
    return true_range(df).ewm(span=period, adjust=False).mean()

def wilder_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - 100/(1 + rs)

def adx(df, period=14):
    up = df["High"].diff()
    dn = df["Low"].diff().abs()
    plus = np.where((up > dn) & (up > 0), up, 0)
    minus = np.where((dn > up) & (df["Low"].diff() < 0), dn, 0)

    trn = true_range(df).ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus).ewm(alpha=1/period, adjust=False).mean() / trn)
    minus_di = 100 * (pd.Series(minus).ewm(alpha=1/period, adjust=False).mean() / trn)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.ewm(alpha=1/period, adjust=False).mean()

def obv(df):
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()

def prepare_df(df):
    df = df.copy()
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["ATR"] = atr(df)
    df["RSI"] = wilder_rsi(df["Close"])
    df["ADX"] = adx(df)
    df["OBV"] = obv(df)
    df["OBV_SMA"] = df["OBV"].rolling(20).mean()
    df["VOL_SMA20"] = df["Volume"].rolling(20).mean()
    df["EMA20_SLOPE"] = df["EMA20"].diff()
    return df

# ============================================================
# WEEKLY TREND (MTF)
# ============================================================
def weekly_trend(df):
    wk = df.resample("W").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    }).dropna()

    wk["EMA20"] = wk["Close"].ewm(span=20).mean()
    wk["SMA50"] = wk["Close"].rolling(50).mean()

    if len(wk) < 60:
        return False

    last = wk.iloc[-1]
    if last["Close"] > last["EMA20"] and last["EMA20"] > last["SMA50"]:
        return True
    return False

# ============================================================
# SWING LOW
# ============================================================
def swing_low(df, idx, lookback=5):
    if idx < lookback:
        return df["Low"].iloc[idx]
    window = df["Low"].iloc[idx-lookback:idx]
    return window.min()

# ============================================================
# DOWNLOAD DATA
# ============================================================
def download_data(tickers):
    out = {}
    for t in tickers:
        try:
            df = yf.download(t, period=DATA_PERIOD, progress=False)
            if len(df) > 250:
                out[t] = prepare_df(df)
        except:
            pass
    return out

# ============================================================
# BACKTEST ENGINE
# ============================================================
class BacktestA3:
    def __init__(self, data):
        self.data = data
        self.cash = CAPITAL
        self.equity = CAPITAL
        self.curve = [CAPITAL]
        self.positions = []
        self.history = []

    def position_size(self, entry, sl):
        risk = entry - sl
        if risk <= 0:
            return 0
        max_risk_amt = self.equity * RISK_PER_TRADE
        qty = int(max_risk_amt / risk)
        return qty

    # ---------------------------------------------
    # ENTRY CHECK
    # ---------------------------------------------
    def check_entry(self, df, i):
        row = df.iloc[i]

        # trend
        if not (row["Close"] > row["EMA20"] > row["SMA50"] > row["SMA200"]):
            return False
        if row["EMA20_SLOPE"] <= 0:
            return False

        # weekly trend
        if not weekly_trend(df[:df.index[i]]):
            return False

        # momentum
        if not (RSI_LOW <= row["RSI"] <= RSI_HIGH):
            return False
        if not (ADX_LOW <= row["ADX"] <= ADX_HIGH):
            return False

        # volume
        if row["Volume"] < VOL_SPIKE * row["VOL_SMA20"]:
            return False

        # OBV
        if row["OBV"] < row["OBV_SMA"]:
            return False

        # breakout
        prev_high = df["High"].iloc[i-BREAKOUT_LOOKBACK:i].max()
        if row["Close"] <= prev_high:
            return False

        return True

    # ---------------------------------------------
    # EXIT / TSL
    # ---------------------------------------------
    def process_position(self, df, pos, i):
        row = df.iloc[i]
        entry = pos["entry"]
        atr = pos["atr"]

        # R multiples
        r1 = entry + atr * TSL_BE_R
        r2 = entry + atr * TSL_EMA20_R
        r3 = entry + atr * TSL_SWING_R

        # move to breakeven
        if pos["sl"] < entry and row["High"] >= r1:
            pos["sl"] = entry

        # trail at EMA20
        if row["High"] >= r2:
            pos["sl"] = max(pos["sl"], row["EMA20"])

        # swing low trail
        if row["High"] >= r3:
            sl_sw = swing_low(df, i, 5)
            pos["sl"] = max(pos["sl"], sl_sw)

        # TARGET HIT
        if row["High"] >= pos["target"]:
            exit_price = pos["target"]
            return exit_price

        # SL HIT
        if row["Low"] <= pos["sl"]:
            exit_price = pos["sl"]
            return exit_price

        return None

    # ---------------------------------------------
    # MAIN BACKTEST LOOP
    # ---------------------------------------------
    def run(self):
        for sym, df in self.data.items():
            for i in range(200, len(df)):
                row = df.iloc[i]

                # process open trades
                new_positions = []
                for pos in self.positions:
                    if pos["symbol"] != sym:
                        new_positions.append(pos)
                        continue

                    exit_price = self.process_position(df, pos, i)
                    if exit_price:
                        pnl = (exit_price - pos["entry"]) * pos["qty"]
                        self.cash += exit_price * pos["qty"]
                        self.history.append({
                            "symbol": sym,
                            "entry": pos["entry"],
                            "exit": exit_price,
                            "pnl": pnl
                        })
                    else:
                        new_positions.append(pos)

                self.positions = new_positions

                # ENTRY
                if len(self.positions) < MAX_POSITIONS and self.check_entry(df, i):
                    sl = row["Close"] - row["ATR"]
                    target = row["Close"] + 3 * row["ATR"]
                    qty = self.position_size(row["Close"], sl)

                    if qty > 0:
                        cost = row["Close"] * qty
                        if self.cash >= cost:
                            self.cash -= cost
                            self.positions.append({
                                "symbol": sym,
                                "entry": row["Close"],
                                "sl": sl,
                                "target": target,
                                "qty": qty,
                                "atr": row["ATR"]
                            })

                # update equity curve
                mtm = 0
                for pos in self.positions:
                    mtm += df.iloc[i]["Close"] * pos["qty"]

                self.equity = self.cash + mtm
                self.curve.append(self.equity)

        return self.results()

    def results(self):
        wins = [t for t in self.history if t["pnl"] > 0]
        winrate = round(len(wins) / len(self.history) * 100, 2) if self.history else 0

        return {
            "curve": self.curve,
            "profit": round(self.curve[-1] - CAPITAL, 2),
            "total_trades": len(self.history),
            "win_rate": winrate,
            "history": self.history
        }

# ============================================================
# MAIN RUN
# ============================================================
if __name__ == "__main__":
    tickers = []
    try:
        df = pd.read_csv("ind_nifty500list.csv")
        tickers = [t+".NS" for t in df["Symbol"].unique()]
    except:
        pass

    if not tickers:
        tickers = ["RELIANCE.NS","TCS.NS","HDFCBANK.NS"]

    logger.info("Downloading data...")
    data = download_data(tickers)

    logger.info("Running backtest...")
    engine = BacktestA3(data)
    stats = engine.run()

    with open("backtest_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Backtest complete.")
    logger.info(stats)

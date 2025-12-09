# app_mt5.py
import os, json, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import MetaTrader5 as mt5
import torch
import torch.nn as nn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# =========================
# PATHS (แก้ให้ตรงเครื่องคุณ)
# =========================
BASE_DIR = Path(
    "/Users/thanaporn/Desktop/EURO_H1_AI/prepared_datasets/boosting_dl_residual"
)
MODEL_DIR = BASE_DIR / "trained_models"
META_PATH = BASE_DIR / "eurusd_struct_meta.json"

# =========================
# Load meta
# =========================
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

feature_cols = meta["feature_cols"]
seq_len = meta["seq_len"]
horizon = meta["horizon"]
target_names = meta["targets_boosting"]  # ["gap_next","range_next","body_next"]

# =========================
# Load Boosting models
# =========================
boost_models = [joblib.load(MODEL_DIR / f"lgb_{t}.pkl") for t in target_names]


def predict_boost_np(x_np_2d):
    preds = []
    for m in boost_models:
        preds.append(m.predict(x_np_2d))
    return np.stack(preds, axis=1)  # [N,3]


# =========================
# Load DL model + scaler
# =========================
class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (k - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        y = self.net(x)
        y = y[..., : x.size(-1)]
        return y + self.down(x)


class TCN(nn.Module):
    def __init__(self, n_features, channels=(64, 64, 64)):
        super().__init__()
        layers = []
        c_in = n_features
        for i, c_out in enumerate(channels):
            layers.append(TCNBlock(c_in, c_out, dilation=2**i))
            c_in = c_out
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(channels[-1], 5)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.transpose(1, 2)
        z = self.tcn(x)
        z_last = z[..., -1]
        raw = self.head(z_last)
        res3 = raw[:, 0:3]
        uw = self.softplus(raw[:, 3:4])
        lw = self.softplus(raw[:, 4:5])
        return torch.cat([res3, uw, lw], dim=1)


device = "mps" if torch.backends.mps.is_available() else "cpu"
model = TCN(n_features=len(feature_cols)).to(device)
model.load_state_dict(torch.load(MODEL_DIR / "tcn_residual.pth", map_location=device))
model.eval()

scaler = joblib.load(MODEL_DIR / "dl_scaler.pkl")
close_pos = feature_cols.index("Close")


# =========================
# Feature fns (เหมือนตอนเทรน)
# =========================
def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def atr(high, low, close, window: int = 14):
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=window, min_periods=1).mean()


def add_basic_features(df: pd.DataFrame):
    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    df["ret_1"] = close.pct_change(1) * 100
    df["ret_4"] = close.pct_change(4) * 100
    df["ret_12"] = close.pct_change(12) * 100

    df["ema_20"] = ema(close, 20)
    df["ema_50"] = ema(close, 50)
    df["ema_100"] = ema(close, 100)

    df["rsi_14"] = rsi(close, 14)
    df["atr_14"] = atr(high, low, close, 14)
    df["vol_20"] = df["ret_1"].rolling(window=20, min_periods=20).std()

    candle_body = close - open_
    candle_range = high - low
    is_bull = candle_body >= 0
    upper_wick = np.where(is_bull, high - close, high - open_)
    lower_wick = np.where(is_bull, open_ - low, close - low)

    df["candle_body"] = candle_body
    df["candle_range"] = candle_range
    df["upper_wick"] = upper_wick
    df["lower_wick"] = lower_wick

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    hour = df["hour"]
    df["session_asia"] = ((hour >= 0) & (hour < 8)).astype(int)
    df["session_london"] = ((hour >= 8) & (hour < 16)).astype(int)
    df["session_ny"] = ((hour >= 16) & (hour < 24)).astype(int)

    df["Spread"] = df["Spread"].astype(float)
    return df


# =========================
# MT5 utilities
# =========================
def mt5_connect():
    """
    ถ้า Terminal เปิดอยู่แล้ว initialize() แบบไม่ใส่ param จะเจอเอง
    ถ้าต้อง login เอง ให้ set ENV:
      MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
    """
    if mt5.initialize():
        return True

    # fallback: try login from env
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    if login and password and server:
        ok = mt5.initialize(login=int(login), password=password, server=server)
        if ok:
            return True

    err = mt5.last_error()
    raise RuntimeError(f"MT5 initialize/login failed: {err}")


def fetch_rates(symbol="EURUSD", timeframe=mt5.TIMEFRAME_D1, bars=300):
    """
    ดึงแท่งล่าสุดจาก MT5
    copy_rates_from_pos(symbol, timeframe, start_pos, count)
    start_pos=0 คือแท่งล่าสุด (รวมแท่งปัจจุบัน) :contentReference[oaicite:4]{index=4}
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No rates for {symbol}")
    df = pd.DataFrame(rates)
    df["Time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("Time").sort_index()
    # map MT5 names -> ours
    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "tick_volume": "Volume",
            "spread": "Spread",
        }
    )
    # real_volume ถ้าไม่มีไม่เป็นไร
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    if "Spread" not in df.columns:
        df["Spread"] = 0.0
    return df[["Open", "High", "Low", "Close", "Volume", "Spread"]]


def predict_next_from_rates(df_ohlcv: pd.DataFrame):
    df_feat = add_basic_features(df_ohlcv.copy()).dropna()
    if len(df_feat) < seq_len:
        raise ValueError(
            f"Need >= {seq_len} rows after feature warmup, got {len(df_feat)}"
        )

    df_last = df_feat.iloc[-seq_len:].copy()

    x_seq = df_last[feature_cols].values.astype(np.float32)
    x_tab = df_last.iloc[-1][feature_cols].values.astype(np.float32).reshape(1, -1)

    boost_pred = predict_boost_np(x_tab)[0]  # [3]

    x_seq_s = scaler.transform(x_seq).reshape(1, seq_len, len(feature_cols))
    x_seq_s = torch.tensor(x_seq_s, dtype=torch.float32).to(device)

    with torch.no_grad():
        dl_pred = model(x_seq_s).cpu().numpy()[0]  # [5]

    res3 = dl_pred[:3]
    uw, lw = dl_pred[3], dl_pred[4]
    gap, rnge, body = boost_pred + res3

    close_t = float(x_tab[0, close_pos])

    open_next = close_t + gap
    close_next = open_next + body
    high_next = max(open_next, close_next) + uw
    low_next = min(open_next, close_next) - lw

    return {
        "Open_next_pred": float(open_next),
        "High_next_pred": float(high_next),
        "Low_next_pred": float(low_next),
        "Close_next_pred": float(close_next),
        "gap_pred": float(gap),
        "range_pred": float(rnge),
        "body_pred": float(body),
        "upper_wick_pred": float(uw),
        "lower_wick_pred": float(lw),
        "last_close_t": close_t,
        "bars_used": len(df_last),
        "last_time": str(df_last.index[-1]),
    }


# =========================
# FastAPI schemas
# =========================
class MT5PredictRequest(BaseModel):
    symbol: str = "EURUSD"
    timeframe: str = "D1"  # "D1","H1","H4" etc
    bars: int = 300  # ดึงมากพอให้ indicator warmup + seq_len


# =========================
# FastAPI app
# =========================
app = FastAPI(title="MT5 Auto-Fetch Forecaster")

TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
}


@app.on_event("startup")
def on_start():
    mt5_connect()


@app.on_event("shutdown")
def on_stop():
    mt5.shutdown()


@app.get("/health")
def health():
    return {"status": "ok", "device": device, "seq_len": seq_len}


@app.post("/predict_mt5")
def predict_mt5(req: MT5PredictRequest):
    try:
        tf = TF_MAP.get(req.timeframe.upper())
        if tf is None:
            raise ValueError(
                f"Unsupported timeframe {req.timeframe}. Use one of {list(TF_MAP.keys())}"
            )

        df_rates = fetch_rates(req.symbol, tf, req.bars)
        out = predict_next_from_rates(df_rates)
        return out

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

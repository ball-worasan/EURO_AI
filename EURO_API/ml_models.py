# ml_models.py
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from typing import Dict, Any

from config import MODEL_DIR, meta, feature_cols, seq_len, horizon, target_names, device

# =========================
# Load Boosting models (best-effort)
# =========================

boost_models = []
boost_models_loaded = False

if MODEL_DIR and target_names:
    try:
        for t in target_names:
            boost_models.append(joblib.load(MODEL_DIR / f"lgb_{t}.pkl"))
        boost_models_loaded = True
    except Exception as e:
        boost_models = []
        boost_models_loaded = False
        print(f"[WARN] failed to load boosting models from {MODEL_DIR}: {e}")
else:
    print("[WARN] MODEL_DIR or target_names not ready; skip boosting load.")


def predict_boost_np(x_np_2d: np.ndarray) -> np.ndarray:
    """
    x_np_2d: shape [N, n_features]
    return: shape [N, 3] -> [gap_next, range_next, body_next]
    """
    if not boost_models_loaded:
        raise RuntimeError(
            "Boosting models not loaded. " "Check EURO_AI_BASE_DIR / trained_models."
        )
    preds = []
    for m in boost_models:
        preds.append(m.predict(x_np_2d))
    return np.stack(preds, axis=1)  # [N,3]


# =========================
# DL model (TCN) + scaler
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
        # x: [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)
        z = self.tcn(x)
        z_last = z[..., -1]
        raw = self.head(z_last)
        res3 = raw[:, 0:3]
        uw = self.softplus(raw[:, 3:4])
        lw = self.softplus(raw[:, 4:5])
        return torch.cat([res3, uw, lw], dim=1)


model = None
dl_loaded = False

if MODEL_DIR and feature_cols:
    try:
        model = TCN(n_features=len(feature_cols)).to(device)
        model.load_state_dict(
            torch.load(MODEL_DIR / "tcn_residual.pth", map_location=device)
        )
        model.eval()
        scaler = joblib.load(MODEL_DIR / "dl_scaler.pkl")
        dl_loaded = True
    except Exception as e:
        model = None
        scaler = None
        dl_loaded = False
        print(f"[WARN] failed to load DL model/scaler from {MODEL_DIR}: {e}")
else:
    scaler = None
    print("[WARN] MODEL_DIR or feature_cols not ready; skip DL load.")

close_pos = feature_cols.index("Close") if "Close" in feature_cols else None

# =========================
# Feature functions
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


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
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
# Core prediction
# =========================


def predict_next_from_rates(df_ohlcv: pd.DataFrame) -> Dict[str, Any]:
    if not (dl_loaded and boost_models_loaded):
        raise RuntimeError(
            "Models are not fully loaded, check /health and EURO_AI_BASE_DIR."
        )

    if seq_len <= 0 or not feature_cols:
        raise RuntimeError("Meta not loaded correctly (seq_len/feature_cols).")

    df_feat = add_basic_features(df_ohlcv.copy()).dropna()
    if len(df_feat) < seq_len:
        raise ValueError(
            f"Need >= {seq_len} rows after feature warmup, got {len(df_feat)}"
        )

    df_last = df_feat.iloc[-seq_len:].copy()

    x_tab = df_last.iloc[-1][feature_cols].values.astype(np.float32).reshape(1, -1)
    boost_pred = predict_boost_np(x_tab)[0]

    x_seq = df_last[feature_cols].values.astype(np.float32)
    x_seq_s = scaler.transform(x_seq).reshape(1, seq_len, len(feature_cols))
    x_seq_s = torch.tensor(x_seq_s, dtype=torch.float32).to(device)

    with torch.no_grad():
        dl_pred = model(x_seq_s).cpu().numpy()[0]

    res3 = dl_pred[:3]
    uw, lw = dl_pred[3], dl_pred[4]
    gap, rnge, body = boost_pred + res3

    close_t = float(df_last.iloc[-1]["Close"])

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

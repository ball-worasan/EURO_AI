import os
import json
import html
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from collections import deque
import time

import torch
import torch.nn as nn

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from datetime import datetime

# =========================
# SIMPLE IN-MEMORY REQUEST LOG
# =========================

# เก็บ log ล่าสุดไม่เกิน 1000 รายการ
REQUEST_LOG = deque(maxlen=1000)


def log_request(entry: dict):
    """เก็บ log ไว้ในหน่วยความจำ"""
    REQUEST_LOG.append(entry)


def format_ts(ts: datetime | None) -> str:
    if not ts:
        return "-"
    return ts.strftime("%Y-%m-%d %H:%M:%S")


# =========================
# CONFIG & PATH RESOLUTION
# =========================

def resolve_base_dir() -> Path:
    """
    เลือก BASE_DIR แบบ dynamic:
    1) ถ้ามี EURO_AI_BASE_DIR -> ใช้อันนั้น
    2) ถ้าไม่มี ลองหาโฟลเดอร์ prepared_datasets/... ที่อยู่ใกล้ไฟล์นี้
    3) ถ้าไม่เจอเลย -> โยน error ให้รู้ตัวชัด ๆ
    """
    env_base = os.getenv("EURO_AI_BASE_DIR")
    if env_base:
        base = Path(env_base).expanduser().resolve()
        if base.exists():
            return base
        else:
            raise RuntimeError(f"EURO_AI_BASE_DIR not found: {base}")

    here = Path(__file__).resolve().parent
    candidates = [
        here / "prepared_datasets/boosting_dl_residual",
        here.parent / "prepared_datasets/boosting_dl_residual",
    ]

    for c in candidates:
        if (c / "eurusd_struct_meta.json").exists():
            return c.resolve()

    raise RuntimeError(
        "Cannot locate model directory. "
        "Set EURO_AI_BASE_DIR env to folder containing eurusd_struct_meta.json + trained_models/"
    )


try:
    BASE_DIR = resolve_base_dir()
except Exception as e:
    print(f"[CONFIG ERROR] {e}")
    BASE_DIR = None

MODEL_DIR = BASE_DIR / "trained_models" if BASE_DIR else None
META_PATH = BASE_DIR / "eurusd_struct_meta.json" if BASE_DIR else None

# กำหนด max bars เพื่อเช็ค “เกินไปไหม”
MAX_BARS_ALLOWED = int(os.getenv("MAX_BARS_ALLOWED", "500"))

# =========================
# Load meta
# =========================
if META_PATH and META_PATH.exists():
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    seq_len = meta["seq_len"]
    horizon = meta["horizon"]
    # ["gap_next","range_next","body_next"]
    target_names = meta["targets_boosting"]
else:
    meta = {}
    feature_cols = []
    seq_len = 0
    horizon = 0
    target_names = []
    print("[WARN] META file not found. Set EURO_AI_BASE_DIR correctly.")

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
            "Boosting models not loaded. "
            "Check EURO_AI_BASE_DIR / trained_models."
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
        self.down = nn.Conv1d(
            c_in, c_out, 1) if c_in != c_out else nn.Identity()

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


device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

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


def predict_next_from_rates(df_ohlcv: pd.DataFrame):
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

    x_tab = df_last.iloc[-1][feature_cols].values.astype(
        np.float32).reshape(1, -1)
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

# =========================
# FastAPI schemas
# =========================


class OHLCVBar(BaseModel):
    time: datetime = Field(..., description="Time of the bar (ISO8601)")
    open: float
    high: float
    low: float
    close: float
    spread: float = 0.0
    volume: float = 0.0   # ให้ตรง meta


class PredictRequest(BaseModel):
    bars: List[OHLCVBar]

# =========================
# FastAPI app
# =========================


app = FastAPI(title="AI Forex Forecaster (with detailed dashboard)")


@app.get("/health")
def health():
    total = len(REQUEST_LOG)
    ok_count = sum(1 for r in REQUEST_LOG if r.get("status") == "ok")
    err_count = sum(1 for r in REQUEST_LOG if r.get("status") == "error")
    last_ts = REQUEST_LOG[-1]["ts"] if REQUEST_LOG else None

    return {
        "status": "ok" if (BASE_DIR and meta) else "config_error",
        "device": device,
        "base_dir": str(BASE_DIR) if BASE_DIR else None,
        "seq_len": seq_len,
        "horizon": horizon,
        "boost_loaded": boost_models_loaded,
        "dl_loaded": dl_loaded,
        "total_requests": total,
        "ok_requests": ok_count,
        "error_requests": err_count,
        "last_request_at": format_ts(last_ts),
        "max_bars_allowed": MAX_BARS_ALLOWED,
    }


def bars_to_dataframe(bars: List[OHLCVBar]) -> pd.DataFrame:
    if not bars:
        raise ValueError("bars list is empty")

    data = {
        "Time": [b.time for b in bars],
        "Open": [b.open for b in bars],
        "High": [b.high for b in bars],
        "Low": [b.low for b in bars],
        "Close": [b.close for b in bars],
        "Spread": [b.spread for b in bars],
        "Volume": [b.volume for b in bars],
    }
    df = pd.DataFrame(data)
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time").set_index("Time")
    return df


@app.post("/predict")
async def predict(req: PredictRequest, request: Request):
    ts_start = time.perf_counter()
    ts_now = datetime.now()
    bars_count = len(req.bars)

    # ข้อมูล client
    client_ip = (request.client.host if request.client else None) or "-"
    xff = request.headers.get("x-forwarded-for")

    # เช็ค completeness เรื่องจำนวนแท่ง
    bars_expected_min = seq_len
    bars_ok_min = bars_count >= bars_expected_min
    bars_ok_exact_seq = (bars_count == bars_expected_min)
    bars_ok_max = bars_count <= MAX_BARS_ALLOWED

    # เตรียม meta request แบบละเอียด (แต่ไม่ใส่ทุกแท่งเพื่อไม่บวมเกิน)
    req_meta = {
        "bars_count": bars_count,
        "bars_expected_min": bars_expected_min,
        "bars_ok_min": bars_ok_min,
        "bars_ok_exact_seq_len": bars_ok_exact_seq,
        "bars_ok_max": bars_ok_max,
    }

    # ตัวอย่าง bar แรก/สุดท้าย
    bars_sample = []
    if bars_count > 0:
        first_bar = req.bars[0]
        last_bar = req.bars[-1]
        bars_sample.append(
            {
                "pos": "first",
                "time": first_bar.time.isoformat(),
                "open": first_bar.open,
                "high": first_bar.high,
                "low": first_bar.low,
                "close": first_bar.close,
                "spread": first_bar.spread,
                "volume": first_bar.volume,
            }
        )
        if bars_count > 1:
            bars_sample.append(
                {
                    "pos": "last",
                    "time": last_bar.time.isoformat(),
                    "open": last_bar.open,
                    "high": last_bar.high,
                    "low": last_bar.low,
                    "close": last_bar.close,
                    "spread": last_bar.spread,
                    "volume": last_bar.volume,
                }
            )

    try:
        df = bars_to_dataframe(req.bars)

        # enrich request meta ด้วยช่วงเวลา / close ล่าสุด
        req_meta["time_start"] = str(df.index[0])
        req_meta["time_end"] = str(df.index[-1])
        req_meta["last_close"] = float(df["Close"].iloc[-1])
        req_meta["last_volume"] = float(df["Volume"].iloc[-1])

        out = predict_next_from_rates(df)
        latency_ms = (time.perf_counter() - ts_start) * 1000.0

        res_meta = {
            "Open_next_pred": out.get("Open_next_pred"),
            "High_next_pred": out.get("High_next_pred"),
            "Low_next_pred": out.get("Low_next_pred"),
            "Close_next_pred": out.get("Close_next_pred"),
            "gap_pred": out.get("gap_pred"),
            "range_pred": out.get("range_pred"),
            "body_pred": out.get("body_pred"),
        }

        log_request(
            {
                "ts": ts_now,
                "ip": client_ip,
                "xff": xff,
                "path": str(request.url.path),
                "status": "ok",
                "bars": bars_count,
                "bars_expected_min": bars_expected_min,
                "bars_ok_min": bars_ok_min,
                "bars_ok_exact_seq_len": bars_ok_exact_seq,
                "bars_ok_max": bars_ok_max,
                "last_time": out.get("last_time"),
                "latency_ms": latency_ms,
                "error": None,
                "req_meta": req_meta,
                "req_bars_sample": bars_sample,
                "res_meta": res_meta,
            }
        )
        return out

    except Exception as e:
        latency_ms = (time.perf_counter() - ts_start) * 1000.0
        err_text = str(e)
        log_request(
            {
                "ts": ts_now,
                "ip": client_ip,
                "xff": xff,
                "path": str(request.url.path),
                "status": "error",
                "bars": bars_count,
                "bars_expected_min": bars_expected_min,
                "bars_ok_min": bars_ok_min,
                "bars_ok_exact_seq_len": bars_ok_exact_seq,
                "bars_ok_max": bars_ok_max,
                "last_time": None,
                "latency_ms": latency_ms,
                "error": err_text,
                "req_meta": req_meta,
                "req_bars_sample": bars_sample,
                "res_meta": None,
            }
        )
        raise HTTPException(status_code=400, detail=err_text)

# =========================
# DASHBOARD (API JSON)
# =========================


@app.get("/history")
def history():
    """คืน log เป็น JSON ละเอียด"""
    items = []
    for r in REQUEST_LOG:
        items.append(
            {
                "ts": format_ts(r["ts"]),
                "ip": r.get("ip"),
                "xff": r.get("xff"),
                "path": r.get("path"),
                "status": r.get("status"),
                "bars": r.get("bars"),
                "bars_expected_min": r.get("bars_expected_min"),
                "bars_ok_min": r.get("bars_ok_min"),
                "bars_ok_exact_seq_len": r.get("bars_ok_exact_seq_len"),
                "bars_ok_max": r.get("bars_ok_max"),
                "last_time": r.get("last_time"),
                "latency_ms": round(r.get("latency_ms", 0.0), 2),
                "error": r.get("error"),
                "req_meta": r.get("req_meta"),
                "req_bars_sample": r.get("req_bars_sample"),
                "res_meta": r.get("res_meta"),
            }
        )
    return {"items": items}

# =========================
# DASHBOARD (GUI)
# =========================


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    total = len(REQUEST_LOG)
    ok_count = sum(1 for r in REQUEST_LOG if r.get("status") == "ok")
    err_count = sum(1 for r in REQUEST_LOG if r.get("status") == "error")
    last_ts = REQUEST_LOG[-1]["ts"] if REQUEST_LOG else None

    # นับจำนวนครั้งต่อ IP
    ip_counts: dict[str, int] = {}
    for r in REQUEST_LOG:
        ip = r.get("ip") or "-"
        ip_counts[ip] = ip_counts.get(ip, 0) + 1

    rows = list(REQUEST_LOG)[::-1]  # ล่าสุดอยู่บน
    html_rows = []
    for idx, r in enumerate(rows):
        status = r.get("status")
        color = "#16a34a" if status == "ok" else "#dc2626"
        ip = r.get("ip") or "-"
        calls_from_ip = ip_counts.get(ip, 0)
        bars = r.get("bars", 0)
        bars_ok_min = r.get("bars_ok_min")
        bars_ok_exact = r.get("bars_ok_exact_seq_len")
        bars_ok_max = r.get("bars_ok_max")
        last_time = r.get("last_time") or "-"
        latency = r.get("latency_ms") or 0.0
        error = r.get("error") or "-"

        # สถานะเรื่อง bars
        if not bars_ok_min:
            bars_status = "SHORT"
            bars_status_color = "#facc15"  # เหลือง
        elif not bars_ok_max:
            bars_status = "TOO_MANY"
            bars_status_color = "#f97316"  # ส้ม
        elif bars_ok_exact:
            bars_status = "EXACT"
            bars_status_color = "#22c55e"  # เขียว
        else:
            bars_status = "OK"
            bars_status_color = "#22c55e"

        # detail JSON
        detail_data = {
            "ts": format_ts(r["ts"]),
            "ip": ip,
            "xff": r.get("xff"),
            "path": r.get("path"),
            "status": status,
            "bars": bars,
            "bars_expected_min": r.get("bars_expected_min"),
            "bars_ok_min": bars_ok_min,
            "bars_ok_exact_seq_len": bars_ok_exact,
            "bars_ok_max": bars_ok_max,
            "last_time": last_time,
            "latency_ms": round(latency, 2),
            "error": r.get("error"),
            "req_meta": r.get("req_meta"),
            "req_bars_sample": r.get("req_bars_sample"),
            "res_meta": r.get("res_meta"),
        }
        detail_json = html.escape(json.dumps(
            detail_data, ensure_ascii=False, indent=2))

        detail_id = f"detail-{idx}"

        html_rows.append(
            f"""
            <tr>
              <td>{format_ts(r['ts'])}</td>
              <td>{ip}</td>
              <td style="text-align:right;">{calls_from_ip}</td>
              <td><span style="color:{color};font-weight:600;">{status.upper()}</span></td>
              <td style="text-align:right;">{bars}</td>
              <td><span style="color:{bars_status_color};font-weight:600;">{bars_status}</span></td>
              <td>{last_time}</td>
              <td style="text-align:right;">{latency:.2f}</td>
              <td>{error}</td>
              <td style="text-align:center;">
                <button onclick="toggleDetail('{detail_id}')" style="background:#1f2937;border:0;border-radius:999px;padding:4px 10px;color:#e5e7eb;font-size:11px;cursor:pointer;">
                  Details
                </button>
              </td>
            </tr>
            <tr id="{detail_id}" style="display:none;">
              <td colspan="10" style="background:#020617;">
                <pre style="white-space:pre-wrap;font-size:12px;margin:6px 0 4px 0;">{detail_json}</pre>
              </td>
            </tr>
            """
        )

    html_page = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <title>AI Forex Forecaster Dashboard</title>
      <style>
        body {{
          font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #0f172a;
          color: #e5e7eb;
          margin: 0;
          padding: 24px;
        }}
        .container {{
          max-width: 1300px;
          margin: 0 auto;
        }}
        h1 {{
          font-size: 24px;
          margin-bottom: 4px;
        }}
        .subtitle {{
          color: #9ca3af;
          font-size: 14px;
          margin-bottom: 16px;
        }}
        .cards {{
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
          gap: 16px;
          margin-bottom: 24px;
        }}
        .card {{
          background: #111827;
          border-radius: 12px;
          padding: 16px 18px;
          border: 1px solid #1f2937;
        }}
        .card-title {{
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: .08em;
          color: #9ca3af;
          margin-bottom: 6px;
        }}
        .card-value {{
          font-size: 20px;
          font-weight: 600;
        }}
        .pill {{
          display: inline-flex;
          align-items: center;
          gap: 6px;
          background: #0b1120;
          border-radius: 999px;
          padding: 4px 10px;
          border: 1px solid #1f2937;
          font-size: 11px;
          color: #9ca3af;
        }}
        table {{
          width: 100%;
          border-collapse: collapse;
          background: #020617;
          border-radius: 12px;
          overflow: hidden;
          border: 1px solid #1f2937;
        }}
        thead {{
          background: #020617;
        }}
        th, td {{
          padding: 8px 10px;
          font-size: 13px;
        }}
        th {{
          text-align: left;
          color: #9ca3af;
          border-bottom: 1px solid #1f2937;
          background: #020617;
          position: sticky;
          top: 0;
          z-index: 1;
        }}
        tbody tr:nth-child(odd) {{
          background: #020617;
        }}
        tbody tr:nth-child(even) {{
          background: #020617;
        }}
        tbody tr:hover {{
          background: #111827;
        }}
        .footer {{
          margin-top: 16px;
          font-size: 12px;
          color: #6b7280;
        }}
      </style>
      <script>
        function toggleDetail(id) {{
          const row = document.getElementById(id);
          if (!row) return;
          row.style.display = (row.style.display === 'none' || row.style.display === '') ? 'table-row' : 'none';
        }}
      </script>
    </head>
    <body>
      <div class="container">
        <h1>AI Forex Forecaster</h1>
        <div class="subtitle">
          Dashboard · device: <b>{device}</b> · seq_len: <b>{seq_len}</b> · horizon: <b>{horizon}</b> · max_bars: <b>{MAX_BARS_ALLOWED}</b>
        </div>

        <div class="cards">
          <div class="card">
            <div class="card-title">Total Requests</div>
            <div class="card-value">{total}</div>
          </div>
          <div class="card">
            <div class="card-title">Success</div>
            <div class="card-value" style="color:#22c55e;">{ok_count}</div>
          </div>
          <div class="card">
            <div class="card-title">Errors</div>
            <div class="card-value" style="color:#f97316;">{err_count}</div>
          </div>
          <div class="card">
            <div class="card-title">Last Request</div>
            <div class="card-value">{format_ts(last_ts)}</div>
          </div>
        </div>

        <div class="pill" style="margin-bottom:8px;">
          <span>Model dir:</span>
          <span style="color:#e5e7eb;">{str(BASE_DIR) if BASE_DIR else "N/A"}</span>
        </div>

        <h2 style="font-size:16px;margin:12px 0;">Request History</h2>
        <div style="max-height:480px;overflow:auto;border-radius:12px;">
          <table>
            <thead>
              <tr>
                <th style="min-width:140px;">Time</th>
                <th style="min-width:120px;">IP</th>
                <th style="text-align:right;min-width:70px;">Calls (IP)</th>
                <th>Status</th>
                <th style="text-align:right;min-width:60px;">Bars</th>
                <th style="min-width:80px;">Bars Check</th>
                <th style="min-width:140px;">Last Bar Time</th>
                <th style="text-align:right;min-width:80px;">Latency (ms)</th>
                <th>Error</th>
                <th style="min-width:70px;">Details</th>
              </tr>
            </thead>
            <tbody>
              {''.join(html_rows) if html_rows else '<tr><td colspan="10" style="text-align:center;padding:20px;color:#6b7280;">No requests yet.</td></tr>'}
            </tbody>
          </table>
        </div>

        <div class="footer">
          Raw JSON endpoints: <code>/health</code>, <code>/history</code>
        </div>
      </div>
    </body>
    </html>
    """
    return HTMLResponse(html_page)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)

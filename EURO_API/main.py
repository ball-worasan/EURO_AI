# main.py
import os
import json
import html
import time
from datetime import datetime

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from config import BASE_DIR, seq_len, horizon, MAX_BARS_ALLOWED, device
from ml_models import predict_next_from_rates
from schemas import PredictRequest, OHLCVBar
from logging_utils import REQUEST_LOG, log_request_memory, format_ts
from db_logging import init_db, log_request_db


# สร้างตาราง DB ตอน start
init_db()

app = FastAPI(title="AI Forex Forecaster (with detailed dashboard + DB logs)")


@app.get("/health")
def health():
    total = len(REQUEST_LOG)
    ok_count = sum(1 for r in REQUEST_LOG if r.get("status") == "ok")
    err_count = sum(1 for r in REQUEST_LOG if r.get("status") == "error")
    last_ts = REQUEST_LOG[-1]["ts"] if REQUEST_LOG else None

    return {
        "status": "ok" if (BASE_DIR and seq_len > 0) else "config_error",
        "device": device,
        "base_dir": str(BASE_DIR) if BASE_DIR else None,
        "seq_len": seq_len,
        "horizon": horizon,
        "total_requests": total,
        "ok_requests": ok_count,
        "error_requests": err_count,
        "last_request_at": format_ts(last_ts),
        "max_bars_allowed": MAX_BARS_ALLOWED,
    }


def bars_to_dataframe(bars: list[OHLCVBar]) -> pd.DataFrame:
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
    user_agent = request.headers.get("user-agent")
    method = request.method

    # body ที่ส่งเข้ามาทั้งก้อน (สำหรับเก็บลง DB)
    req_body_dict = req.dict()

    # เช็ค completeness เรื่องจำนวนแท่ง
    bars_expected_min = seq_len
    bars_ok_min = bars_count >= bars_expected_min
    bars_ok_exact_seq = bars_count == bars_expected_min
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

        # meta ของผลลัพธ์ (สำหรับเก็บ log)
        res_meta = {
            "Open_next_pred": out.get("Open_next_pred"),
            "High_next_pred": out.get("High_next_pred"),
            "Low_next_pred": out.get("Low_next_pred"),
            "Close_next_pred": out.get("Close_next_pred"),
            "gap_pred": out.get("gap_pred"),
            "range_pred": out.get("range_pred"),
            "body_pred": out.get("body_pred"),
        }

        core_log = {
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

        # log ลง memory (dashboard)
        log_request_memory(core_log)

        # log ลง DB (ละเอียด)
        log_request_db(
            core_log=core_log,
            request_body=req_body_dict,
            response_body=out,
            method=method,
            user_agent=user_agent,
        )

        return out

    except Exception as e:
        latency_ms = (time.perf_counter() - ts_start) * 1000.0
        err_text = str(e)

        core_log = {
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

        # memory log
        log_request_memory(core_log)

        # DB log (response_body = None หรือ error detail ก็ได้)
        error_response = {"detail": err_text}
        log_request_db(
            core_log=core_log,
            request_body=req_body_dict,
            response_body=error_response,
            method=method,
            user_agent=user_agent,
        )

        raise HTTPException(status_code=400, detail=err_text)


# =========================
# DASHBOARD (API JSON)
# =========================


@app.get("/history")
def history():
    """คืน log เป็น JSON ละเอียดจาก in-memory (เร็ว ๆ)"""
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
    from logging_utils import REQUEST_LOG, format_ts  # ป้องกัน circular (safe อยู่แล้ว)

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
        detail_json = html.escape(json.dumps(detail_data, ensure_ascii=False, indent=2))
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
            <div class="card-title">Total Requests (memory)</div>
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

        <h2 style="font-size:16px;margin:12px 0;">Request History (in-memory)</h2>
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
          Raw JSON endpoints: <code>/health</code>, <code>/history</code> ·
          DB file: <code>request_logs.db</code> (SQLite)
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

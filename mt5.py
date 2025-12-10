import os
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import pandas as pd
import requests
from dotenv import load_dotenv

# =========================
# LOAD ENV + DEBUG ENV
# =========================
load_dotenv()


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


log("===== MT5 FETCH & AI PREDICT START =====")
log(f"MT5_LOGIN = {os.getenv('MT5_LOGIN')}")
log(f"MT5_SERVER = {os.getenv('MT5_SERVER')}")
log(f"AI_BASE_URL = {os.getenv('AI_BASE_URL')}")
log(f"MODEL_WARMUP_BARS = {os.getenv('MODEL_WARMUP_BARS')}")

# =========================
# CONFIG
# =========================

MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "IUXGlobal-Real")

AI_BASE_URL = os.getenv("AI_BASE_URL", "http://127.0.0.1:8000")

# meta ของคุณใช้ 24 แท่ง
SEQ_LEN_FIXED = 24

# warmup สำหรับ indicator
WARMUP_BARS = int(os.getenv("MODEL_WARMUP_BARS", "200"))

TOTAL_BARS_REQUIRED = SEQ_LEN_FIXED + WARMUP_BARS

# =========================
# MT5 HELPERS
# =========================


def mt5_init_and_login():
    log("Initializing MT5...")
    if not mt5.initialize():
        raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")

    if MT5_LOGIN == 0 or MT5_PASSWORD == "":
        raise RuntimeError("❌ Please set MT5_LOGIN and MT5_PASSWORD in .env")

    log("Logging in to MT5...")
    authorized = mt5.login(
        login=MT5_LOGIN,
        password=MT5_PASSWORD,
        server=MT5_SERVER,
    )

    if not authorized:
        err = mt5.last_error()
        mt5.shutdown()
        raise RuntimeError(f"❌ MT5 login failed: {err}")

    log(f"✅ MT5 login success: {MT5_LOGIN} @ {MT5_SERVER}")


def ensure_symbol(symbol: str):
    """เปิด symbol ให้พร้อมใช้งาน ถ้าไม่เจอให้โยน error"""
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"❌ Symbol {symbol} not found in MT5")

    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"❌ Cannot select symbol {symbol}")

    print(f"✅ Symbol {symbol} ready.")


# =========================
# FETCH DATA
# =========================

def fetch_eurusd_d1_for_model(symbol: str) -> pd.DataFrame:
    """
    ✅ ดึง D1 ย้อนหลัง:
    - ใช้จริง = 24 แท่ง
    - ดึงจาก MT5 = 24 + warmup
    - จบที่ 'เมื่อวาน'
    - columns = Time, Open, High, Low, Close, Volume, Spread
    """
    log(f"SEQ_LEN (model) = {SEQ_LEN_FIXED}")
    log(f"WARMUP_BARS = {WARMUP_BARS}")
    log(f"TOTAL_BARS_REQUIRED = {TOTAL_BARS_REQUIRED}")

    utc_now = datetime.now(timezone.utc)
    yesterday_date = utc_now.date() - timedelta(days=1)

    days_back = TOTAL_BARS_REQUIRED + 10
    utc_from = utc_now - timedelta(days=days_back)

    log(f"Fetching MT5 D1 from {utc_from} to {utc_now}")

    rates = mt5.copy_rates_range(
        symbol,
        mt5.TIMEFRAME_D1,
        utc_from,
        utc_now,
    )

    if rates is None or len(rates) == 0:
        raise RuntimeError("❌ MT5 returned no D1 data")

    log(f"MT5 raw bars received = {len(rates)}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    df = df[df["time"].dt.date <= yesterday_date]
    log(f"Bars after filter to yesterday = {len(df)}")

    if len(df) < TOTAL_BARS_REQUIRED:
        raise RuntimeError(
            f"❌ Not enough D1 bars. Need {TOTAL_BARS_REQUIRED}, got {len(df)}"
        )

    df = df.tail(TOTAL_BARS_REQUIRED).copy()
    df = df.set_index("time")

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

    # ✅ ตรงกับ meta ของคุณเป๊ะ
    df = df[["Open", "High", "Low", "Close", "Volume", "Spread"]]

    log("✅ Final MT5 dataframe ready")
    log(f"Final bars = {len(df)}")
    log(f"First bar time = {df.index[0]}")
    log(f"Last bar time = {df.index[-1]}")

    log("Sample last 3 bars:")
    print(df.tail(3))

    return df


# =========================
# CALL AI MODEL API
# =========================

def dataframe_to_bars_payload(df: pd.DataFrame):
    bars = []
    for ts, row in df.iterrows():
        bars.append(
            {
                "time": ts.isoformat(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "spread": float(row["Spread"]),
                "volume": float(row["Volume"]),
            }
        )
    log(f"Payload bars count = {len(bars)}")
    return {"bars": bars}


def call_ai_predict(df: pd.DataFrame):
    payload = dataframe_to_bars_payload(df)
    url = f"{AI_BASE_URL}/predict"

    log(f"Calling AI API: {url}")
    resp = requests.post(url, json=payload, timeout=15)

    log(f"AI Response status = {resp.status_code}")

    if resp.status_code != 200:
        log("❌ AI API ERROR RESPONSE:")
        print(resp.text)
        resp.raise_for_status()

    return resp.json()


def resolve_symbol_name(base: str = "EURUSD") -> str:
    """
    พยายามหา symbol จริงจาก MT5 ที่ตรงกับ base เช่น 'EURUSD'
    - ก่อนอื่นดูใน env: MT5_SYMBOL ถ้ามีใช้เลย
    - ถ้าไม่มี env ให้สแกน symbols_get() หา:
      1) ชื่อเท่ากับ base พอดี
      2) ชื่อขึ้นต้นด้วย base (EURUSD*, เช่น EURUSD.i)
      3) ชื่อมี base อยู่ข้างใน (เผื่อเคสแปลก ๆ)
    """
    env_symbol = os.getenv("MT5_SYMBOL")
    if env_symbol:
        print(f"[DEBUG] Using MT5_SYMBOL from env: {env_symbol}")
        return env_symbol

    all_symbols = mt5.symbols_get()
    names = [s.name for s in all_symbols]

    print(f"[DEBUG] Total symbols in MT5: {len(names)}")
    # ลอง log ตัวอย่าง symbol ที่มี EURUSD ให้ดู
    candidates = [n for n in names if "EURUSD" in n.upper()]
    if candidates:
        print("[DEBUG] Found EURUSD-like symbols:", candidates[:10])

    # 1) ตรงเป๊ะ
    if base in names:
        return base

    # 2) เริ่มต้นด้วย base
    for n in names:
        if n.upper().startswith(base.upper()):
            print(f"[DEBUG] Auto-picked symbol: {n}")
            return n

    # 3) มี base อยู่ข้างใน
    for n in names:
        if base.upper() in n.upper():
            print(f"[DEBUG] Auto-picked symbol (contains): {n}")
            return n

    raise RuntimeError(
        f"❌ Cannot find any symbol matching base '{base}'. "
        "Try setting MT5_SYMBOL=YOUR_SYMBOL in .env"
    )


# =========================
# MAIN FLOW
# =========================

def main():
    # 1) เปิด + login MT5
    mt5_init_and_login()

    try:
        # 2) resolve symbol name
        base = "EURUSD"
        symbol = resolve_symbol_name(base)
        print(f"[INFO] Using symbol: {symbol}")

        # 3) เปิด symbol
        ensure_symbol(symbol)

        # 4) ดึง D1 ย้อนหลัง จบที่ 'เมื่อวาน' ให้แท่งพอใช้
        df = fetch_eurusd_d1_for_model(symbol)
        print(f"Fetched {len(df)} D1 bars, last bar time = {df.index[-1]}")

        # 5) ส่งเข้า AI model
        result = call_ai_predict(df)
        print("AI prediction result:")
        print(result)

    finally:
        print("[INFO] Shutting down MT5...")
        mt5.shutdown()
        print("[INFO] ✅ MT5 shutdown complete.")


if __name__ == "__main__":
    main()

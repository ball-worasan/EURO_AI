# logging_utils.py
from collections import deque
from datetime import datetime
from typing import Deque, Dict, Any

# เก็บ log ล่าสุดไม่เกิน 1000 รายการ
REQUEST_LOG: Deque[Dict[str, Any]] = deque(maxlen=1000)


def log_request_memory(entry: Dict[str, Any]) -> None:
    """เก็บ log ไว้ในหน่วยความจำ"""
    REQUEST_LOG.append(entry)


def format_ts(ts: datetime | None) -> str:
    if not ts:
        return "-"
    return ts.strftime("%Y-%m-%d %H:%M:%S")

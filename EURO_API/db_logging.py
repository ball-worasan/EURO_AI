# db_logging.py
import json
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

# SQLite local file
DATABASE_URL = "sqlite:///./request_logs.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # สำหรับ SQLite + multi thread
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class RequestLog(Base):
    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True, index=True)
    ts = Column(DateTime, index=True)
    ip = Column(String(64), index=True)
    xff = Column(String(256))
    path = Column(String(256), index=True)
    method = Column(String(16))
    user_agent = Column(String(512))

    status = Column(String(32), index=True)
    bars = Column(Integer)
    bars_expected_min = Column(Integer)
    bars_ok_min = Column(Integer)
    bars_ok_exact_seq_len = Column(Integer)
    bars_ok_max = Column(Integer)
    last_time = Column(String(64))
    latency_ms = Column(Float)

    error = Column(Text)

    # เก็บข้อมูลดิบเป็น JSON string
    req_body = Column(Text)  # body ที่ client ส่งมา (bars ฯลฯ)
    req_meta = Column(Text)  # meta ต่าง ๆ
    req_bars_sample = Column(Text)  # ตัวอย่าง bar first/last
    res_meta = Column(Text)  # meta ของผลลัพธ์
    res_body = Column(Text)  # ผลลัพธ์เต็ม ๆ ที่ส่งกลับไป


def init_db() -> None:
    """สร้างตาราง ถ้ายังไม่มี"""
    Base.metadata.create_all(bind=engine)


def _json_dumps_safe(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return json.dumps({"_error": "json_dump_failed"}, ensure_ascii=False)


def log_request_db(
    core_log: Dict[str, Any],
    request_body: Optional[Dict[str, Any]],
    response_body: Optional[Dict[str, Any]],
    method: str,
    user_agent: Optional[str],
) -> None:
    """
    core_log = dict ที่เราใช้ใน in-memory log (ip, status, bars ฯลฯ)
    request_body = req.dict()
    response_body = out หรือ None เวลา error
    """
    db = SessionLocal()
    try:
        obj = RequestLog(
            ts=core_log.get("ts", datetime.utcnow()),
            ip=core_log.get("ip"),
            xff=core_log.get("xff"),
            path=core_log.get("path"),
            method=method,
            user_agent=user_agent or "",
            status=core_log.get("status"),
            bars=core_log.get("bars"),
            bars_expected_min=core_log.get("bars_expected_min"),
            bars_ok_min=(
                int(bool(core_log.get("bars_ok_min")))
                if core_log.get("bars_ok_min") is not None
                else None
            ),
            bars_ok_exact_seq_len=(
                int(bool(core_log.get("bars_ok_exact_seq_len")))
                if core_log.get("bars_ok_exact_seq_len") is not None
                else None
            ),
            bars_ok_max=(
                int(bool(core_log.get("bars_ok_max")))
                if core_log.get("bars_ok_max") is not None
                else None
            ),
            last_time=core_log.get("last_time"),
            latency_ms=core_log.get("latency_ms"),
            error=core_log.get("error"),
            req_body=_json_dumps_safe(request_body),
            req_meta=_json_dumps_safe(core_log.get("req_meta")),
            req_bars_sample=_json_dumps_safe(core_log.get("req_bars_sample")),
            res_meta=_json_dumps_safe(core_log.get("res_meta")),
            res_body=_json_dumps_safe(response_body),
        )
        db.add(obj)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[DB LOG ERROR] {e}")
    finally:
        db.close()

# config.py
import os
import json
from pathlib import Path

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
    target_names = meta["targets_boosting"]
else:
    meta = {}
    feature_cols = []
    seq_len = 0
    horizon = 0
    target_names = []
    print("[WARN] META file not found. Set EURO_AI_BASE_DIR correctly.")

# =========================
# Device
# =========================
# `torch` is an optional dependency for some workflows. Guard the import so the
# module can be imported in editors/CI even when `torch` isn't installed.
try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
else:
    # Fallback: allow overriding via env var, default to CPU. This avoids crashes
    # when opening files in editors that don't have the project's venv active.
    device = os.getenv("EURO_AI_DEVICE", "cpu")
    print("[WARN] 'torch' not available; using device=", device)

# schemas.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List


class OHLCVBar(BaseModel):
    time: datetime = Field(..., description="Time of the bar (ISO8601)")
    open: float
    high: float
    low: float
    close: float
    spread: float = 0.0
    volume: float = 0.0  # ให้ตรง meta


class PredictRequest(BaseModel):
    bars: List[OHLCVBar]

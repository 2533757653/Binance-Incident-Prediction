"""
Builder-B 输出 Signal 模型。

严格遵循 .workspace/iter-1/dataContract.md §1.4。
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.common.schemas import Signal as _CommonSignal  # noqa: F401  re-export


# 重新暴露 dataContract §1.4 字段，便于下游引用一致
class Signal(BaseModel):
    """Builder-B 产出的信号对象（与 src.common.schemas.Signal 同源）。"""

    signal_id: str
    ts: datetime
    event_id: str
    event_title: str
    symbol: str
    side: Literal["YES", "NO"]
    entry_price: float = Field(ge=0.01, le=0.99)
    spread: float = Field(ge=0.001)
    confidence: float = Field(ge=0.0, le=1.0)
    expected_value: float = Field(ge=-1.0, le=1.0)
    win_prob: float = Field(ge=0.0, le=1.0)
    payoff_ratio: float = Field(gt=0)
    rationale: str = Field(max_length=80)
    strategy: str = "factor_v1"
    factors: dict
    ttl_seconds: int = Field(gt=0, default=60)
    expire_at: datetime

    @field_validator("factors")
    @classmethod
    def _factors_min_keys(cls, v: dict) -> dict:
        required = {"mom_15m", "bb_pct_1h", "atr_break_4h", "spread", "time_to_settle_min"}
        missing = required - set(v.keys())
        if missing:
            raise ValueError(f"factors dict missing required keys: {missing}")
        return v

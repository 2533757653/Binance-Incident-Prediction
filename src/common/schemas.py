"""
Horizon-Incident 跨 Builder 数据结构（Pydantic v2）。
严格遵循 .workspace/iter-1/dataContract.md。
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ============================================================
# §1.1 PriceTick
# ============================================================
class PriceTick(BaseModel):
    """Binance 现货 WebSocket 成交 tick。"""

    ts: datetime
    symbol: Literal["BTCUSDT", "ETHUSDT"]
    price: float = Field(gt=0)
    qty: float = Field(gt=0)
    trade_id: int = Field(ge=0)
    is_buyer_maker: bool


# ============================================================
# §1.2 KlineBar
# ============================================================
class KlineBar(BaseModel):
    """Binance K 线（含回测用历史）。"""

    symbol: str
    interval: Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    open_time: datetime
    close_time: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    quote_volume: float = Field(ge=0)
    trades_count: int = Field(ge=0)


# ============================================================
# §1.3 EventContract
# ============================================================
class EventContract(BaseModel):
    """Binance 事件合约活跃事件。"""

    event_id: str
    symbol: str
    title: str
    underlying: Literal["BTC", "ETH"]
    strike_price: float = Field(gt=0)
    direction: Literal["ABOVE", "BELOW"]
    time_to_expiry: Literal["1h", "4h", "1d"]
    settle_time: datetime
    current_yes_price: float = Field(ge=0.01, le=0.99)
    current_no_price: float = Field(ge=0.01, le=0.99)
    yes_bid: float = Field(gt=0)
    yes_ask: float = Field(gt=0)
    no_bid: float = Field(gt=0)
    no_ask: float = Field(gt=0)
    spread: float = Field(ge=0.001)
    volume_24h: float = 0.0
    open_interest: float = 0.0
    status: Literal["TRADING", "EXPIRED", "PENDING"] = "TRADING"

    @field_validator("yes_ask")
    @classmethod
    def _ask_gt_bid(cls, v: float, info) -> float:
        bid = info.data.get("yes_bid")
        if bid is not None and v <= bid:
            raise ValueError(f"yes_ask ({v}) must be > yes_bid ({bid})")
        return v

    @field_validator("no_ask")
    @classmethod
    def _no_ask_gt_bid(cls, v: float, info) -> float:
        bid = info.data.get("no_bid")
        if bid is not None and v <= bid:
            raise ValueError(f"no_ask ({v}) must be > no_bid ({bid})")
        return v


# ============================================================
# §1.4 Signal
# ============================================================
class Signal(BaseModel):
    """Builder-B 输出：信号对象。"""

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
    expected_value: float = Field(ge=-1.0, le=10.0)

    @field_validator("factors")
    @classmethod
    def _factors_min_keys(cls, v: dict) -> dict:
        # iter-6: 放宽约束，只要求 time_to_settle_min（其他因子随策略变）
        required = {"time_to_settle_min"}
        missing = required - set(v.keys())
        if missing:
            raise ValueError(f"factors dict missing required key: {missing}")
        return v


# ============================================================
# §1.5 BacktestResult
# ============================================================
class BacktestResult(BaseModel):
    """Builder-C 输出：回测结果。"""

    strategy: str
    symbol: str
    start_time: datetime
    end_time: datetime
    total_signals: int = Field(ge=0)
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    pushes: int = Field(ge=0)
    win_rate: float = Field(ge=0.0, le=1.0)
    avg_payoff_ratio: float = Field(gt=0)
    expected_value: float = Field(ge=-1.0, le=1.0)
    total_pnl_pct: float
    max_drawdown_pct: float = Field(ge=0)
    signals_per_day: float = Field(ge=0)
    avg_latency_ms: float = Field(ge=0)
    config_snapshot: dict
    factor_contributions: dict = Field(default_factory=dict)


def make_signal_id(event_id: str, ts: datetime) -> str:
    """生成 sig-{event_id}-{ts_serial}。"""
    serial = ts.strftime("%Y%m%d%H%M%S")
    return f"sig-{event_id}-{serial}"

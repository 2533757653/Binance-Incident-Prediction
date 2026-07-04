"""进程内消息总线（re-export 自 ``src.common.bus``）。

数据接入层把 PriceTick / EventContract 推入 bus；Builder-B 从这里消费。
"""

from __future__ import annotations

from src.common.bus import (  # noqa: F401
    event_queue,
    price_queue,
    signal_queue,
)

__all__ = [
    "event_queue",
    "price_queue",
    "signal_queue",
]

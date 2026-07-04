"""
信号去重：同一标的 + 同一方向，TTL 时间内只保留最高 confidence 那条。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class SignalDedup:
    """
    按 (symbol, side) 缓存最近一次信号的时间。
    同一 (symbol, side) 在 ttl_seconds 内只让 1 条通过；不同则不合并。
    """
    ttl_seconds: int = 3600  # 1 小时
    _last: dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)
    # _last[(symbol, side)] = (timestamp, confidence)

    def should_emit(self, symbol: str, side: str, confidence: float, now: float | None = None) -> tuple[bool, str]:
        """
        返回 (should_emit, reason)。
        - true + "fresh"：新信号，输出
        - true + "higher_conf"：同方向但 confidence 更高，覆盖
        - false + "duplicate_lower"：同方向但 confidence 更低，丢弃
        - false + "opposite_side"：反向信号（让两个方向同时存在）
        """
        now = now if now is not None else time.time()
        key = (symbol, side)
        last = self._last.get(key)
        if last is None:
            self._last[key] = (now, confidence)
            return True, "fresh"
        last_ts, last_conf = last
        if now - last_ts > self.ttl_seconds:
            # TTL 过期 → 新信号
            self._last[key] = (now, confidence)
            return True, "ttl_expired"
        if confidence > last_conf:
            # 同方向 confidence 提升 → 覆盖
            self._last[key] = (now, confidence)
            return True, "higher_conf"
        # 同方向 lower conf → 丢弃
        return False, "duplicate_lower"

    def clear(self) -> None:
        self._last.clear()
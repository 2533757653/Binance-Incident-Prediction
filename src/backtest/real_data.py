"""
真实数据接入（生产级）：
- 多源拉取（Bybit→OKX→Gate，无需 Binance，国内可直连），见 src/data/sources.py
- 支持 90 天历史 + 自动缓存到 data/cache/
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from src.common.schemas import EventContract, KlineBar


_TZ_CN = timezone(timedelta(hours=8))
log = logging.getLogger(__name__)

# 每根 K 线对应的分钟数
_INTERVAL_MIN = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "4h": 240, "1d": 1440,
}


def _ts_to_sh(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=_TZ_CN)


def _interval_to_ms(interval: str) -> int:
    """K 线间隔 → 毫秒（仍被 src/realtime/incremental_klines.py 引用）。"""
    return _INTERVAL_MIN[interval] * 60 * 1000


def fetch_klines_paginated(
    symbol: str,
    interval: str,
    *,
    days: int,
    cache_dir: str = "data/cache",
    force_refresh: bool = False,
    max_retries: int = 3,
) -> List[KlineBar]:
    """拉指定天数的 K 线（多源 + 缓存）。"""
    if interval not in _INTERVAL_MIN:
        raise ValueError(f"unsupported interval: {interval}")
    cache_path = Path(cache_dir) / f"{symbol}_{interval}_{days}d.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # 读缓存
    if not force_refresh and cache_path.exists():
        log.info("[real_data] 缓存命中 %s", cache_path)
        return _parse_klines(cache_path.read_text(encoding="utf-8"), interval, symbol)

    # 多源拉取（Bybit→OKX→Gate，无需 Binance）
    from src.data.sources import fetch_klines_rows
    deduped = fetch_klines_rows(symbol, interval, days=days)

    # 缓存（沿用 Binance 同款 9 字段数组格式，旧缓存兼容）
    cache_path.write_text(json.dumps(deduped), encoding="utf-8")
    log.info("[real_data] %s %s 拉取完成，%d 根，缓存 -> %s",
             symbol, interval, len(deduped), cache_path)
    return _parse_klines(json.dumps(deduped), interval, symbol)


def _parse_klines(raw_json: str, interval: str, symbol: str) -> List[KlineBar]:
    rows = json.loads(raw_json)
    out: List[KlineBar] = []
    for k in rows:
        out.append(KlineBar(
            symbol=symbol,
            interval=interval,
            open_time=_ts_to_sh(int(k[0])),
            close_time=_ts_to_sh(int(k[6])),
            open=float(k[1]),
            high=float(k[2]),
            low=float(k[3]),
            close=float(k[4]),
            volume=float(k[5]),
            quote_volume=float(k[7]),
            trades_count=int(k[8]),
        ))
    return out


def build_events_from_klines(
    klines: List[KlineBar],
    *,
    symbol: str,
    direction: str = "ABOVE",
    strike_offset_pct: float = 0.02,
    event_minutes: Optional[int] = None,
) -> List[EventContract]:
    """
    从 K 线构造事件合约列表。

    event_minutes: 事件窗口（默认从 kline 的 interval 推断）。
    """
    if event_minutes is None:
        event_minutes = _INTERVAL_MIN.get(klines[0].interval if klines else "1h", 60)
    import random
    rng = random.Random(hash(symbol + str(event_minutes)) & 0xFFFFFFFF)
    out: List[EventContract] = []
    tf_label = f"{event_minutes // 60}h" if event_minutes >= 60 else f"{event_minutes}m"
    for bar in klines[:-1]:
        offset = (rng.random() * 2 - 1) * strike_offset_pct
        strike = round(bar.open * (1.0 + offset), 2)
        if direction == "ABOVE":
            edge = (bar.close - strike) / max(strike, 1.0)
            yes_prob = max(0.05, min(0.95, 0.5 + edge * 6))
        else:
            edge = (strike - bar.close) / max(strike, 1.0)
            yes_prob = max(0.05, min(0.95, 0.5 + edge * 6))
        spread = round(0.01 + rng.random() * 0.015, 4)
        yes_bid = round(yes_prob - spread / 2, 4)
        yes_ask = round(yes_prob + spread / 2, 4)
        no_bid = round(1 - yes_ask, 4)
        no_ask = round(1 - yes_bid, 4)
        # Pydantic 约束 time_to_expiry ∈ {"1h","4h","1d"}；30m 用 "1h" 兼容
        tf_field = "1h" if tf_label == "30m" else tf_label if tf_label in ("1h","4h","1d") else "1h"
        out.append(EventContract(
            event_id=f"{symbol}-{tf_field}-{direction}-{int(strike)}-{bar.open_time.strftime('%Y%m%d%H%M%S')}",
            symbol=symbol,
            title=f"{symbol[:3]} {tf_label} 后 {'≥' if direction=='ABOVE' else '<'} {int(strike)}?",
            underlying=symbol[:3],
            strike_price=strike,
            direction=direction,
            time_to_expiry=tf_field,
            settle_time=bar.close_time,
            current_yes_price=round((yes_bid + yes_ask) / 2, 4),
            current_no_price=round((no_bid + no_ask) / 2, 4),
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            spread=spread,
            volume_24h=1000.0,
            open_interest=500.0,
            status="TRADING",
        ))
    return out


def build_aggregated_klines(
    klines_30m: List[KlineBar],
    target_interval: str,
) -> List[KlineBar]:
    """从 30m K 线聚合成 1h / 4h / 1d。"""
    if target_interval not in ("1h", "4h", "1d"):
        raise ValueError(f"unsupported aggregation target: {target_interval}")
    if not klines_30m:
        return []
    ratio = _INTERVAL_MIN[target_interval] // _INTERVAL_MIN["30m"]
    out: List[KlineBar] = []
    sym = klines_30m[0].symbol
    for i in range(0, len(klines_30m) - ratio + 1, ratio):
        group = klines_30m[i:i + ratio]
        out.append(KlineBar(
            symbol=sym,
            interval=target_interval,
            open_time=group[0].open_time,
            close_time=group[-1].close_time,
            open=group[0].open,
            high=max(k.high for k in group),
            low=min(k.low for k in group),
            close=group[-1].close,
            volume=sum(k.volume for k in group),
            quote_volume=sum(k.quote_volume for k in group),
            trades_count=sum(k.trades_count for k in group),
        ))
    return out


def build_mixed_events_dual(
    klines: List[KlineBar],
    *,
    symbol: str,
    strike_offset_pct: float = 0.02,
    event_minutes: int = 30,
) -> List[EventContract]:
    """
    构造 30m 事件列表（half ABOVE, half BELOW）。
    """
    half = len(klines) // 2
    above = build_events_from_klines(klines[:half], symbol=symbol, direction="ABOVE", strike_offset_pct=strike_offset_pct, event_minutes=event_minutes)
    below = build_events_from_klines(klines[half:], symbol=symbol, direction="BELOW", strike_offset_pct=strike_offset_pct, event_minutes=event_minutes)
    # 交错合并
    merged = []
    for a, b in zip(above, below):
        merged.append(a)
        merged.append(b)
    return merged


def load_binance_klines(
    cache_path: str | Path,
    *,
    interval: str = "1h",
    symbol: str = "BTCUSDT",
) -> List[KlineBar]:
    rows = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    return _parse_klines(json.dumps(rows), interval, symbol)
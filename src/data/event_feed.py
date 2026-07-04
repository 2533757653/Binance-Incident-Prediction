"""事件合约接入（Builder-A 产出）。

数据源：币安 eapi（``https://eapi.binance.com/eapi/v1/``）—— spec §0 实测国内直连阻塞，
      默认走代理；代理失败时降级到 ``src/data/event_contracts_sample.json``（兜底）。
- ``GET /eapi/v1/exchangeInfo``  —— 合约基础信息
- ``GET /eapi/v1/markets``       —— 当前活跃市场（含 strike / settle / bid-ask）

出口：``EventContract``（dataContract §1.3）写入 ``event_queue``，并提供
     ``fetch_active_events()`` 给 main.py / 信号层调用。
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from queue import Queue
from typing import Iterable, Optional

import httpx

from src.common.bus import event_queue
from src.common.errors import FeedError, ProxyError
from src.common.schemas import EventContract
from src.data.proxy import EXIT_PROXY_RETRY_FAIL, get_proxies, has_proxy

log = logging.getLogger(__name__)

_EAPI = "https://eapi.binance.com/eapi/v1"
_TZ_SH = timezone(timedelta(hours=8))
_FALLBACK_FILE = Path(__file__).resolve().parent / "event_contracts_sample.json"

# 本轮只做 1h / 4h
_ALLOWED_TF = {"1h", "4h"}


def _now_sh() -> datetime:
    return datetime.now(tz=_TZ_SH)


def _ts_to_sh(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=_TZ_SH)


# ============================================================
# 退避
# ============================================================
def _backoff(attempt: int, schedule: list[int]) -> None:
    if attempt >= len(schedule):
        return
    wait = schedule[attempt]
    log.warning("[event_feed] 第 %d 次失败，%ds 后重试 …", attempt + 1, wait)
    time.sleep(wait)


# ============================================================
# eapi REST 客户端
# ============================================================
class EapiClient:
    def __init__(self, proxies: Optional[dict[str, str]] = None, timeout: float = 8.0) -> None:
        self.proxies = proxies or get_proxies()
        self.timeout = timeout
        # httpx ≥ 0.28: proxies 改为 transport.proxy
        proxy_url = (self.proxies or {}).get("https://") or (self.proxies or {}).get("http://")
        transport = httpx.HTTPTransport(proxy=proxy_url) if proxy_url else None
        self._client = httpx.Client(timeout=timeout, transport=transport)

    def close(self) -> None:
        self._client.close()

    # ---------------------------------------------------------- /exchangeInfo
    def exchange_info(self) -> dict:
        r = self._client.get(f"{_EAPI}/exchangeInfo")
        r.raise_for_status()
        return r.json()

    # ---------------------------------------------------------- /markets
    def markets(self, symbol: Optional[str] = None) -> list[dict]:
        params: dict = {}
        if symbol:
            params["symbol"] = symbol
        r = self._client.get(f"{_EAPI}/markets", params=params)
        r.raise_for_status()
        return r.json() or []


# ============================================================
# 解析
# ============================================================
def _to_event_id(symbol: str, tf: str, direction: str, strike: float, settle: datetime) -> str:
    """dataContract §0.7：{SYMBOL}-{TF}-{DIRECTION}-{STRIKE}-{TS}。"""
    return f"{symbol}-{tf}-{direction}-{int(strike)}-{settle.strftime('%Y%m%d%H%M%S')}"


def _parse_market_row(row: dict) -> Optional[EventContract]:
    """把 eapi /markets 的一条记录解析为 EventContract。"""
    try:
        symbol = (row.get("symbol") or "").upper()
        if symbol not in {"BTCUSDT", "ETHUSDT"}:
            return None
        tf = (row.get("timeToExpiry") or "").lower()
        if tf not in _ALLOWED_TF:
            return None
        direction = (row.get("side") or "").upper()
        if direction not in {"ABOVE", "BELOW"}:
            return None

        strike = float(row["strikePrice"])
        settle_ms = int(row["expireTime"])
        settle = _ts_to_sh(settle_ms)

        # /markets 返回的报价字段（按 2024 eapi 文档）
        yes_bid = float(row.get("yesBidPrice") or row.get("bidPrice") or 0.0)
        yes_ask = float(row.get("yesAskPrice") or row.get("askPrice") or 0.0)
        no_bid = float(row.get("noBidPrice") or 0.0)
        no_ask = float(row.get("noAskPrice") or 0.0)

        # 兜底：如果只有一组 bid/ask，按 NO 价对称推 YES
        if yes_bid <= 0 or yes_ask <= 0:
            return None
        if no_bid <= 0 or no_ask <= 0:
            no_bid = max(0.01, round(1.0 - yes_ask, 4))
            no_ask = max(0.02, round(1.0 - yes_bid, 4))

        yes_mid = round((yes_bid + yes_ask) / 2, 4)
        no_mid = round((no_bid + no_ask) / 2, 4)
        spread = round(yes_ask - yes_bid, 4)
        if spread < 0.001:
            return None

        title = f"{symbol[:3]} {tf} 后 {'≥' if direction == 'ABOVE' else '<'} {int(strike)}?"
        underlying = symbol[:3]

        return EventContract(
            event_id=_to_event_id(symbol, tf, direction, strike, settle),
            symbol=symbol,
            title=title,
            underlying=underlying,
            strike_price=strike,
            direction=direction,
            time_to_expiry=tf,
            settle_time=settle,
            current_yes_price=yes_mid,
            current_no_price=no_mid,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            spread=spread,
            volume_24h=float(row.get("volume") or 0.0),
            open_interest=float(row.get("openInterest") or 0.0),
            status="TRADING",
        )
    except Exception as e:  # noqa: BLE001
        log.warning("[event_feed] 解析失败，跳过: %s | row=%s", e, row)
        return None


# ============================================================
# 拉取（含重试 + 降级）
# ============================================================
def fetch_active_events(
    client: EapiClient,
    *,
    backoff_sec: tuple[int, ...] = (2, 5, 10),
    symbols: Iterable[str] = ("BTCUSDT", "ETHUSDT"),
) -> list[EventContract]:
    """拉取当前活跃的 1h/4h 事件。代理失败 → 走兜底 sample。

    返回：过滤后的 ``EventContract`` 列表。"""
    if not has_proxy(client.proxies):
        log.warning("[event_feed] 无代理配置（币安 eapi 国内直连阻塞），降级到 sample 兜底")
        return _load_fallback(symbols)

    last_err: Optional[Exception] = None
    max_attempts = len(backoff_sec) + 1
    for attempt in range(max_attempts):
        if attempt > 0:
            _backoff(attempt - 1, list(backoff_sec))
        try:
            events: list[EventContract] = []
            for sym in symbols:
                rows = client.markets(symbol=sym)
                for row in rows:
                    ev = _parse_market_row(row)
                    if ev is not None:
                        events.append(ev)
            if not events:
                last_err = FeedError("markets 返回 0 条")
                log.warning("[event_feed] attempt=%d markets 0 条", attempt + 1)
                continue
            log.info("[event_feed] 拉取到 %d 条活跃事件", len(events))
            return events
        except (httpx.ProxyError, httpx.ConnectError, ProxyError) as e:
            last_err = e
            log.error("[event_feed] 代理失败 attempt=%d: %s", attempt + 1, e)
            continue
        except httpx.HTTPStatusError as e:
            last_err = e
            code = e.response.status_code
            log.error("[event_feed] HTTP %s attempt=%d: %s", code, attempt + 1, e)
            if code in (429, 418, 503):
                continue
            raise FeedError(f"eapi HTTP {code}: {e}") from e
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.exception("[event_feed] 未知错误 attempt=%d", attempt + 1)
            continue

    log.error("[event_feed] %d 次重试全部失败，降级到 sample 兜底: %s",
              max_attempts, last_err)
    return _load_fallback(symbols)


def _load_fallback(symbols: Iterable[str]) -> list[EventContract]:
    sym_set = {s.upper() for s in symbols}
    if not _FALLBACK_FILE.exists():
        log.error("[event_feed] 兜底文件不存在: %s", _FALLBACK_FILE)
        return []
    try:
        with _FALLBACK_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        log.error("[event_feed] 读兜底文件失败: %s", e)
        return []
    out: list[EventContract] = []
    for row in data:
        try:
            ev = EventContract(**row)
            if ev.symbol.upper() in sym_set and ev.time_to_expiry in _ALLOWED_TF:
                out.append(ev)
        except Exception as e:  # noqa: BLE001
            log.warning("[event_feed] 兜底数据校验失败: %s | %s", e, row)
    log.warning("[event_feed] 兜底模式生效，加载 %d 条", len(out))
    return out


def push_to_queue(events: list[EventContract], q: Optional[Queue] = None) -> int:
    """写 bus 队列。返回实际写入条数。"""
    q = q or event_queue
    n = 0
    for ev in events:
        try:
            q.put_nowait(ev)
            n += 1
        except Exception:  # noqa: BLE001
            pass
    return n


__all__ = [
    "EapiClient",
    "fetch_active_events",
    "push_to_queue",
]

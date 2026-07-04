"""K 线接入（Builder-D 产出 · iter-2 增强版）。

数据源：币安现货 REST（无需鉴权）
    GET https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=720

支持 interval：``1m`` / ``5m`` / ``15m`` / ``1h`` / ``4h`` / ``1d``
缓存：``data/cache/{symbol}_{interval}_klines.json``，TTL = 24h

返回：``list[KlineBar]``（对齐 dataContract §1.2）
- 直连 httpx（Clash TUN 模式已接管，proxies=None）
- 失败显式抛 ``FeedError``，**不再走 sample 兜底**

导出：
- ``KlinesFetcher`` —— 拉取 + 缓存 + Pydantic 校验
- ``fetch_klines()`` —— 快捷函数
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable, Optional

import httpx
from pydantic import ValidationError

from src.common.errors import FeedError
from src.common.schemas import KlineBar

log = logging.getLogger(__name__)

# 端点（iter-2 已实测 TUN 直连 4.8s 通）
_BINANCE_API = "https://api.binance.com/api/v3/klines"

# 缓存
_CACHE_DIR = Path(__file__).resolve().parent / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_TTL_SEC = 24 * 3600

# 超时（spec §2.1）
_TIMEOUT_SEC = 10.0

# 支持的 interval（dataContract §1.2）
_ALLOWED_INTERVALS = {"1m", "5m", "15m", "1h", "4h", "1d"}

# Binance interval → 秒数（用于推 close_time）
_INTERVAL_SEC = {
    "1m": 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
    "1d": 24 * 60 * 60,
}

# 时区：上海 UTC+8（dataContract §0.1）
_TZ_SH = timezone(timedelta(hours=8))


def _now_sh() -> datetime:
    return datetime.now(tz=_TZ_SH)


def _ts_to_sh(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=_TZ_SH)


def _cache_path(symbol: str, interval: str) -> Path:
    return _CACHE_DIR / f"{symbol}_{interval}_klines.json"


def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return (time.time() - path.stat().st_mtime) < _CACHE_TTL_SEC
    except OSError:
        return False


def _read_cache(path: Path) -> Optional[list[dict]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            return data
    except Exception as e:  # noqa: BLE001
        log.warning("[klines_fetcher] 读缓存失败: %s", e)
    return None


def _write_cache(path: Path, rows: list[dict]) -> None:
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
        log.info("[klines_fetcher] 缓存写入 %s：%d 根", path, len(rows))
    except OSError as e:  # noqa: BLE001
        log.warning("[klines_fetcher] 写缓存失败（不影响数据）: %s", e)


# ============================================================
# raw row → KlineBar
# ============================================================
def _parse_kline_row(symbol: str, interval: str, row: list) -> Optional[KlineBar]:
    """Binance /klines 返回的 list 转 KlineBar（Pydantic 校验）。

    原始格式（11 个字段）：
        [
          0  open_time      ms
          1  open
          2  high
          3  low
          4  close
          5  volume
          6  close_time     ms
          7  quote_volume
          8  trades_count
          9  taker_buy_base_volume
          10 taker_buy_quote_volume
        ]
    """
    try:
        if len(row) < 9:
            return None
        open_time = _ts_to_sh(int(row[0]))
        close_time = _ts_to_sh(int(row[6]))
        return KlineBar(
            symbol=symbol,
            interval=interval,
            open_time=open_time,
            close_time=close_time,
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
            quote_volume=float(row[7]),
            trades_count=int(row[8]),
        )
    except (ValueError, TypeError, IndexError) as e:
        log.debug("[klines_fetcher] 解析失败: %s | %s", e, row)
        return None
    except ValidationError as e:
        log.debug("[klines_fetcher] 校验失败: %s | %s", e, row)
        return None


# ============================================================
# 客户端
# ============================================================
class KlinesFetcher:
    """币安 K 线客户端。

    用法::

        k = KlinesFetcher()
        bars = k.fetch("BTCUSDT", "1h", limit=720)  # list[KlineBar]
    """

    def __init__(self, *, timeout: float = _TIMEOUT_SEC) -> None:
        # TUN 模式：直连
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout, transport=httpx.HTTPTransport(proxy=None))

    def close(self) -> None:
        self._client.close()

    # ---------------------------------------------------------- 网络
    def _fetch_remote(self, symbol: str, interval: str, limit: int) -> list[list]:
        if interval not in _ALLOWED_INTERVALS:
            raise FeedError(f"不支持的 interval: {interval}")
        if limit < 1 or limit > 1000:
            raise FeedError(f"limit 越界（1~1000）: {limit}")
        try:
            r = self._client.get(
                _BINANCE_API,
                params={"symbol": symbol.upper(), "interval": interval, "limit": limit},
            )
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as e:
            raise FeedError(f"binance klines HTTP 错误: {e}") from e
        except Exception as e:  # noqa: BLE001
            raise FeedError(f"binance klines 未知错误: {e}") from e

        if not isinstance(data, list):
            raise FeedError(f"binance klines 返回非 list: {type(data).__name__}")
        return data

    # ---------------------------------------------------------- 公开
    def fetch(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 720,
        *,
        force: bool = False,
    ) -> list[KlineBar]:
        """拉 K 线 + 缓存 + Pydantic 校验。

        :param symbol: 例 ``BTCUSDT`` / ``ETHUSDT`` / ``SOLUSDT``
        :param interval: 1m / 5m / 15m / 1h / 4h / 1d
        :param limit: 1~1000
        :param force: 强制重拉网络
        """
        symbol = symbol.upper()
        if interval not in _ALLOWED_INTERVALS:
            raise FeedError(f"不支持的 interval: {interval}")

        path = _cache_path(symbol, interval)
        if not force and _cache_is_fresh(path):
            cached = _read_cache(path)
            if cached:
                bars = [b for b in (_parse_kline_row(symbol, interval, r) for r in cached) if b]
                if bars:
                    log.info("[klines_fetcher] 缓存命中 %s_%s：%d 根", symbol, interval, len(bars))
                    return bars

        raw = self._fetch_remote(symbol, interval, limit)
        if not raw:
            raise FeedError(f"binance klines 返回空（{symbol}/{interval}）")

        # 缓存 raw（list[list]）
        _write_cache(path, raw)

        bars = [b for b in (_parse_kline_row(symbol, interval, r) for r in raw) if b]
        if not bars:
            raise FeedError(f"binance klines 全部解析失败（{symbol}/{interval}）")
        log.info("[klines_fetcher] 拉取 %s/%s 成功：%d 根", symbol, interval, len(bars))
        return bars

    # ---------------------------------------------------------- 批量
    def fetch_many(
        self,
        symbols: Iterable[str],
        interval: str = "1h",
        limit: int = 720,
        *,
        force: bool = False,
    ) -> dict[str, list[KlineBar]]:
        """批量拉多个 symbol。失败时该 symbol 跳过（不抛错）。"""
        out: dict[str, list[KlineBar]] = {}
        for sym in symbols:
            try:
                out[sym] = self.fetch(sym, interval, limit, force=force)
            except FeedError as e:
                log.warning("[klines_fetcher] %s 拉取失败: %s", sym, e)
        return out


# ============================================================
# 便捷函数
# ============================================================
def fetch_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 720,
    *,
    force: bool = False,
) -> list[KlineBar]:
    """快捷函数：拉一次 + 自动关闭。"""
    f = KlinesFetcher()
    try:
        return f.fetch(symbol, interval, limit, force=force)
    finally:
        f.close()


__all__ = [
    "KlinesFetcher",
    "fetch_klines",
    "ALLOWED_INTERVALS" if False else "_ALLOWED_INTERVALS",
]

# 显式导出允许的 interval（避免上面那行误导）
__all__ = [
    "KlinesFetcher",
    "fetch_klines",
]

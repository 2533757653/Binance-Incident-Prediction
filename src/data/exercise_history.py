"""历史结算数据接入（Builder-D 产出 · iter-2）。

数据源：币安 eapi 公共端点（无需鉴权）：
    GET https://eapi.binance.com/eapi/v1/exerciseHistory

每条记录结构（实测 2026-06-24）：
    {
      "symbol":         "BTC-260624-66500-C",
      "strikePrice":    "66500",
      "realStrikePrice": "62674.824",  # 到期时实际价（"settle 价"）
      "expiryDate":     1782288000000, # 毫秒
      "strikeResult":   "EXTRINSIC_VALUE_EXPIRED"  # 或 REALISTIC_VALUE_STRICKEN
    }

行为：
- 直连 httpx（Clash TUN 模式已接管，proxies=None）
- JSON 缓存到 ``data/cache/exercise_history.json``，TTL = 24h
- 失败显式抛 ``FeedError``，**不再走 sample 兜底**（spec §五铁律）

导出：
- ``ExerciseHistoryFetcher`` —— 拉取 + 缓存
- ``fetch_exercise_history()`` —— 快捷函数
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import httpx

from src.common.errors import FeedError

log = logging.getLogger(__name__)

# 端点（iter-2 已实测 TUN 直连 5.7s 通）
_EAPI = "https://eapi.binance.com/eapi/v1/exerciseHistory"

# 缓存
_CACHE_DIR = Path(__file__).resolve().parent / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_FILE = _CACHE_DIR / "exercise_history.json"
_CACHE_TTL_SEC = 24 * 3600

# 时区：上海 UTC+8（dataContract §0.1）
_TZ_SH = timezone(timedelta(hours=8))

# 超时（spec §2.1：timeout_sec=10）
_TIMEOUT_SEC = 10.0


def _now_sh() -> datetime:
    return datetime.now(tz=_TZ_SH)


def _cache_is_fresh() -> bool:
    """判断缓存是否在 TTL 内。"""
    if not _CACHE_FILE.exists():
        return False
    try:
        mtime = _CACHE_FILE.stat().st_mtime
        return (time.time() - mtime) < _CACHE_TTL_SEC
    except OSError:
        return False


def _read_cache() -> Optional[list[dict]]:
    if not _CACHE_FILE.exists():
        return None
    try:
        with _CACHE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            return data
    except Exception as e:  # noqa: BLE001
        log.warning("[exercise_history] 读缓存失败: %s", e)
    return None


def _write_cache(rows: list[dict]) -> None:
    try:
        with _CACHE_FILE.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
        log.info("[exercise_history] 缓存写入 %s：%d 条", _CACHE_FILE, len(rows))
    except OSError as e:  # noqa: BLE001
        log.warning("[exercise_history] 写缓存失败（不影响数据）: %s", e)


# ============================================================
# 客户端
# ============================================================
class ExerciseHistoryFetcher:
    """拉取币安 eapi 历史结算记录。

    用法::

        f = ExerciseHistoryFetcher()
        rows = f.fetch()            # 优先缓存
        fresh = f.fetch(force=True) # 强制拉
    """

    def __init__(self, *, timeout: float = _TIMEOUT_SEC) -> None:
        # TUN 模式：强制直连，不传 proxies
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout, transport=httpx.HTTPTransport(proxy=None))

    def close(self) -> None:
        self._client.close()

    # ---------------------------------------------------------- 网络
    def _fetch_remote(self) -> list[dict]:
        """直连 eapi 拉一次。失败抛 FeedError。"""
        try:
            r = self._client.get(_EAPI)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as e:
            raise FeedError(f"eapi exerciseHistory HTTP 错误: {e}") from e
        except Exception as e:  # noqa: BLE001
            raise FeedError(f"eapi exerciseHistory 未知错误: {e}") from e

        if not isinstance(data, list):
            raise FeedError(f"eapi exerciseHistory 返回非 list: {type(data).__name__}")
        return data

    # ---------------------------------------------------------- 公开
    def fetch(self, *, force: bool = False) -> list[dict]:
        """返回原始 list[dict]。

        :param force: 强制重新拉网络，忽略缓存
        """
        if not force and _cache_is_fresh():
            cached = _read_cache()
            if cached:
                log.info("[exercise_history] 缓存命中：%d 条", len(cached))
                return cached

        rows = self._fetch_remote()
        if not rows:
            raise FeedError("eapi exerciseHistory 返回空列表")

        _write_cache(rows)
        log.info("[exercise_history] 拉取成功：%d 条", len(rows))
        return rows


# ============================================================
# 便捷函数
# ============================================================
def fetch_exercise_history(*, force: bool = False) -> list[dict]:
    """快捷函数：拉一次 + 自动关闭 client。"""
    f = ExerciseHistoryFetcher()
    try:
        return f.fetch(force=force)
    finally:
        f.close()


__all__ = [
    "ExerciseHistoryFetcher",
    "fetch_exercise_history",
]

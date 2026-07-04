"""
Funding Rate + Open Interest 数据接入（fapi.binance.com，国内直连 + Clash 代理）。

数据来源：
- /fapi/v1/fundingRate    → 资金费率（每 8h 结算一次）
- /fapi/v1/openInterest    → 当前持仓量
- /futures/data/globalLongShortAccountRatio → 多空账户比

新增特征：
- funding_rate        （最近一次资金费率，正数=多头付空头）
- oi_change_24h       （24h 持仓量变化率）
- long_short_ratio    （多空账户比，>1=多头主导）
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Lock
from typing import Optional

import httpx

log = logging.getLogger("horizon.live.funding")

_TZ_CN = timezone(timedelta(hours=8))
_CACHE_DIR = Path("data/cache/realtime")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_FAPI_BASE = "https://fapi.binance.com"
_DATA_BASE = "https://fapi.binance.com/futures/data"
_LOCKS: dict[str, Lock] = {}
_GUARD = Lock()


def _lock(path: str) -> Lock:
    with _GUARD:
        if path not in _LOCKS:
            _LOCKS[path] = Lock()
        return _LOCKS[path]


def _cache_path(name: str) -> Path:
    return _CACHE_DIR / f"fapi_{name}.json"


def _fapi_get(url: str, params: Optional[dict] = None, proxy_url: str = "", timeout: float = 15.0) -> dict | list:
    """通用 fapi GET。"""
    if proxy_url:
        try:
            transport = httpx.HTTPTransport(proxy=httpx.Proxy(url=proxy_url))
            client = httpx.Client(timeout=timeout, transport=transport, follow_redirects=True)
        except Exception:
            client = httpx.Client(timeout=timeout, follow_redirects=True)
    else:
        client = httpx.Client(timeout=timeout, follow_redirects=True)
    try:
        r = client.get(url, params=params or {})
        r.raise_for_status()
        return r.json()
    finally:
        client.close()


def _cache_get(name: str, ttl_seconds: int) -> Optional[dict | list]:
    """读短 TTL 缓存。"""
    p = _cache_path(name)
    if not p.exists():
        return None
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
        age = time.time() - meta.get("ts", 0)
        if age < ttl_seconds:
            return meta.get("data")
    except Exception:
        pass
    return None


def _cache_put(name: str, data):
    """写短 TTL 缓存。"""
    p = _cache_path(name)
    try:
        p.write_text(json.dumps({"ts": time.time(), "data": data}, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        log.debug("[fapi] 缓存写失败 %s: %s", name, e)


def fetch_funding_rate(symbol: str = "BTCUSDT", *, proxy_url: str = "",
                      cache_ttl: int = 60) -> Optional[float]:
    """
    获取最近一次的资金费率（funding rate，8h 结算一次）。

    返回值：float（小数，例如 0.0001 = +0.01%）
    """
    cache_name = f"funding_{symbol}"
    cached = _cache_get(cache_name, cache_ttl)
    if cached is not None:
        return float(cached) if cached is not None else None

    with _lock(cache_name):
        cached = _cache_get(cache_name, cache_ttl)
        if cached is not None:
            return float(cached) if cached is not None else None

        try:
            data = _fapi_get(f"{_FAPI_BASE}/fapi/v1/fundingRate",
                             params={"symbol": symbol, "limit": 1},
                             proxy_url=proxy_url)
            if isinstance(data, list) and data:
                rate = float(data[-1].get("fundingRate", 0))
                _cache_put(cache_name, rate)
                return rate
        except Exception as e:
            log.warning("[fapi] funding rate 失败 %s: %s", symbol, e)
    return None


def fetch_oi_change_24h(symbol: str = "BTCUSDT", *, proxy_url: str = "",
                         cache_ttl: int = 120) -> Optional[float]:
    """
    24h 持仓量变化率。

    步骤：
    1. 当前 OI（实时）
    2. 24h 前的 OI（历史 endpoint）
    3. 算 (current - old) / old
    """
    cache_name = f"oi_change_{symbol}"
    cached = _cache_get(cache_name, cache_ttl)
    if cached is not None:
        return float(cached) if cached is not None else None

    with _lock(cache_name):
        cached = _cache_get(cache_name, cache_ttl)
        if cached is not None:
            return float(cached) if cached is not None else None

        try:
            # 当前 OI
            cur_data = _fapi_get(f"{_FAPI_BASE}/fapi/v1/openInterest",
                                 params={"symbol": symbol}, proxy_url=proxy_url)
            cur_oi = float(cur_data.get("openInterest", 0))
            if cur_oi <= 0:
                return None
            # 24h 前 OI
            hist_data = _fapi_get(f"{_FAPI_BASE}/futures/data/openInterestHist",
                                  params={"symbol": symbol, "period": "5m", "limit": 300},
                                  proxy_url=proxy_url)
            # 取 24h = 288 根 5m 之前的数据（如果够）
            if isinstance(hist_data, list) and len(hist_data) >= 288:
                old_oi = float(hist_data[-288].get("sumOpenInterest", 0))
            elif isinstance(hist_data, list) and hist_data:
                old_oi = float(hist_data[0].get("sumOpenInterest", 0))
            else:
                old_oi = cur_oi
            if old_oi <= 0:
                old_oi = cur_oi
            change_pct = (cur_oi - old_oi) / old_oi
            _cache_put(cache_name, change_pct)
            return change_pct
        except Exception as e:
            log.warning("[fapi] OI change 失败 %s: %s", symbol, e)
    return None


def fetch_long_short_ratio(symbol: str = "BTCUSDT", *, proxy_url: str = "",
                            cache_ttl: int = 300) -> Optional[float]:
    """
    多空账户比（>1 = 多头主导，<1 = 空头主导）。
    """
    cache_name = f"ls_ratio_{symbol}"
    cached = _cache_get(cache_name, cache_ttl)
    if cached is not None:
        return float(cached) if cached is not None else None

    with _lock(cache_name):
        cached = _cache_get(cache_name, cache_ttl)
        if cached is not None:
            return float(cached) if cached is not None else None

        try:
            data = _fapi_get(f"{_DATA_BASE}/globalLongShortAccountRatio",
                             params={"symbol": symbol, "period": "5m", "limit": 1},
                             proxy_url=proxy_url)
            if isinstance(data, list) and data:
                ratio = float(data[-1].get("longShortRatio", 1.0))
                _cache_put(cache_name, ratio)
                return ratio
        except Exception as e:
            log.warning("[fapi] LS ratio 失败 %s: %s", symbol, e)
    return None


def fetch_funding_oi_features(symbol: str, *, proxy_url: str = "") -> dict:
    """
    一次性拉所有特征：funding rate + OI change + long/short ratio。

    返回 dict 含 funding_rate, oi_change_24h, long_short_ratio（可能 None）。
    """
    fr = fetch_funding_rate(symbol, proxy_url=proxy_url, cache_ttl=60)
    oi = fetch_oi_change_24h(symbol, proxy_url=proxy_url, cache_ttl=120)
    ls = fetch_long_short_ratio(symbol, proxy_url=proxy_url, cache_ttl=300)
    return {
        "funding_rate": fr,
        "oi_change_24h": oi,
        "long_short_ratio": ls,
    }
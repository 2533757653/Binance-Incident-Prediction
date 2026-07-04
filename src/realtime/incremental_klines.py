"""
增量 K 线更新（实时场景）。

痛点：原 fetch_klines 每轮都拉 200 根 K 线（≈ 4 天），但 99.9% 是重复数据。
     Clash 端口被 5 个并发请求挤爆。

方案：
- 首次启动：全量拉 200 根，写入 data/cache/realtime/{symbol}_{interval}.json
- 后续：读缓存最后 open_time → 用 startTime 参数只拉新的 → 拼接 → 写回
- 返回最近 N 根（默认 200）保持内存可控

收益：
- Clash 请求量从 200 → 1~3 根（节省 99%）
- 大幅降低 SSL 握手压力
- 启动后稳态请求次数 = K 线更新频率（30m 一次 / 1h 一次）
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

import httpx

from src.backtest.real_data import _interval_to_ms, _parse_klines

log = logging.getLogger("horizon.live.cache")

_TZ_CN = timezone(timedelta(hours=8))
_CACHE_DIR = Path("data/cache/realtime")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 写缓存的全局锁（多线程/多进程并发写保护）
_FILE_LOCKS: dict[str, threading.Lock] = {}
_FILE_LOCKS_GUARD = threading.Lock()


def _get_lock(path: str) -> threading.Lock:
    with _FILE_LOCKS_GUARD:
        if path not in _FILE_LOCKS:
            _FILE_LOCKS[path] = threading.Lock()
        return _FILE_LOCKS[path]


def _cache_path(symbol: str, interval: str) -> Path:
    return _CACHE_DIR / f"{symbol}_{interval}.json"


def _load_cache(symbol: str, interval: str) -> List[list]:
    """读缓存文件，返回 raw list（open_time 升序）。"""
    path = _cache_path(symbol, interval)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        return data
    except Exception as e:
        log.warning("[cache] 读缓存失败 %s: %s", path, e)
        return []


def _save_cache(symbol: str, interval: str, rows: List[list]) -> None:
    """写缓存（原子：先写临时文件再 rename）。"""
    path = _cache_path(symbol, interval)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as e:
        log.warning("[cache] 写缓存失败 %s: %s", path, e)


def _last_open_time(rows: List[list]) -> int:
    """返回缓存最后一根 K 线的 open_time（ms）。"""
    if not rows:
        return 0
    return int(rows[-1][0])


def fetch_klines_incremental(
    symbol: str,
    interval: str,
    *,
    limit: int = 200,
    proxy_url: str = "",
    timeout: float = 20.0,
    internal_retries: int = 5,
) -> List[list]:
    """
    增量拉 K 线（带本地缓存 + iter-13 SSL 容错）。

    iter-13 增强：
    - 内部 retry 5 次（指数退避）
    - SSL context 显式指定兼容 ciphers（避免 UNEXPECTED_EOF）
    - 连接复用（每次重试不重建 client）
    """
    path = _cache_path(symbol, interval)
    lock = _get_lock(str(path))
    with lock:
        cached = _load_cache(symbol, interval)
        last_ts = _last_open_time(cached)

        if last_ts == 0:
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            need_incremental = False
        else:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": last_ts + 1,
                "limit": 20,
            }
            need_incremental = True

        # iter-13: SSL context 用更宽松的 ciphers + 减少 SSL EOF
        ssl_context = None
        try:
            import ssl
            ssl_context = ssl.create_default_context()
            # 显式指定较新但兼容的 ciphers（避免 UNEXPECTED_EOF）
            ssl_context.set_ciphers(
                "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20"
            )
        except Exception:
            pass

        # 构造 client
        if proxy_url:
            try:
                transport = httpx.HTTPTransport(proxy=httpx.Proxy(url=proxy_url))
                client = httpx.Client(timeout=timeout, transport=transport, verify=ssl_context or True, follow_redirects=True)
            except Exception as e:
                log.warning("[cache] 构造代理 client 失败: %s，回退直连", e)
                client = httpx.Client(timeout=timeout, verify=ssl_context or True, follow_redirects=True)
        else:
            client = httpx.Client(timeout=timeout, verify=ssl_context or True, follow_redirects=True)

        try:
            new_rows = None
            last_err = None
            for attempt in range(internal_retries):
                try:
                    r = client.get("https://api.binance.com/api/v3/klines", params=params)
                    r.raise_for_status()
                    new_rows = r.json()
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if attempt < internal_retries - 1:
                        wait = min(8, 2 ** attempt)
                        log.warning("[cache] %s %s 内部重试 %d/%d: %s，%ds 后重试",
                                    symbol, interval, attempt + 1, internal_retries, e, wait)
                        time.sleep(wait)
                    else:
                        log.warning("[cache] %s %s 内部重试 %d 次全部失败: %s",
                                    symbol, interval, internal_retries, e)
            if new_rows is None:
                raise last_err or RuntimeError("internal retries failed")
        finally:
            client.close()

        if not new_rows:
            return cached[-limit:] if cached else []

        # 合并去重
        seen = {int(r[0]): r for r in cached}
        for r in new_rows:
            seen[int(r[0])] = r
        merged = sorted(seen.values(), key=lambda r: int(r[0]))
        merged = merged[-limit:]

        _save_cache(symbol, interval, merged)

        mode = "incremental" if need_incremental else "full"
        added = len(new_rows)
        total = len(merged)
        log.info("[cache] %s %s %s: 缓存 %d → 拉新 %d → 合并 %d",
                  symbol, interval, mode, len(cached), added, total)

        return merged


def fetch_eapi_incremental(
    endpoint: str,
    *,
    proxy_url: str = "",
    timeout: float = 20.0,
    cache_ttl_seconds: int = 60,
) -> List[dict]:
    """
    eapi 端点的轻量缓存（短 TTL，因为期权价格变化快）。
    ticker/mark 这种数据每 60 秒缓存一次足够。
    """
    path = _CACHE_DIR / f"eapi_{endpoint.replace('/', '_')}.json"
    lock = _get_lock(str(path))
    with lock:
        # 读缓存（检查 TTL）
        if path.exists():
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(meta, dict) and "ts" in meta and "data" in meta:
                    age = (datetime.now(tz=_TZ_CN).timestamp() - meta["ts"])
                    if age < cache_ttl_seconds:
                        log.debug("[cache] eapi %s 命中缓存（age=%ds）", endpoint, int(age))
                        return meta["data"]
            except Exception:
                pass

        # 缓存过期或缺失 → 拉新
        url = f"https://eapi.binance.com/eapi/v1{endpoint}"
        if proxy_url:
            try:
                transport = httpx.HTTPTransport(proxy=httpx.Proxy(url=proxy_url))
                client = httpx.Client(timeout=timeout, transport=transport, follow_redirects=True)
            except Exception:
                client = httpx.Client(timeout=timeout, follow_redirects=True)
        else:
            client = httpx.Client(timeout=timeout, follow_redirects=True)

        try:
            r = client.get(url)
            r.raise_for_status()
            data = r.json()
        finally:
            client.close()

        # 写缓存
        try:
            path.write_text(
                json.dumps({"ts": datetime.now(tz=_TZ_CN).timestamp(), "data": data}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            log.debug("[cache] eapi 缓存写失败: %s", e)

        return data
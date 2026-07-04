"""Deribit DVOL（隐含波动率指数，"加密版 VIX"）数据源。

- 免费、无 key、国内可直连；日频；回溯 ~2.7 年（分页可更久，DVOL 自 2021-03）。
- BTC / ETH 各一条。属**价格无关**信息：期权市场对未来波动的定价。
- 与 realized vol 之差 = **方差风险溢价(variance risk premium)**，vol 研究里被认为有预测力。
- 因果对齐：某 bar 只取 ts<=该 bar 开盘时刻的最近一条。
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import httpx
import numpy as np

log = logging.getLogger("data.deribit_vol")
_UA = {"User-Agent": "Mozilla/5.0 (Horizon DVOL fetcher)"}
_CACHE_DIR = Path("data/cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"


def fetch_dvol(currency: str = "BTC", days: int = 1800, ttl_sec: int = 6 * 3600) -> List[Tuple[int, float]]:
    """返回 [(ts_ms, dvol_close), ...] 升序，分页回溯。带本地缓存。"""
    cache = _CACHE_DIR / f"dvol_{currency}.json"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < ttl_sec:
        try:
            return [(int(a), float(b)) for a, b in json.loads(cache.read_text(encoding="utf-8"))]
        except Exception:
            pass
    now = int(time.time() * 1000)
    start = now - days * 86_400_000
    end = now
    out: List[Tuple[int, float]] = []
    seen = set()
    with httpx.Client(timeout=25, headers=_UA) as cli:
        for _ in range(6):
            r = cli.get(_URL, params={"currency": currency, "start_timestamp": start,
                                      "end_timestamp": end, "resolution": "1D"})
            r.raise_for_status()
            data = (r.json().get("result") or {}).get("data") or []
            if not data:
                break
            for row in data:
                ts = int(row[0])
                if ts not in seen:
                    seen.add(ts)
                    out.append((ts, float(row[4])))  # close
            oldest = int(data[0][0])
            if oldest <= start:
                break
            end = oldest - 1
            time.sleep(0.15)
    out.sort()
    if out:
        cache.write_text(json.dumps(out), encoding="utf-8")
        log.info("[dvol] %s %d 条", currency, len(out))
    return out


def align_dvol(bar_times: Sequence, series: Optional[List[Tuple[int, float]]] = None,
               currency: str = "BTC") -> np.ndarray:
    """把日频 DVOL 因果对齐到每个 bar：取 ts<=bar 开盘时刻的最近一条。无历史处返回 NaN。"""
    if series is None:
        series = fetch_dvol(currency)
    if not series:
        return np.full(len(bar_times), np.nan)
    ts = np.array([s[0] for s in series], dtype=np.int64)
    val = np.array([s[1] for s in series], dtype=float)
    bt = np.array([int(t.timestamp() * 1000) for t in bar_times], dtype=np.int64)
    idx = np.searchsorted(ts, bt, side="right") - 1
    out = np.where(idx >= 0, val[np.clip(idx, 0, len(val) - 1)], np.nan)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from datetime import datetime, timezone, timedelta
    tz = timezone(timedelta(hours=8))
    def d(ms): return datetime.fromtimestamp(ms / 1000, tz=tz).strftime("%Y-%m-%d")
    for ccy in ("BTC", "ETH"):
        s = fetch_dvol(ccy)
        if s:
            print(f"{ccy} DVOL {len(s)} 条  {d(s[0][0])} ~ {d(s[-1][0])}  最新={s[-1][1]:.1f}")

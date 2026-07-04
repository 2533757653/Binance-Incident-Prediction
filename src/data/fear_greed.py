"""恐慌贪婪指数 (Crypto Fear & Greed Index) 数据源 —— alternative.me。

- 免费、无需 key、国内可直连；日频；回溯 ~8.4 年（2018-02 至今）。
- 属**价格无关**信息源：综合波动率/动量/社媒/调查/BTC 占比/搜索趋势。
- 全市场单一指数（BTC 主导），BTC/ETH 共用同一条序列。
- 所有取值按【因果】对齐：某 bar 只取 ts<=该 bar 开盘时刻的最近一条。
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import httpx
import numpy as np

log = logging.getLogger("data.fear_greed")
_UA = {"User-Agent": "Mozilla/5.0 (Horizon FnG fetcher)"}
_CACHE = Path("data/cache/fng.json")
_CACHE.parent.mkdir(parents=True, exist_ok=True)


def fetch_fng(ttl_sec: int = 6 * 3600) -> List[Tuple[int, int]]:
    """返回 [(ts_sec, value_0_100), ...] 升序全量历史（约 3000+ 天）。带本地缓存。"""
    if _CACHE.exists() and (time.time() - _CACHE.stat().st_mtime) < ttl_sec:
        try:
            return [(int(a), int(b)) for a, b in json.loads(_CACHE.read_text(encoding="utf-8"))]
        except Exception:
            pass
    r = httpx.get("https://api.alternative.me/fng/",
                  params={"limit": 0, "format": "json"}, headers=_UA, timeout=30)
    r.raise_for_status()
    data = r.json().get("data") or []
    out = sorted((int(x["timestamp"]), int(x["value"])) for x in data)
    if out:
        _CACHE.write_text(json.dumps(out), encoding="utf-8")
        log.info("[fng] 拉取 %d 条 (%d~%d)", len(out), out[0][0], out[-1][0])
    return out


def align_fng(bar_times: Sequence, series: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
    """把日频 F&G 因果对齐到每个 bar：取 ts<=bar 开盘时刻的最近一条。

    bar_times: List[datetime]（tz-aware）。返回长度相同的 float 数组（0~100）。
    """
    if series is None:
        series = fetch_fng()
    if not series:
        return np.full(len(bar_times), 50.0)
    ts = np.array([s[0] for s in series], dtype=np.int64)
    val = np.array([s[1] for s in series], dtype=float)
    bt = np.array([int(t.timestamp()) for t in bar_times], dtype=np.int64)
    idx = np.searchsorted(ts, bt, side="right") - 1
    idx = np.clip(idx, 0, len(val) - 1)
    return val[idx]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from datetime import datetime, timezone, timedelta
    tz = timezone(timedelta(hours=8))
    def d(t): return datetime.fromtimestamp(t, tz=tz).strftime("%Y-%m-%d")
    s = fetch_fng()
    print(f"F&G {len(s)} 条  {d(s[0][0])} ~ {d(s[-1][0])}  最新值={s[-1][1]}")

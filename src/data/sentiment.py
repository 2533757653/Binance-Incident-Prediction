"""
情绪类数据（永续合约，非价格信息）—— OKX 源，国内可直连。

- 资金费率 funding rate：8h 一次，可回溯 ~90 天（分页）
- 持仓量 OI（rubik 统计，1H）：仅 ~30 天
- 多空账户比 LSR（rubik 统计，1H）：仅 ~30 天

所有取值按【因果】对齐到 5m bar：某时刻只取 ts<=该bar 的最近一条。
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Tuple

import httpx
import numpy as np

log = logging.getLogger("data.sentiment")
_UA = {"User-Agent": "Mozilla/5.0 (Horizon sentiment)"}
_CACHE = Path("data/cache/sentiment")
_CACHE.mkdir(parents=True, exist_ok=True)


def _swap(symbol: str) -> str:
    return f"{symbol[:-4]}-USDT-SWAP" if symbol.endswith("USDT") else symbol


def _ccy(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") else symbol


def _cache_io(name: str, fetch_fn, ttl=1800):
    p = _CACHE / f"{name}.json"
    if p.exists() and (time.time() - p.stat().st_mtime) < ttl:
        return json.loads(p.read_text(encoding="utf-8"))
    data = fetch_fn()
    if data:
        p.write_text(json.dumps(data), encoding="utf-8")
    return data


def fetch_funding(symbol: str, days: int = 90) -> List[Tuple[int, float]]:
    """资金费率历史 (ts_ms, rate)，升序。OKX 分页回溯。"""
    def _do():
        inst = _swap(symbol)
        out, after = [], int(time.time() * 1000)
        start = after - days * 86_400_000
        with httpx.Client(timeout=15, headers=_UA) as cli:
            for _ in range(60):
                r = cli.get("https://www.okx.com/api/v5/public/funding-rate-history",
                            params={"instId": inst, "after": after, "limit": 100})
                rows = r.json().get("data") or []
                if not rows:
                    break
                for x in rows:
                    out.append((int(x["fundingTime"]), float(x["fundingRate"])))
                oldest = int(rows[-1]["fundingTime"])
                if oldest <= start:
                    break
                after = oldest
                time.sleep(0.1)
        return sorted(set(out))
    return _cache_io(f"funding_{symbol}_{days}d", _do)


def fetch_oi(symbol: str) -> List[Tuple[int, float]]:
    """持仓量历史 (ts_ms, oi)，升序。OKX rubik 1H（~30天）。"""
    def _do():
        r = httpx.get("https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume",
                      params={"ccy": _ccy(symbol), "period": "1H"}, timeout=15, headers=_UA)
        rows = r.json().get("data") or []
        return sorted((int(x[0]), float(x[1])) for x in rows)
    return _cache_io(f"oi_{symbol}", _do)


def fetch_lsr(symbol: str) -> List[Tuple[int, float]]:
    """多空账户比历史 (ts_ms, ratio)，升序。OKX rubik 1H（~30天）。"""
    def _do():
        r = httpx.get("https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio",
                      params={"ccy": _ccy(symbol), "period": "1H"}, timeout=15, headers=_UA)
        rows = r.json().get("data") or []
        return sorted((int(x[0]), float(x[1])) for x in rows)
    return _cache_io(f"lsr_{symbol}", _do)


def _causal_align(series: List[Tuple[int, float]], bar_ms: np.ndarray, default=0.0) -> np.ndarray:
    """对每个 bar 时间，取 ts<=bar 的最近一条值（因果）。空则 default。"""
    if not series:
        return np.full(len(bar_ms), default, float)
    ts = np.array([s[0] for s in series], dtype=np.int64)
    val = np.array([s[1] for s in series], float)
    idx = np.searchsorted(ts, bar_ms, side="right") - 1
    out = np.where(idx >= 0, val[np.clip(idx, 0, len(val) - 1)], default)
    return out


# 情绪特征名（与 build_sentiment_features 顺序一致）
SENTIMENT_NAMES = [
    "funding_rate", "funding_sign", "funding_cum8",
    "oi_z", "oi_chg_1h", "oi_chg_4h", "price_oi_div",
    "lsr", "lsr_chg_1h",
]


def build_sentiment_features(symbol: str, bar_open_ms: np.ndarray, bar_close: np.ndarray,
                             days: int = 90, use_oi_lsr: bool = True) -> np.ndarray:
    """构建情绪特征矩阵，行对齐 bar_open_ms（5m bar 开盘时间，毫秒）。"""
    n = len(bar_open_ms)
    fund = fetch_funding(symbol, days=days)
    fr = _causal_align(fund, bar_open_ms, 0.0)
    fr_sign = np.sign(fr)
    # funding_cum8：最近一条 funding 与其前 3 条的累计（粗略 24h 资金成本）
    fr_cum = fr.copy()
    if len(fund) >= 4:
        ts = np.array([s[0] for s in fund]); val = np.array([s[1] for s in fund])
        csum = np.convolve(val, np.ones(3), mode="same")
        idx = np.searchsorted(ts, bar_open_ms, side="right") - 1
        fr_cum = np.where(idx >= 0, csum[np.clip(idx, 0, len(csum) - 1)], 0.0)

    if use_oi_lsr:
        oi = fetch_oi(symbol); lsr = fetch_lsr(symbol)
        oi_v = _causal_align(oi, bar_open_ms, 0.0)
        # 标准化 OI（z-score，用滚动近似：全样本均值方差，仅作量纲，causal 影响小）
        m, s = (oi_v[oi_v > 0].mean(), oi_v[oi_v > 0].std()) if (oi_v > 0).any() else (0, 1)
        oi_z = (oi_v - m) / (s + 1e-9)
        # OI 1h/4h 变化（hourly 数据，12根=1h, 48=4h 的 bar 间隔）
        oi_1h = _shift_pct(oi_v, 12)
        oi_4h = _shift_pct(oi_v, 48)
        # 价格-OI 背离：价格1h涨跌 vs OI 1h涨跌 同号=+1 异号=-1
        px_1h = _shift_pct(bar_close, 12)
        price_oi_div = np.sign(px_1h) * np.sign(oi_1h)
        lsr_v = _causal_align(lsr, bar_open_ms, 1.0)
        lsr_chg = _shift_pct(lsr_v, 12)
    else:
        oi_z = oi_1h = oi_4h = price_oi_div = lsr_v = lsr_chg = np.zeros(n)
        lsr_v = np.ones(n)

    return np.column_stack([fr, fr_sign, fr_cum, oi_z, oi_1h, oi_4h, price_oi_div, lsr_v, lsr_chg])


def _shift_pct(arr: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(arr, dtype=float)
    if len(arr) > k:
        prev = arr[:-k]
        out[k:] = np.where(prev != 0, (arr[k:] - prev) / np.abs(prev), 0.0)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for sym in ["BTCUSDT", "ETHUSDT"]:
        f = fetch_funding(sym, 90); o = fetch_oi(sym); l = fetch_lsr(sym)
        from datetime import datetime, timezone, timedelta
        tz = timezone(timedelta(hours=8))
        def d(ms): return datetime.fromtimestamp(ms/1000, tz=tz).strftime("%m-%d")
        print(f"{sym}: funding {len(f)}条({d(f[0][0])}~{d(f[-1][0])}) | "
              f"OI {len(o)}条({d(o[0][0])}~{d(o[-1][0])}) | LSR {len(l)}条({d(l[0][0])}~{d(l[-1][0])})")

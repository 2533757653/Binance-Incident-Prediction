"""
多源行情数据层（无需 Binance，国内可直连）。

主源 Bybit（单次1000根，拉历史快）→ 兜底 OKX → 兜底 Gate.io。
输出统一为 Binance 同款 9 字段数组：
  [open_time_ms, open, high, low, close, volume, close_time_ms, quote_volume, trades]
这样可直接复用 real_data._parse_klines 与既有缓存格式。
"""
from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional

import httpx

log = logging.getLogger("data.sources")

_INTERVAL_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}

_UA = {"User-Agent": "Mozilla/5.0 (Horizon-Incident data fetcher)"}


def _now_ms() -> int:
    return int(time.time() * 1000)


# ════════════════════════════════════════════════════════════════
# Bybit  (spot, 单次 1000, start/end 毫秒, 返回新→旧)
# ════════════════════════════════════════════════════════════════
_BYBIT_INTERVAL = {"1m": "1", "5m": "5", "15m": "15", "30m": "30",
                   "1h": "60", "4h": "240", "1d": "D"}


def _fetch_bybit(symbol: str, interval: str, start_ms: int, end_ms: int) -> List[list]:
    iv = _BYBIT_INTERVAL[interval]
    step = _INTERVAL_MS[interval]
    sym = f"{symbol[:-4]}USDT" if symbol.endswith("USDT") else symbol
    out: List[list] = []
    cur_end = end_ms
    with httpx.Client(timeout=20.0, headers=_UA) as cli:
        while cur_end > start_ms:
            r = cli.get("https://api.bybit.com/v5/market/kline", params={
                "category": "spot", "symbol": sym, "interval": iv,
                "start": start_ms, "end": cur_end, "limit": 1000,
            })
            r.raise_for_status()
            j = r.json()
            rows = (j.get("result") or {}).get("list") or []
            if not rows:
                break
            # rows: [start, open, high, low, close, volume, turnover]  新→旧
            for k in rows:
                ot = int(k[0])
                out.append([ot, k[1], k[2], k[3], k[4], k[5],
                            ot + step - 1, k[6], 0])
            oldest = int(rows[-1][0])
            if oldest <= start_ms:
                break
            cur_end = oldest - 1
            time.sleep(0.15)
    return out


# ════════════════════════════════════════════════════════════════
# OKX  (history-candles, 单次 100, after=更早, 返回新→旧)
# ════════════════════════════════════════════════════════════════
_OKX_BAR = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1H", "4h": "4H", "1d": "1D"}


def _fetch_okx(symbol: str, interval: str, start_ms: int, end_ms: int) -> List[list]:
    bar = _OKX_BAR[interval]
    step = _INTERVAL_MS[interval]
    inst = f"{symbol[:-4]}-USDT" if symbol.endswith("USDT") else symbol
    out: List[list] = []
    after = end_ms
    with httpx.Client(timeout=20.0, headers=_UA) as cli:
        while after > start_ms:
            r = cli.get("https://www.okx.com/api/v5/market/history-candles", params={
                "instId": inst, "bar": bar, "after": after, "limit": 100,
            })
            r.raise_for_status()
            rows = r.json().get("data") or []
            if not rows:
                break
            # [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]  新→旧
            for k in rows:
                ot = int(k[0])
                if ot < start_ms:
                    continue
                out.append([ot, k[1], k[2], k[3], k[4], k[5],
                            ot + step - 1, k[7] if len(k) > 7 else 0, 0])
            oldest = int(rows[-1][0])
            if oldest <= start_ms:
                break
            after = oldest
            time.sleep(0.12)
    return out


# ════════════════════════════════════════════════════════════════
# Gate.io  (candlesticks, 单次 1000, to=窗口末, 返回旧→新, 秒级)
# ════════════════════════════════════════════════════════════════
_GATE_INTERVAL = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                  "1h": "1h", "4h": "4h", "1d": "1d"}


def _fetch_gate(symbol: str, interval: str, start_ms: int, end_ms: int) -> List[list]:
    iv = _GATE_INTERVAL[interval]
    step = _INTERVAL_MS[interval]
    pair = f"{symbol[:-4]}_USDT" if symbol.endswith("USDT") else symbol
    out: List[list] = []
    to_s = end_ms // 1000
    start_s = start_ms // 1000
    with httpx.Client(timeout=20.0, headers=_UA) as cli:
        while to_s > start_s:
            r = cli.get("https://api.gateio.ws/api/v4/spot/candlesticks", params={
                "currency_pair": pair, "interval": iv, "to": to_s, "limit": 1000,
            })
            r.raise_for_status()
            rows = r.json()
            if not rows:
                break
            # [t(s), quote_vol, close, high, low, open, base_vol, closed]  旧→新
            for k in rows:
                ot = int(k[0]) * 1000
                out.append([ot, k[5], k[3], k[4], k[2], k[6],
                            ot + step - 1, k[1], 0])
            oldest = int(rows[0][0])
            if oldest <= start_s:
                break
            to_s = oldest - 1
            time.sleep(0.15)
    return out


_PROVIDERS: dict[str, Callable[..., List[list]]] = {
    "bybit": _fetch_bybit,
    "okx": _fetch_okx,
    "gate": _fetch_gate,
}
DEFAULT_ORDER = ["bybit", "okx", "gate"]


def fetch_klines_rows(
    symbol: str,
    interval: str,
    *,
    days: int,
    providers: Optional[List[str]] = None,
) -> List[list]:
    """拉 days 天 K 线，返回 Binance 同款 9 字段数组（升序、去重）。
    依次尝试 providers，第一个成功返回足够数据的即采用。"""
    if interval not in _INTERVAL_MS:
        raise ValueError(f"unsupported interval: {interval}")
    providers = providers or DEFAULT_ORDER
    end_ms = _now_ms()
    start_ms = end_ms - days * 86_400_000
    expected = days * 86_400_000 // _INTERVAL_MS[interval]

    last_err: Optional[Exception] = None
    for name in providers:
        fn = _PROVIDERS.get(name)
        if fn is None:
            continue
        try:
            t0 = time.time()
            rows = fn(symbol, interval, start_ms, end_ms)
            # 去重 + 升序
            seen = set()
            dedup = []
            for r in rows:
                if r[0] not in seen and r[0] >= start_ms:
                    seen.add(r[0])
                    dedup.append(r)
            dedup.sort(key=lambda r: r[0])
            if len(dedup) >= expected * 0.8:
                log.info("[sources] %s %s via %s: %d 根 (%.1fs)",
                         symbol, interval, name, len(dedup), time.time() - t0)
                return dedup
            log.warning("[sources] %s 数据偏少 (%d/%d)，换源", name, len(dedup), expected)
            last_err = RuntimeError(f"{name} returned too few rows")
        except Exception as e:
            log.warning("[sources] %s 失败: %s，换源", name, e)
            last_err = e
    raise RuntimeError(f"所有数据源失败: {last_err}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for sym in ["BTCUSDT", "ETHUSDT"]:
        rows = fetch_klines_rows(sym, "5m", days=2)
        print(f"{sym}: {len(rows)} bars | first={rows[0][:5]} | last={rows[-1][:5]}")

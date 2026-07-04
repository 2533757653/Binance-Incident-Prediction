"""现货价格接入（Builder-A 产出）。

数据源：CryptoCompare REST（国内直连稳定，spec §0.1）。
- 实时价：``GET https://min-api.cryptocompare.com/data/price`` 每 5s 一次
- 历史 K 线：``GET .../data/v2/histohour?limit=720``（30 天 × 24h）首次拉取后缓存到
  ``data/cache/{symbol}_1h_30d.json``，后续直接读盘

出口：``PriceTick``（dataContract §1.1）写入 ``price_queue``，同时提供 ``stream_prices()``
     异步生成器供 main.py / 回测使用。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from queue import Queue
from typing import Iterator, Optional

import httpx

from src.common.bus import price_queue
from src.common.errors import FeedError
from src.common.schemas import PriceTick

log = logging.getLogger(__name__)

# CryptoCompare 在中国一般不需要代理（spec §0 验证过）
# 但保留 proxy 入口，env 里有就带
_CRYPTOCOMPARE = "https://min-api.cryptocompare.com/data"

# 缓存目录：src/data/cache/
_CACHE_DIR = Path(__file__).resolve().parent / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 启动时跑一次的"latest trade"延迟戳（用于把 REST 价伪装成 tick）
# 真实生产用 WebSocket 拿逐笔成交；本轮按 spec §0 走 REST 轮询。
_PRICE_POLL_SEC = 5

# 时区：上海 UTC+8（dataContract §0.1）
_TZ_SH = timezone(timedelta(hours=8))


def _now_sh() -> datetime:
    return datetime.now(tz=_TZ_SH)


class CryptoCompareClient:
    """CryptoCompare 现货价格客户端。"""

    def __init__(
        self,
        symbols: tuple[str, ...] = ("BTC", "ETH"),
        vs: str = "USD",
        proxies: Optional[dict[str, str]] = None,
        timeout: float = 8.0,
    ) -> None:
        self.symbols = symbols
        self.vs = vs
        # 保留入参仅为接口兼容 —— CryptoCompare 国内直连，**强制不走代理**
        self.proxies = {}
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout, transport=httpx.HTTPTransport(proxy=None))
        self._trade_id_seq = 0  # REST 拿不到 trade_id，本地自增（不影响业务）

    def close(self) -> None:
        self._client.close()

    # ---------------------------------------------------------- 实时价
    def fetch_spot_prices(self) -> dict[str, float]:
        """``GET /data/price?fsym=BTC&tsym=USD`` —— 多 symbol 时串行调用。"""
        out: dict[str, float] = {}
        for sym in self.symbols:
            try:
                r = self._client.get(
                    f"{_CRYPTOCOMPARE}/price",
                    params={"fsym": sym, "tsyms": self.vs},
                )
                r.raise_for_status()
                payload = r.json()
                price = float(payload[self.vs])
                out[sym] = price
            except Exception as e:  # noqa: BLE001
                log.warning("[price_feed] 拉取 %s/%s 失败: %s", sym, self.vs, e)
        return out

    # ---------------------------------------------------------- 历史 K 线
    def fetch_historical_1h(self, symbol: str, hours: int = 720) -> list[dict]:
        """``GET /data/v2/histohour?fsym=BTC&tsym=USD&limit=720``。"""
        r = self._client.get(
            f"{_CRYPTOCOMPARE}/v2/histohour",
            params={"fsym": symbol, "tsym": self.vs, "limit": hours},
        )
        r.raise_for_status()
        body = r.json()
        if body.get("Response") == "Error":
            raise FeedError(f"CryptoCompare histohour error: {body}")
        return body["Data"]["Data"]

    def _cache_path(self, symbol: str) -> Path:
        return _CACHE_DIR / f"{symbol}_{self.vs}_1h_30d.json"

    def load_or_fetch_history(self, symbol: str, hours: int = 720, *, refresh: bool = False) -> list[dict]:
        """30 天 1h K 线，优先读缓存；缺失或 ``refresh=True`` 时拉取。"""
        path = self._cache_path(symbol)
        if not refresh and path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if data and isinstance(data, list):
                    log.info("[price_feed] 缓存命中 %s：%d 根 K 线", symbol, len(data))
                    return data
            except Exception as e:  # noqa: BLE001
                log.warning("[price_feed] 读缓存 %s 失败: %s，将重拉", path, e)
        log.info("[price_feed] 拉取 %s 历史 1h × %d …", symbol, hours)
        data = self.fetch_historical_1h(symbol, hours)
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            log.info("[price_feed] 缓存写入 %s：%d 根", path, len(data))
        except Exception as e:  # noqa: BLE001
            log.warning("[price_feed] 写缓存失败（不影响数据）: %s", e)
        return data

    # ---------------------------------------------------------- PriceTick
    def to_price_tick(self, symbol_short: str, price: float) -> PriceTick:
        """CryptoCompare 没有逐笔成交号；trade_id 本地自增。"""
        self._trade_id_seq += 1
        return PriceTick(
            ts=_now_sh(),
            symbol=f"{symbol_short}USDT",
            price=round(price, 4),
            qty=0.0,  # 现货聚合价无成交量
            trade_id=self._trade_id_seq,
            is_buyer_maker=False,
        )


# ============================================================
# 流式轮询
# ============================================================
def stream_prices(
    client: CryptoCompareClient,
    out_queue: Optional[Queue] = None,
    interval_sec: int = _PRICE_POLL_SEC,
    stop_after: Optional[int] = None,
) -> Iterator[PriceTick]:
    """生成器：每 ``interval_sec`` 拉一次价格，吐 ``PriceTick``，同时写 ``out_queue``。

    :param stop_after: 累计吐 N 条后停止（None=无限）。main.py 设 2 跑完退出。
    """
    out_queue = out_queue or price_queue
    sent = 0
    while True:
        prices = client.fetch_spot_prices()
        for sym, px in prices.items():
            tick = client.to_price_tick(sym, px)
            try:
                out_queue.put_nowait(tick)
            except Exception:  # noqa: BLE001
                # 队列满：消费者背压
                pass
            log.info(
                "[price_feed] %s %.2f @ %s",
                tick.symbol, tick.price, tick.ts.strftime("%H:%M:%S"),
            )
            sent += 1
            yield tick
            if stop_after is not None and sent >= stop_after:
                return
        if stop_after is not None and sent >= stop_after:
            return
        time.sleep(interval_sec)


def start_background(
    client: CryptoCompareClient,
    out_queue: Optional[Queue] = None,
    interval_sec: int = _PRICE_POLL_SEC,
) -> threading.Thread:
    """后台线程版：不停轮询写到 ``out_queue``。供 main.py 用。"""
    out_queue = out_queue or price_queue

    def _loop() -> None:
        for _ in stream_prices(client, out_queue=out_queue, interval_sec=interval_sec):
            pass

    t = threading.Thread(target=_loop, name="price-feed", daemon=True)
    t.start()
    return t


__all__ = [
    "CryptoCompareClient",
    "stream_prices",
    "start_background",
]

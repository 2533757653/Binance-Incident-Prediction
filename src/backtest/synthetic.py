"""
合成事件流（因为事件合约 API 国内不通 → 用历史 1h K 线反推事件结果）。

每个 1h K 线 = 一个 1h 事件，strike = K 线起点价 ± 1%~3% 随机偏移。
回测时让 SignalGenerator 在每根 K 线起点时刻生成信号，
然后看 close vs strike 判定胜负。
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Iterator, List

from src.common.schemas import EventContract, KlineBar


_TZ = timezone(timedelta(hours=8))


def synthesize_events_from_klines(
    klines_1h: List[KlineBar],
    *,
    symbol: str = "BTCUSDT",
    direction: str = "ABOVE",
    strike_offset_pct: float = 0.01,
    rng_seed: int = 42,
) -> List[EventContract]:
    """
    从 1h K 线列表合成对应事件列表。
    跳过最后一根（无对应结算结果）。
    """
    rng = random.Random(rng_seed)
    out: List[EventContract] = []
    for i, bar in enumerate(klines_1h[:-1]):
        # strike = open 价 ± 1%~3% 随机偏移
        offset = (rng.random() * 2 - 1) * strike_offset_pct
        strike = round(bar.open * (1.0 + offset), 2)
        # 当前中间价 = 隐含概率 = (close - strike 距离的归一化)
        if direction == "ABOVE":
            # 价格越高于 strike，YES 概率越高
            edge = (bar.close - strike) / max(strike, 1.0)
            yes_prob = max(0.05, min(0.95, 0.5 + edge * 5))
        else:  # BELOW
            edge = (strike - bar.close) / max(strike, 1.0)
            yes_prob = max(0.05, min(0.95, 0.5 + edge * 5))
        spread = 0.01 + rng.random() * 0.015
        yes_bid = round(yes_prob - spread / 2, 4)
        yes_ask = round(yes_prob + spread / 2, 4)
        no_bid = round(1 - yes_ask, 4)
        no_ask = round(1 - yes_bid, 4)
        out.append(EventContract(
            event_id=f"{symbol}-1H-{direction}-{int(strike)}-{bar.open_time.strftime('%Y%m%d%H%M%S')}",
            symbol=symbol,
            title=f"{symbol[:3]} 1h 后 {'≥' if direction=='ABOVE' else '<'} {int(strike)}?",
            underlying=symbol[:3],
            strike_price=strike,
            direction=direction,
            time_to_expiry="1h",
            settle_time=bar.close_time,
            current_yes_price=round((yes_bid + yes_ask) / 2, 4),
            current_no_price=round((no_bid + no_ask) / 2, 4),
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            spread=round(spread, 4),
            volume_24h=1000.0,
            open_interest=500.0,
            status="TRADING",
        ))
    return out


def make_synthetic_klines(
    *,
    n: int = 720,
    start_price: float = 65000.0,
    volatility: float = 0.005,
    drift: float = 0.0002,
    interval: str = "1h",
    symbol: str = "BTCUSDT",
    rng_seed: int = 42,
) -> List[KlineBar]:
    """
    用 AR(1) + trend 合成 K 线（用于本地测试/离线回测）。
    让合成数据有"短期反转"模式：mom 因子 + 均值回归因子能赢。
    """
    import numpy as np
    rng = np.random.default_rng(rng_seed)
    # AR(1) 系数 0.7：昨天涨 → 今天继续涨一点 → 反转
    rets = []
    prev = 0.0
    for _ in range(n):
        shock = rng.normal(loc=drift, scale=volatility)
        rets.append(0.7 * prev + shock)
        prev = rets[-1]
    rets = np.array(rets)
    prices = start_price * np.exp(np.cumsum(rets))
    base = datetime(2026, 5, 25, 0, 0, tzinfo=_TZ)
    out: List[KlineBar] = []
    for i, p in enumerate(prices):
        out.append(KlineBar(
            symbol=symbol,
            interval=interval,
            open_time=base + timedelta(hours=i),
            close_time=base + timedelta(hours=i + 1),
            open=float(prices[i - 1]) if i > 0 else p,
            high=p * (1 + abs(rng.normal(0, volatility / 2))),
            low=p * (1 - abs(rng.normal(0, volatility / 2))),
            close=float(p),
            volume=100.0,
            quote_volume=float(p) * 100.0,
            trades_count=int(abs(rng.normal(200, 50))),
        ))
    return out
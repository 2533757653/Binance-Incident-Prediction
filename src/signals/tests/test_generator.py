"""
Builder-B SignalGenerator 端到端单测：
mock EventContract + mock K 线 → 期望 Signal 列表。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.common.schemas import EventContract, KlineBar
from src.signals.generator import SignalGenerator
from src.signals.scoring import ScoringConfig

_TZ = timezone(timedelta(hours=8))


# ============================================================
# helpers
# ============================================================
def _mk_event(
    *,
    event_id: str = "BTCUSDT-1H-ABOVE-70000-20260624150000",
    yes_price: float = 0.62,
    no_price: float = 0.38,
    spread: float = 0.02,
    direction: str = "ABOVE",
    settle_minutes: int = 50,
    status: str = "TRADING",
    underlying: str = "BTC",
    now: datetime = None,
) -> EventContract:
    if now is None:
        now = datetime(2026, 6, 24, 14, 10, tzinfo=_TZ)
    settle = now + timedelta(minutes=settle_minutes)
    half = spread / 2
    return EventContract(
        event_id=event_id,
        symbol="BTCUSDT",
        title=f"BTC 1h 后 {'≥' if direction=='ABOVE' else '<'} 70000?",
        underlying=underlying,
        strike_price=70000.0,
        direction=direction,
        time_to_expiry="1h",
        settle_time=settle,
        current_yes_price=yes_price,
        current_no_price=no_price,
        yes_bid=yes_price - half,
        yes_ask=yes_price + half,
        no_bid=no_price - half,
        no_ask=no_price + half,
        spread=spread,
        volume_24h=1000.0,
        open_interest=500.0,
        status=status,
    )


# 测试用的固定 now（避免真实时间已过期）
MOCK_NOW = datetime(2026, 6, 24, 14, 10, tzinfo=_TZ)


def _mk_klines(
    n: int,
    start: float,
    end: float,
    interval: str = "1h",
    symbol: str = "BTCUSDT",
) -> list[KlineBar]:
    """线性插值生成 K 线。"""
    base = datetime(2026, 6, 23, 0, 0, tzinfo=_TZ)
    out = []
    for i in range(n):
        c = start + (end - start) * (i / max(n - 1, 1))
        o = h = l = c
        out.append(KlineBar(
            symbol=symbol, interval=interval,
            open_time=base + timedelta(hours=i),
            close_time=base + timedelta(hours=i + 1),
            open=o, high=h, low=l, close=c,
            volume=1.0, quote_volume=c, trades_count=10,
        ))
    return out


def _mk_quad_klines(
    n: int,
    k: float = 0.3,
    interval: str = "1h",
    symbol: str = "BTCUSDT",
) -> list[KlineBar]:
    """
    二次曲线 K 线（让末尾尖锐以触发 BB%b > 0.95）：
    close_i = 100 + k * i^2
    """
    base = datetime(2026, 6, 23, 0, 0, tzinfo=_TZ)
    out = []
    for i in range(n):
        c = 100.0 + k * i * i
        o = h = l = c
        out.append(KlineBar(
            symbol=symbol, interval=interval,
            open_time=base + timedelta(hours=i),
            close_time=base + timedelta(hours=i + 1),
            open=o, high=h, low=l, close=c,
            volume=1.0, quote_volume=c, trades_count=10,
        ))
    return out


# ============================================================
# 测试
# ============================================================
def test_generator_yields_signal_on_strong_uptrend():
    """
    强上涨趋势：mom > 0.5%, BB%b > 0.95, ATR 正向 → YES 信号。
    用二次曲线造数据让 BB%b 突破 0.95。
    """
    # 用二次曲线让末尾尖锐
    def quad(n, k):
        return [100.0 + 0.3 * i * i for i in range(n)]

    klines_1m = _mk_quad_klines(30, 0.2, interval="1m")
    klines_1h = _mk_quad_klines(20, 0.3, interval="1h")
    klines_4h = _mk_quad_klines(20, 0.5, interval="4h")
    ev = _mk_event(yes_price=0.55, no_price=0.45, spread=0.01, settle_minutes=55, direction="ABOVE", now=MOCK_NOW)

    gen = SignalGenerator()
    sig = gen.consume_event(ev, klines_1m, klines_1h, klines_4h, now=MOCK_NOW)
    assert sig is not None, f"强上涨应出 YES 信号 (factors: mom={sig and sig.factors})"
    assert sig.side == "YES"
    assert sig.entry_price == 0.55
    # 2/3 一致度 + 55min 时间窗 → 置信度会被时间惩罚压到 ~0.4，仍能通过信号过滤
    assert sig.confidence > 0.4
    assert sig.expected_value > 0
    assert set(["mom_15m", "bb_pct_1h", "atr_break_4h", "spread", "time_to_settle_min"]).issubset(sig.factors.keys())


def test_generator_filters_low_confidence():
    """
    弱信号：mom 略超阈，BB 中位，ATR 中位 → confidence 应较低 → 可能被过滤。
    """
    klines_1m = _mk_klines(30, 100.0, 100.5, interval="1m")  # +0.5% 边界
    klines_1h = _mk_klines(20, 100.0, 101.0, interval="1h")  # 温和上升
    klines_4h = _mk_klines(20, 100.0, 100.0, interval="4h")  # 几乎平
    ev = _mk_event(yes_price=0.5, no_price=0.5, spread=0.04, now=MOCK_NOW)

    gen = SignalGenerator()
    sig = gen.consume_event(ev, klines_1m, klines_1h, klines_4h, now=MOCK_NOW)
    if sig is not None:
        assert sig.confidence < 0.6


def test_generator_returns_none_for_wide_spread():
    """spread > 0.05 → 过滤。"""
    klines_1m = _mk_quad_klines(30, 0.2, interval="1m")
    klines_1h = _mk_quad_klines(20, 0.3, interval="1h")
    klines_4h = _mk_quad_klines(20, 0.5, interval="4h")
    ev = _mk_event(spread=0.10, settle_minutes=55, now=MOCK_NOW)
    gen = SignalGenerator()
    sig = gen.consume_event(ev, klines_1m, klines_1h, klines_4h, now=MOCK_NOW)
    assert sig is None


def test_generator_returns_none_for_expired_status():
    klines_1m = _mk_quad_klines(30, 0.2, interval="1m")
    klines_1h = _mk_quad_klines(20, 0.3, interval="1h")
    klines_4h = _mk_quad_klines(20, 0.5, interval="4h")
    ev = _mk_event(status="EXPIRED", now=MOCK_NOW)
    gen = SignalGenerator()
    sig = gen.consume_event(ev, klines_1m, klines_1h, klines_4h, now=MOCK_NOW)
    assert sig is None


def test_generator_returns_none_for_too_close_to_settle():
    klines_1m = _mk_quad_klines(30, 0.2, interval="1m")
    klines_1h = _mk_quad_klines(20, 0.3, interval="1h")
    klines_4h = _mk_quad_klines(20, 0.5, interval="4h")
    ev = _mk_event(settle_minutes=0, now=MOCK_NOW)  # 已经到期
    gen = SignalGenerator()
    sig = gen.consume_event(ev, klines_1m, klines_1h, klines_4h, now=MOCK_NOW)
    assert sig is None


def test_generator_batch():
    """批处理：3 个事件，1 个应出信号。"""
    klines_1m = _mk_quad_klines(30, 0.2, interval="1m")
    klines_1h = _mk_quad_klines(20, 0.3, interval="1h")
    klines_4h = _mk_quad_klines(20, 0.5, interval="4h")

    e1 = _mk_event(event_id="E1", spread=0.01, settle_minutes=55, now=MOCK_NOW)  # 通过
    e2 = _mk_event(event_id="E2", spread=0.10, settle_minutes=55, now=MOCK_NOW)  # spread 太宽，过滤
    e3 = _mk_event(event_id="E3", status="EXPIRED", now=MOCK_NOW)  # 状态过滤

    gen = SignalGenerator()
    klines_by_symbol = {"BTCUSDT": {"1m": klines_1m, "1h": klines_1h, "4h": klines_4h}}
    out = gen.consume_batch([e1, e2, e3], klines_by_symbol, now=MOCK_NOW)
    assert len(out) == 1
    assert out[0].event_id == "E1"


def test_signal_pydantic_validation():
    """生成的 Signal 一定通过 Pydantic 校验。"""
    klines_1m = _mk_quad_klines(30, 0.2, interval="1m")
    klines_1h = _mk_quad_klines(20, 0.3, interval="1h")
    klines_4h = _mk_quad_klines(20, 0.5, interval="4h")
    ev = _mk_event(yes_price=0.55, no_price=0.45, spread=0.01, settle_minutes=55, now=MOCK_NOW)
    gen = SignalGenerator()
    sig = gen.consume_event(ev, klines_1m, klines_1h, klines_4h, now=MOCK_NOW)
    assert sig is not None
    dumped = sig.model_dump()
    from src.common.schemas import Signal as S2
    s2 = S2(**dumped)
    assert s2.signal_id == sig.signal_id

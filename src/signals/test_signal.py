"""
Builder-B 端到端可独立运行的演示脚本。

运行：
    python -m src.signals.test_signal

输出：
    - 终端打印合成的 BTC/ETH 事件 + 因子值 + 生成的 Signal
    - 退出码 0 = 全部 OK
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

from src.common.schemas import EventContract, KlineBar
from src.signals.generator import SignalGenerator


_TZ = timezone(timedelta(hours=8))


def make_synth_klines(
    n: int,
    start: float,
    end: float,
    interval: str = "1h",
    symbol: str = "BTCUSDT",
) -> list[KlineBar]:
    """线性插值合成 K 线（无未来函数：close 都在 open_time 时刻"完成"）。"""
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


def make_quad_klines(
    n: int,
    k: float = 0.3,
    interval: str = "1h",
    symbol: str = "BTCUSDT",
) -> list[KlineBar]:
    """二次曲线 K 线（让末尾尖锐以触发 BB%b > 0.95）。"""
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


def make_synth_event(
    symbol: str,
    underlying: str,
    direction: str,
    strike: float,
    yes_price: float,
    no_price: float,
    spread: float,
    settle_minutes: int,
    event_id_suffix: str,
) -> EventContract:
    now = datetime(2026, 6, 24, 14, 10, tzinfo=_TZ)
    settle = now + timedelta(minutes=settle_minutes)
    half = spread / 2
    direction_label = "≥" if direction == "ABOVE" else "<"
    title = f"{underlying} 1h 后 {direction_label} {strike}?"
    return EventContract(
        event_id=f"{symbol}-1H-{direction}-{int(strike)}-{event_id_suffix}",
        symbol=symbol,
        title=title,
        underlying=underlying,
        strike_price=strike,
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
        status="TRADING",
    )


def main() -> int:
    print("=" * 80)
    print("Horizon-Incident · Builder-B · SignalGenerator 端到端测试")
    print("=" * 80)

    # 1) 强上涨 BTC 事件（二次曲线造数据让 BB%b > 0.95，ATR 强正偏离）
    btc_klines_1m = make_quad_klines(30, 0.2, interval="1m", symbol="BTCUSDT")
    btc_klines_1h = make_quad_klines(20, 0.3, interval="1h", symbol="BTCUSDT")
    btc_klines_4h = make_quad_klines(20, 0.5, interval="4h", symbol="BTCUSDT")

    # 2) 横盘 ETH 事件（弱趋势，不应过 confidence 阈值）
    eth_klines_1m = make_synth_klines(30, 100.0, 100.3, interval="1m", symbol="ETHUSDT")
    eth_klines_1h = make_synth_klines(20, 100.0, 101.0, interval="1h", symbol="ETHUSDT")
    eth_klines_4h = make_synth_klines(20, 100.0, 100.5, interval="4h", symbol="ETHUSDT")

    events = [
        make_synth_event(
            symbol="BTCUSDT", underlying="BTC", direction="ABOVE",
            strike=70000, yes_price=0.55, no_price=0.45, spread=0.01,
            settle_minutes=55, event_id_suffix="20260624150000",
        ),
        make_synth_event(
            symbol="ETHUSDT", underlying="ETH", direction="BELOW",
            strike=3500, yes_price=0.50, no_price=0.50, spread=0.04,
            settle_minutes=55, event_id_suffix="20260624150000",
        ),
    ]

    klines_by_symbol = {
        "BTCUSDT": {"1m": btc_klines_1m, "1h": btc_klines_1h, "4h": btc_klines_4h},
        "ETHUSDT": {"1m": eth_klines_1m, "1h": eth_klines_1h, "4h": eth_klines_4h},
    }

    gen = SignalGenerator()
    sigs = gen.consume_batch(events, klines_by_symbol)

    print(f"\n[输入]  事件数: {len(events)}")
    for e in events:
        print(f"  - {e.event_id}  settle_in={int((e.settle_time - datetime.now(tz=_TZ)).total_seconds()/60)}min  spread={e.spread}")

    print(f"\n[输出]  信号数: {len(sigs)}")
    for s in sigs:
        print(f"  * {s.signal_id}")
        print(f"    side={s.side}  entry={s.entry_price}  conf={s.confidence:.3f}  ev={s.expected_value:+.3f}")
        print(f"    rationale: {s.rationale}")
        print(f"    factors: {s.factors}")
        print()

    # 校验：至少应该有 1 个 BTC YES 信号
    btc_yes = [s for s in sigs if s.symbol == "BTCUSDT" and s.side == "YES"]
    if not btc_yes:
        print("[FAIL] 期望至少 1 个 BTC YES 信号", file=sys.stderr)
        return 1

    # 校验：所有 Signal 都通过 Pydantic
    for s in sigs:
        s.model_validate(s.model_dump())

    # 校验：factors dict 至少含 5 个键
    required = {"mom_15m", "bb_pct_1h", "atr_break_4h", "spread", "time_to_settle_min"}
    for s in sigs:
        if not required.issubset(s.factors.keys()):
            print(f"[FAIL] signal {s.signal_id} factors missing keys", file=sys.stderr)
            return 1

    print("=" * 80)
    print("[PASS] 端到端信号生成 OK")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Backtester — 用合成事件流跑 SignalGenerator，记录每次胜负。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, List

from src.common.schemas import BacktestResult, EventContract, KlineBar, Signal
from src.signals.generator import SignalGenerator


@dataclass
class TradeRecord:
    signal: Signal
    settle_price: float
    won: bool          # True = 信号方向正确，结算拿到 1.0
    pnl: float         # 单笔盈亏比例（以 entry_price 计）


@dataclass
class BacktestRun:
    symbol: str
    strategy: str
    records: List[TradeRecord] = field(default_factory=list)
    latency_ms_samples: List[float] = field(default_factory=list)

    @property
    def total_signals(self) -> int:
        return len(self.records)

    @property
    def wins(self) -> int:
        return sum(1 for r in self.records if r.won)

    @property
    def losses(self) -> int:
        return sum(1 for r in self.records if not r.won)

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.total_signals, 1)

    def total_pnl_pct(self) -> float:
        return sum(r.pnl for r in self.records)

    def max_drawdown_pct(self) -> float:
        cum, peak, dd = 0.0, 0.0, 0.0
        for r in self.records:
            cum += r.pnl
            peak = max(peak, cum)
            dd = max(dd, peak - cum)
        return dd

    def avg_payoff_ratio(self) -> float:
        wins = [r.pnl for r in self.records if r.won]
        losses = [abs(r.pnl) for r in self.records if not r.won]
        if not wins or not losses:
            return 1.0  # 无胜负记录时按"中性" 1.0
        return (sum(wins) / len(wins)) / (sum(losses) / len(losses))

    def signals_per_day(self, days: float) -> float:
        return self.total_signals / max(days, 0.01)

    def avg_latency_ms(self) -> float:
        if not self.latency_ms_samples:
            return 0.0
        return sum(self.latency_ms_samples) / len(self.latency_ms_samples)


def run_backtest(
    *,
    symbol: str,
    klines_1m: List[KlineBar] | None,
    klines_1h: List[KlineBar],
    klines_4h: List[KlineBar],
    events: List[EventContract],
    generator: SignalGenerator,
    strategy: str = "factor_v1",
) -> BacktestRun:
    """
    给定一组事件 + 对应的 K 线（1m / 1h / 4h），跑回测。

    假设 events[i] 对应 klines_1h[i]（即事件 settle_time = klines_1h[i].close_time）。
    1m K 线用于 mom_15m（15 分钟动量）。当 klines_1m=None 时，用最近 15 根 1h K 线代替。
    """
    run = BacktestRun(symbol=symbol, strategy=strategy)
    for i, ev in enumerate(events):
        if i >= len(klines_1h):
            break
        # 1h K 线：取最近 240 根
        h_start = max(0, i - 240)
        k1h = klines_1h[h_start : i + 1]
        # 4h K 线：取最近 60 根
        q_start = max(0, i // 4 - 60)
        k4h = klines_4h[q_start : q_start + 60]
        # 1m K 线替代方案：用最近 15 根 1h K 线喂给 mom_15m（精度降级但能跑）
        if klines_1m is not None:
            m_end = (i + 1) * 60
            m_start = max(0, m_end - 60 * 6)
            k1m = klines_1m[m_start:m_end]
        else:
            k1m = k1h[-15:] if len(k1h) >= 15 else k1h
        if len(k1m) < 5 or len(k1h) < 20 or len(k4h) < 14:
            continue

        t0 = time.perf_counter()
        # 在事件到期 30 分钟前发信号（给 1h 事件留足判断窗口）
        sig = generator.consume_event(ev, k1m, k1h, k4h, now=ev.settle_time - timedelta(minutes=30))
        latency_ms = (time.perf_counter() - t0) * 1000
        run.latency_ms_samples.append(latency_ms)
        if sig is None:
            continue

        # 判定胜负：事件在 settle 时刻的 close vs strike
        settle_bar = klines_1h[i]
        settle_price = settle_bar.close
        if ev.direction == "ABOVE":
            yes_won = settle_price >= ev.strike_price
        else:
            yes_won = settle_price < ev.strike_price
        signal_correct = (sig.side == "YES" and yes_won) or (sig.side == "NO" and not yes_won)

        entry = sig.entry_price
        if signal_correct:
            pnl = (1.0 - entry) / entry
        else:
            pnl = -1.0
        run.records.append(TradeRecord(signal=sig, settle_price=settle_price, won=signal_correct, pnl=pnl))

    return run


def to_backtest_result(run: BacktestRun, start: datetime, end: datetime, days: float, config: dict) -> BacktestResult:
    """把 BacktestRun 转成 BacktestResult schema。"""
    return BacktestResult(
        strategy=run.strategy,
        symbol=run.symbol,
        start_time=start,
        end_time=end,
        total_signals=run.total_signals,
        wins=run.wins,
        losses=run.losses,
        pushes=0,
        win_rate=round(run.win_rate, 4),
        avg_payoff_ratio=round(run.avg_payoff_ratio(), 4),
        expected_value=round(
            run.win_rate * run.avg_payoff_ratio() - (1 - run.win_rate), 4
        ),
        total_pnl_pct=round(run.total_pnl_pct() * 100, 2),
        max_drawdown_pct=round(run.max_drawdown_pct() * 100, 2),
        signals_per_day=round(run.signals_per_day(days), 2),
        avg_latency_ms=round(run.avg_latency_ms(), 2),
        config_snapshot=config,
    )


from datetime import timedelta  # noqa: E402  (avoid circular)
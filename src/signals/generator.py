"""
Builder-B 主类：SignalGenerator。

功能：
- 从 event_queue 消费 EventContract
- 用最近 240 根 1h K 线（实际是合成的 1h 序列；同时维护 1m/4h 序列）计算 3 个因子
- 多数派投票决定 side，置信度 = 一致度 × 价差惩罚 × 时间惩罚
- 评分过滤（expected_value / confidence / spread / 时间窗）
- 通过过滤则生成 Signal 写入 signal_queue

本轮 iter-1 假定 K 线由调用方注入（因 Builder-A 与 Builder-C 尚未交付真实 WS 流），
SignalGenerator 也支持 `consume_event(event, klines_1m, klines_1h, klines_4h)` 同步入口。
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone, timedelta
from queue import Empty, Queue
from typing import Iterable, List, Optional, Sequence

from src.common.bus import event_queue, signal_queue
from src.common.errors import SignalError
from src.common.schemas import EventContract, Signal, make_signal_id

from .factors import (
    compute_all_factors,
    compute_atr_break_4h,
    compute_bb_pct_1h,
    compute_mom_15m,
    FactorResult,
)
from .scoring import (
    ScoringConfig,
    compute_expected_value,
    compute_payoff_ratio,
    passes_filter,
)

logger = logging.getLogger(__name__)

# 东八区时区（dataContract §0.1）
_TZ_CN = timezone(timedelta(hours=8))


# ============================================================
# 主类
# ============================================================
class SignalGenerator:
    """
    信号生成器。

    用法 1（同步，单次事件）：
        gen = SignalGenerator()
        sig = gen.consume_event(event, klines_1m, klines_1h, klines_4h)

    用法 2（异步，从 event_queue 消费）：
        gen = SignalGenerator()
        gen.run_forever()  # 阻塞，内部从 src.common.bus.event_queue 读
    """

    def __init__(
        self,
        scoring_config: Optional[ScoringConfig] = None,
        strategy_name: str = "factor_v1",
        ttl_seconds: int = 60,
        # 因子参数（与 config/strategy.yaml 对齐）
        mom_lookback: int = 15,
        mom_threshold: float = 0.002,        # 降低阈值（0.5% → 0.2%）
        bb_period: int = 20,
        bb_std: float = 2.0,
        bb_extreme: float = 0.80,            # 极端阈值（0.95 → 0.80，更敏感）
        atr_period: int = 14,
        atr_breakout_mult: float = 1.0,      # 突破阈值（1.5 → 1.0）
    ):
        self.cfg = scoring_config or ScoringConfig()
        self.strategy_name = strategy_name
        self.ttl_seconds = ttl_seconds
        self.mom_lookback = mom_lookback
        self.mom_threshold = mom_threshold
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.bb_extreme = bb_extreme
        self.atr_period = atr_period
        self.atr_breakout_mult = atr_breakout_mult

    # --------------------------------------------------------
    # 内部：3 因子 → 方向 + 置信度
    # --------------------------------------------------------
    def _decide_direction(self, fr: FactorResult) -> tuple[str, float, int]:
        """
        多数派投票决定 side。

        返回 (side, agreement_score, votes_yes)
        agreement_score ∈ [0, 1]：多数派一致度 = 多数票 / 3
        votes_yes：投 YES 的因子数
        """
        votes: list[str] = []

        # 因子 1：MOM_15m
        if fr.mom_15m > self.mom_threshold:
            votes.append("YES")
        elif fr.mom_15m < -self.mom_threshold:
            votes.append("NO")
        else:
            votes.append("HOLD")

        # 因子 2：BB_1h（超买→NO，超卖→YES）
        if fr.bb_pct_1h > self.bb_extreme:
            votes.append("NO")
        elif fr.bb_pct_1h < (1 - self.bb_extreme):
            votes.append("YES")
        else:
            votes.append("HOLD")

        # 因子 3：ATR_4h（同向 = 价格偏离方向）
        if fr.atr_break_4h > 0.5:
            votes.append("YES")
        elif fr.atr_break_4h < -0.5:
            votes.append("NO")
        else:
            votes.append("HOLD")

        yes_count = votes.count("YES")
        no_count = votes.count("NO")
        hold_count = votes.count("HOLD")

        if yes_count > no_count:
            side = "YES"
            agreement = yes_count / 3.0
        elif no_count > yes_count:
            side = "NO"
            agreement = no_count / 3.0
        else:
            # 平局 → 不出信号（HOLD）
            return ("HOLD", 0.0, yes_count)

        # 一致度：基础值是多数比例；若全 HOLD 中夹了 1 票反对 → 折扣
        if hold_count > 0 and yes_count == 2:
            agreement = 2 / 3.0
        if hold_count > 0 and no_count == 2:
            agreement = 2 / 3.0

        return (side, float(agreement), yes_count)

    def _build_confidence(
        self,
        agreement: float,
        spread: float,
        seconds_to_settle: float,
    ) -> float:
        """
        置信度 = 多数派一致度 × 价差惩罚 × 时间惩罚。

        价差惩罚 = sqrt(max(0, 1 - spread / 0.05))   # spread=0 → 1, spread=0.02 → 0.77, spread=0.05 → 0
        时间惩罚：以 2h 为满分基准，4h 截到 1.0
          = clamp(sqrt(seconds_to_settle / 7200), 0, 1)
          → 2h = 1.0, 1h = 0.71, 30min = 0.50, 10min = 0.29
        """
        spread_penalty = math.sqrt(max(0.0, 1.0 - spread / 0.05))
        time_penalty = math.sqrt(max(0.0, seconds_to_settle / 7200.0))
        time_penalty = min(1.0, time_penalty)
        return float(agreement * spread_penalty * time_penalty)

    def _win_prob_from_factors(
        self,
        fr: FactorResult,
        side: str,
    ) -> float:
        """
        把因子强度映射到胜率。
        简易线性：基础 0.5 + 0.08 × 一致度 + 0.04 × |atr_break_4h|（capped at 0.80）
        保守 cap 避免信号"过度自信"。
        """
        base = 0.5
        agreement_bonus = 0.08 * 3 if side != "HOLD" else 0.0
        atr_bonus = min(0.18, 0.04 * abs(fr.atr_break_4h))
        return float(min(0.80, base + agreement_bonus + atr_bonus))

    # --------------------------------------------------------
    # 内部：构造 rationale
    # --------------------------------------------------------
    def _build_rationale(self, fr: FactorResult, side: str) -> str:
        mom_pct = fr.mom_15m * 100
        return (
            f"动量{mom_pct:+.2f}% BB%b={fr.bb_pct_1h:.2f} "
            f"ATR偏离={fr.atr_break_4h:+.2f}σ → {side}"
        )

    # --------------------------------------------------------
    # 公开：同步单事件消费
    # --------------------------------------------------------
    def consume_event(
        self,
        event: EventContract,
        klines_1m: Sequence,
        klines_1h: Sequence,
        klines_4h: Sequence,
        now: Optional[datetime] = None,
    ) -> Optional[Signal]:
        """
        同步入口：给定一个 EventContract + 三组 K 线，产出一个 Signal 或 None（被过滤）。

        返回 None = 过滤掉 / 无方向。
        """
        if event.status != "TRADING":
            logger.debug("skip non-trading event %s", event.event_id)
            return None

        now = now or datetime.now(tz=_TZ_CN)
        seconds_to_settle = (event.settle_time - now).total_seconds()
        if seconds_to_settle < 0:
            logger.debug("skip expired event %s", event.event_id)
            return None

        # 1) 算因子
        fr = compute_all_factors(
            klines_1m=klines_1m,
            klines_1h=klines_1h,
            klines_4h=klines_4h,
            mom_lookback=self.mom_lookback,
            mom_threshold=self.mom_threshold,
            bb_period=self.bb_period,
            bb_std=self.bb_std,
            bb_extreme=self.bb_extreme,
            atr_period=self.atr_period,
            atr_breakout_mult=self.atr_breakout_mult,
        )

        # 2) 多数派方向
        side, agreement, _ = self._decide_direction(fr)
        if side == "HOLD":
            return None

        # 3) entry_price：取 side 对应的中间价
        if side == "YES":
            entry_price = event.current_yes_price
        else:
            entry_price = event.current_no_price

        payoff_ratio = compute_payoff_ratio(entry_price)
        confidence = self._build_confidence(agreement, event.spread, seconds_to_settle)
        win_prob = self._win_prob_from_factors(fr, side)
        ev = compute_expected_value(win_prob, payoff_ratio)

        # 4) 过滤
        factors_dict = {
            "mom_15m": round(fr.mom_15m, 6),
            "bb_pct_1h": round(fr.bb_pct_1h, 6),
            "atr_break_4h": round(fr.atr_break_4h, 6),
            "spread": round(event.spread, 6),
            "time_to_settle_min": round(seconds_to_settle / 60.0, 4),
        }

        decision = passes_filter(
            confidence=confidence,
            expected_value=ev,
            spread=event.spread,
            seconds_to_settle=seconds_to_settle,
            payoff_ratio=payoff_ratio,
            config=self.cfg,
        )
        if not decision.passed:
            logger.debug(
                "filtered %s: %s (conf=%.3f ev=%.3f spread=%.3f sec=%.1f)",
                event.event_id, decision.reason, confidence, ev, event.spread, seconds_to_settle,
            )
            return None

        # 5) 装配 Signal
        rationale = self._build_rationale(fr, side)
        if len(rationale) > 80:
            rationale = rationale[:80]

        signal = Signal(
            signal_id=make_signal_id(event.event_id, now),
            ts=now,
            event_id=event.event_id,
            event_title=event.title,
            symbol=event.symbol,
            side=side,
            entry_price=round(float(entry_price), 4),
            spread=round(float(event.spread), 4),
            confidence=round(float(confidence), 6),
            expected_value=round(float(ev), 6),
            win_prob=round(float(win_prob), 6),
            payoff_ratio=round(float(payoff_ratio), 6),
            rationale=rationale,
            strategy=self.strategy_name,
            factors=factors_dict,
            ttl_seconds=self.ttl_seconds,
            expire_at=now + timedelta(seconds=self.ttl_seconds),
        )
        return signal

    # --------------------------------------------------------
    # 公开：批处理 + 异步消费
    # --------------------------------------------------------
    def consume_batch(
        self,
        events: Iterable[EventContract],
        klines_by_symbol: dict,
        now: Optional[datetime] = None,
    ) -> List[Signal]:
        """
        批量入口：events 列表 + {symbol: {'1m': [...], '1h': [...], '4h': [...]}}
        返回产出的 Signal 列表。
        """
        out: list[Signal] = []
        for ev in events:
            sym = ev.symbol
            ks = klines_by_symbol.get(sym, {})
            sig = self.consume_event(
                event=ev,
                klines_1m=ks.get("1m", []),
                klines_1h=ks.get("1h", []),
                klines_4h=ks.get("4h", []),
                now=now,
            )
            if sig is not None:
                out.append(sig)
        return out

    def run_forever(self, klines_by_symbol: Optional[dict] = None, poll_timeout: float = 1.0) -> None:
        """
        阻塞循环：从 event_queue 读事件，需要调用方提供 klines_by_symbol。
        """
        if klines_by_symbol is None:
            klines_by_symbol = {}
        logger.info("SignalGenerator run_forever started")
        while True:
            try:
                event: EventContract = event_queue.get(timeout=poll_timeout)
            except Empty:
                continue
            except Exception as e:
                raise SignalError(f"event_queue read failed: {e}") from e
            ks = klines_by_symbol.get(event.symbol, {})
            try:
                sig = self.consume_event(
                    event=event,
                    klines_1m=ks.get("1m", []),
                    klines_1h=ks.get("1h", []),
                    klines_4h=ks.get("4h", []),
                )
            except Exception as e:
                logger.exception("consume_event failed for %s: %s", event.event_id, e)
                continue
            if sig is not None:
                signal_queue.put(sig)
                logger.info("published signal %s %s conf=%.3f ev=%.3f",
                            sig.signal_id, sig.side, sig.confidence, sig.expected_value)

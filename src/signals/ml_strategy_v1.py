"""
Builder-E · ML 策略（ml_v1）骨架。

iter-2 阶段：先用 sklearn 的 LogisticRegression 拟合一组合成因子 → 胜率，
           然后用同一个模型在回测时产信号。

约定：
- 与 factor_v1 走同一个接口：consume_event(event, k1m, k1h, k4h, now=None) -> Optional[Signal]
- 通过 strategy_name="ml_v1" 与 factor_v1 区分
- 模型在 __init__ 时用 1000+ 组合成样本训练（自包含的 offline 模式）
- 训练数据由 make_synthetic_training_data() 生成（无未来函数：t 时刻的因子 vs t+1h 的胜负）
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

import numpy as np

from src.common.schemas import EventContract, Signal, make_signal_id
from src.common.errors import SignalError
from src.signals.factors import compute_all_factors, FactorResult
from src.signals.scoring import (
    ScoringConfig,
    compute_expected_value,
    compute_payoff_ratio,
    passes_filter,
)

logger = logging.getLogger(__name__)

_TZ = timezone(timedelta(hours=8))


# ============================================================
# 训练数据生成（合成）
# ============================================================
def make_synthetic_training_data(
    *,
    n_samples: int = 1500,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    生成合成训练数据：3 维因子 (mom_15m, bb_pct_1h, atr_break_4h) → 0/1 胜率标签。

    设计：mom 与 atr_break 同向时胜率更高（动量效应），bb_pct 极端时反向
        → 简单可学习的关系，让 LR 能跑出 > 0.5 准确率。
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, 3))
    # 标签：mom+atr 同号 + |bb-0.5| 适度 → 胜
    mom = X[:, 0]
    bb = X[:, 1]
    atr = X[:, 2]
    logit = (
        1.2 * mom               # 动量正向
        + 0.8 * atr             # 趋势加强
        - 0.6 * np.tanh(3.0 * (bb - 0.5))   # 极端位置反向
        - 0.2                   # 偏置 → 整体胜率 < 0.5，模拟真实市场
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n_samples) < p).astype(int)
    return X, y


# ============================================================
# MLStrategy
# ============================================================
class MLStrategy:
    """
    简单 LR 分类器作为"ml_v1"策略。
    因子与 factor_v1 完全相同，仅在评分环节用 LR 预测胜率替代 hand-crafted 规则。
    """

    def __init__(
        self,
        scoring_config: Optional[ScoringConfig] = None,
        strategy_name: str = "ml_v1",
        ttl_seconds: int = 60,
        model=None,
    ):
        self.cfg = scoring_config or ScoringConfig()
        self.strategy_name = strategy_name
        self.ttl_seconds = ttl_seconds
        self.model = model  # sklearn-like; 需有 .predict_proba(X) → (n, 2)

        if self.model is None:
            self._fit_default()

    def _fit_default(self) -> None:
        from sklearn.linear_model import LogisticRegression
        X, y = make_synthetic_training_data(n_samples=2000, seed=11)
        self.model = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        self.model.fit(X, y)
        logger.info("ml_v1 default model trained on n=%d (acc=%.2f)", len(y), float(self.model.score(X, y)))

    # --------------------------------------------------------
    # 复用 factor_v1 的因子计算；只是评分换成 LR
    # --------------------------------------------------------
    def _features_from_factors(self, fr: FactorResult) -> np.ndarray:
        # 用 mom_15m, bb_pct_1h, atr_break_4h 三个数
        return np.array([[fr.mom_15m, fr.bb_pct_1h, fr.atr_break_4h]], dtype=float)

    def _decide_side(self, fr: FactorResult) -> str:
        mom = fr.mom_15m
        atr = fr.atr_break_4h
        if mom > 0.005 and atr > 0.5:
            return "YES"
        if mom < -0.005 and atr < -0.5:
            return "NO"
        # 弱信号 → 用 BB 辅助
        if fr.bb_pct_1h < 0.2:
            return "YES"
        if fr.bb_pct_1h > 0.8:
            return "NO"
        return "HOLD"

    def consume_event(
        self,
        event: EventContract,
        klines_1m: Sequence,
        klines_1h: Sequence,
        klines_4h: Sequence,
        now: Optional[datetime] = None,
    ) -> Optional[Signal]:
        if event.status != "TRADING":
            return None
        now = now or datetime.now(tz=_TZ)
        seconds_to_settle = (event.settle_time - now).total_seconds()
        if seconds_to_settle < 0:
            return None

        fr = compute_all_factors(
            klines_1m=klines_1m, klines_1h=klines_1h, klines_4h=klines_4h,
        )
        side = self._decide_side(fr)
        if side == "HOLD":
            return None

        # LR 预测胜率
        Xf = self._features_from_factors(fr)
        try:
            probs = self.model.predict_proba(Xf)[0]
            win_prob = float(probs[1])  # 类别 1 = YES 胜（在我们训练数据里是"信号方向胜"）
        except Exception as e:
            logger.exception("ml_v1 predict_proba failed: %s", e)
            return None

        # ML 输出的 confidence = LR 预测胜率（更接近真实胜率）
        confidence = float(max(0.0, min(1.0, win_prob)))

        if side == "YES":
            entry_price = event.current_yes_price
        else:
            entry_price = event.current_no_price
        payoff_ratio = compute_payoff_ratio(entry_price)
        ev = compute_expected_value(win_prob, payoff_ratio)
        # 防御性 clamp：dataContract §1.4 要求 expected_value ∈ [-1, 1]
        # （ML 输出的 win_prob 可能配极低 entry 价导致 ev 爆掉）
        if ev > 1.0:
            ev = 1.0
        if ev < -1.0:
            ev = -1.0

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
            return None

        mom_pct = fr.mom_15m * 100
        rationale = f"ML 胜率{win_prob*100:.1f}% 动量{mom_pct:+.2f}% BB%b={fr.bb_pct_1h:.2f} → {side}"
        if len(rationale) > 80:
            rationale = rationale[:80]

        return Signal(
            signal_id=make_signal_id(event.event_id, now),
            ts=now,
            event_id=event.event_id,
            event_title=event.title,
            symbol=event.symbol,
            side=side,
            entry_price=round(float(entry_price), 4),
            spread=round(float(event.spread), 4),
            confidence=round(confidence, 6),
            expected_value=round(float(ev), 6),
            win_prob=round(float(win_prob), 6),
            payoff_ratio=round(float(payoff_ratio), 6),
            rationale=rationale,
            strategy=self.strategy_name,
            factors=factors_dict,
            ttl_seconds=self.ttl_seconds,
            expire_at=now + timedelta(seconds=self.ttl_seconds),
        )

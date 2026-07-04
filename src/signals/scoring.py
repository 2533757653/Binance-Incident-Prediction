"""
Builder-B 评分与过滤模块。

期望值公式：
    expected_value = win_prob * payoff_ratio - (1 - win_prob)
    payoff_ratio = (1 - entry_price) / entry_price

过滤条件（任一不满足即丢弃）：
    - expected_value < 0
    - confidence < min_confidence
    - spread > max_spread
    - seconds_to_settle < min_seconds_to_settle
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ============================================================
# 评分数据类
# ============================================================
@dataclass
class ScoringConfig:
    min_confidence: float = 0.20       # 软过滤（降低，让真实市场也能出信号）
    min_expected_value: float = 0.0    # 期望值 < 0 不出
    min_payoff_ratio: float = 0.5      # 入场价 ≤ 0.667（赔率 ≥ 0.5）
    max_spread: float = 0.06
    min_seconds_to_settle: int = 60
    max_seconds_to_settle: int = 4 * 3600  # 4h
    toast_threshold: float = 0.60      # 弹窗阈值


# ============================================================
# 数值计算
# ============================================================
def compute_payoff_ratio(entry_price: float) -> float:
    """
    payoff_ratio = (1 - entry) / entry。
    entry 必须在 (0, 1) 之间。
    """
    if entry_price <= 0 or entry_price >= 1:
        raise ValueError(f"entry_price must be in (0, 1), got {entry_price}")
    return (1.0 - entry_price) / entry_price


def compute_expected_value(win_prob: float, payoff_ratio: float) -> float:
    """
    expected_value = win_prob * payoff_ratio - (1 - win_prob)。
    简化：= win_prob * (payoff_ratio + 1) - 1。
    """
    if win_prob < 0 or win_prob > 1:
        raise ValueError(f"win_prob must be in [0, 1], got {win_prob}")
    if payoff_ratio < 0:
        raise ValueError(f"payoff_ratio must be >= 0, got {payoff_ratio}")
    return win_prob * payoff_ratio - (1.0 - win_prob)


# ============================================================
# 过滤
# ============================================================
@dataclass
class FilterDecision:
    passed: bool
    reason: str = ""
    expected_value: float = 0.0
    payoff_ratio: float = 0.0


def passes_filter(
    *,
    confidence: float,
    expected_value: float,
    spread: float,
    seconds_to_settle: float,
    payoff_ratio: float = 0.0,
    config: Optional[ScoringConfig] = None,
) -> FilterDecision:
    """
    任一硬性条件不满足即返回 passed=False + 原因。
    """
    cfg = config or ScoringConfig()
    reasons: list[str] = []
    if expected_value < cfg.min_expected_value:
        reasons.append(f"expected_value<{cfg.min_expected_value}")
    if confidence < cfg.min_confidence:
        reasons.append(f"confidence<{cfg.min_confidence}")
    if spread > cfg.max_spread:
        reasons.append(f"spread>{cfg.max_spread}")
    if seconds_to_settle < cfg.min_seconds_to_settle:
        reasons.append(f"seconds_to_settle<{cfg.min_seconds_to_settle}")
    if seconds_to_settle > cfg.max_seconds_to_settle:
        reasons.append(f"seconds_to_settle>{cfg.max_seconds_to_settle}")
    if payoff_ratio > 0 and payoff_ratio < cfg.min_payoff_ratio:
        reasons.append(f"payoff_ratio<{cfg.min_payoff_ratio}")
    if reasons:
        return FilterDecision(passed=False, reason=";".join(reasons),
                              expected_value=expected_value, payoff_ratio=payoff_ratio)
    return FilterDecision(passed=True, expected_value=expected_value, payoff_ratio=payoff_ratio)

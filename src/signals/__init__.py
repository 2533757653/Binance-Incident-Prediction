"""Horizon-Incident 信号引擎。"""
from .models import Signal
from .factors import (
    compute_mom_15m,
    compute_bb_pct_1h,
    compute_atr_break_4h,
    compute_all_factors,
    FactorResult,
)
from .scoring import (
    compute_payoff_ratio,
    compute_expected_value,
    passes_filter,
)
from .generator import SignalGenerator

__all__ = [
    "Signal",
    "compute_mom_15m",
    "compute_bb_pct_1h",
    "compute_atr_break_4h",
    "compute_all_factors",
    "FactorResult",
    "compute_payoff_ratio",
    "compute_expected_value",
    "passes_filter",
    "SignalGenerator",
]

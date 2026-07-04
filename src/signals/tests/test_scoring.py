"""
Builder-B 评分模块单测。
"""
from __future__ import annotations

import pytest

from src.signals.scoring import (
    ScoringConfig,
    compute_expected_value,
    compute_payoff_ratio,
    passes_filter,
)


# ============================================================
# payoff_ratio
# ============================================================
def test_payoff_ratio_50_50():
    """entry=0.5 → payoff = (1-0.5)/0.5 = 1.0。"""
    assert compute_payoff_ratio(0.5) == 1.0


def test_payoff_ratio_high_price():
    """entry=0.8 → payoff = 0.25。"""
    assert abs(compute_payoff_ratio(0.8) - 0.25) < 1e-12


def test_payoff_ratio_low_price():
    """entry=0.2 → payoff = 4.0。"""
    assert abs(compute_payoff_ratio(0.2) - 4.0) < 1e-12


def test_payoff_ratio_invalid():
    with pytest.raises(ValueError):
        compute_payoff_ratio(0)
    with pytest.raises(ValueError):
        compute_payoff_ratio(1)
    with pytest.raises(ValueError):
        compute_payoff_ratio(-0.1)
    with pytest.raises(ValueError):
        compute_payoff_ratio(1.5)


# ============================================================
# expected_value
# ============================================================
def test_expected_value_basic():
    """win_prob=0.6, payoff=1 → ev = 0.6*1 - 0.4 = 0.2。"""
    assert abs(compute_expected_value(0.6, 1.0) - 0.2) < 1e-12


def test_expected_value_break_even():
    """win_prob=0.5, payoff=1 → ev = 0。"""
    assert abs(compute_expected_value(0.5, 1.0)) < 1e-12


def test_expected_value_negative():
    """win_prob=0.3, payoff=1 → ev = -0.4。"""
    assert abs(compute_expected_value(0.3, 1.0) - (-0.4)) < 1e-12


# ============================================================
# passes_filter
# ============================================================
def test_filter_pass():
    d = passes_filter(
        confidence=0.6, expected_value=0.1,
        spread=0.02, seconds_to_settle=1800, payoff_ratio=1.5,
    )
    assert d.passed is True
    assert d.reason == ""


def test_filter_fail_low_confidence():
    d = passes_filter(
        confidence=0.3, expected_value=0.1,
        spread=0.02, seconds_to_settle=1800, payoff_ratio=1.5,
    )
    assert d.passed is False
    assert "confidence" in d.reason


def test_filter_fail_negative_ev():
    d = passes_filter(
        confidence=0.6, expected_value=-0.1,
        spread=0.02, seconds_to_settle=1800, payoff_ratio=1.5,
    )
    assert d.passed is False
    assert "expected_value" in d.reason


def test_filter_fail_wide_spread():
    d = passes_filter(
        confidence=0.6, expected_value=0.1,
        spread=0.10, seconds_to_settle=1800, payoff_ratio=1.5,
    )
    assert d.passed is False
    assert "spread" in d.reason


def test_filter_fail_too_close_to_settle():
    d = passes_filter(
        confidence=0.6, expected_value=0.1,
        spread=0.02, seconds_to_settle=30, payoff_ratio=1.5,
    )
    assert d.passed is False
    assert "seconds_to_settle" in d.reason


def test_filter_fail_too_far_to_settle():
    d = passes_filter(
        confidence=0.6, expected_value=0.1,
        spread=0.02, seconds_to_settle=10 * 3600, payoff_ratio=1.5,
    )
    assert d.passed is False
    assert "seconds_to_settle" in d.reason

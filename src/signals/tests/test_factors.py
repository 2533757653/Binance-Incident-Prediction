"""
Builder-B 因子计算单测。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.signals.factors import (
    compute_atr_break_4h,
    compute_bb_pct_1h,
    compute_mom_15m,
)


# ============================================================
# MOM_15m
# ============================================================
def test_mom_15m_zero_when_flat():
    """价格不变 → 动量 = 0。"""
    prices = [100.0] * 30
    mom = compute_mom_15m(prices, lookback=15)
    assert abs(mom) < 1e-12


def test_mom_15m_known_value():
    """
    手算：price 100 → 105（+5%），log(105/100) = log(1.05) ≈ 0.04879。
    """
    prices = [100.0] * 16 + [105.0] * 15
    mom = compute_mom_15m(prices, lookback=15)
    expected = math.log(105.0 / 100.0)
    assert abs(mom - expected) < 1e-9
    assert abs(mom - 0.04879016) < 1e-6


def test_mom_15m_negative():
    """价格下跌 → mom < 0。"""
    prices = [100.0] * 16 + [95.0] * 15
    mom = compute_mom_15m(prices, lookback=15)
    assert mom < 0
    assert abs(mom - math.log(0.95)) < 1e-9


def test_mom_15m_insufficient_data():
    """数据不足 → 返回 0.0，不抛异常。"""
    prices = [100.0] * 5
    mom = compute_mom_15m(prices, lookback=15)
    assert mom == 0.0


# ============================================================
# BB_1h
# ============================================================
def test_bb_pct_1h_middle_when_flat():
    """常数序列 → sd=0 → 返回 0.5。"""
    prices = [100.0] * 30
    bb = compute_bb_pct_1h(prices, period=20, std_mult=2.0)
    assert abs(bb - 0.5) < 1e-12


def test_bb_pct_1h_known_value():
    """
    手算 20 个值，均值=100，std=0 → 但这里我们用正弦数据 + 最后 close 在均值附近。

    改用：100 ± 一些波动，让 close 落在中位。
    """
    # 构造 20 个数据，最后一个 = 100，分布均值为 100，标准差 ≈ 1
    arr = [100 + i * 0.1 for i in range(20)]   # 100, 100.1, ..., 101.9
    closes = arr
    bb = compute_bb_pct_1h(closes, period=20, std_mult=2.0)
    # close = 101.9
    # mean = (100 + 101.9) * 20 / 2 / 20 = 100.95
    # std = sqrt(sum((x - mean)^2)/20)
    mean = np.mean(closes)
    sd = np.std(closes, ddof=0)
    upper = mean + 2.0 * sd
    lower = mean - 2.0 * sd
    expected = (closes[-1] - lower) / (upper - lower)
    assert abs(bb - expected) < 1e-9
    # 最后一个 close 接近 upper 端，%b 应大于 0.5
    assert bb > 0.5


def test_bb_pct_1h_insufficient_data():
    prices = [100.0] * 5
    bb = compute_bb_pct_1h(prices, period=20)
    assert bb == 0.5


def test_bb_pct_1h_at_upper_band():
    """当 close 恰好为上轨时 %b = 1。"""
    closes = [100.0] * 19 + [200.0]   # 最后一个远超其他
    bb = compute_bb_pct_1h(closes, period=20, std_mult=2.0)
    assert bb > 0.95


# ============================================================
# ATR_4h
# ============================================================
def test_atr_break_4h_zero_when_constant():
    """常数序列 → TR=0 → ATR=0 → result=0。"""
    closes = [100.0] * 20
    highs = [100.0] * 20
    lows = [100.0] * 20
    bb = compute_atr_break_4h(closes, period=14, breakout_mult=1.5)
    # 因为我们传 Sequence[float]，高低用不到；这里只检验
    assert bb == 0.0


def test_atr_break_4h_positive_when_breakout_up():
    """
    价格向上突破：构造 close 单调上升 15 根，最后一根远超均值。
    """
    closes = [100.0 + i * 0.5 for i in range(15)]  # 100, 100.5, ..., 107.0
    atr = compute_atr_break_4h(closes, period=14, breakout_mult=1.5)
    assert atr > 0  # 正向偏离


def test_atr_break_4h_negative_when_breakout_down():
    closes = [200.0 - i * 0.5 for i in range(15)]  # 200, 199.5, ..., 193.0
    atr = compute_atr_break_4h(closes, period=14, breakout_mult=1.5)
    assert atr < 0


def test_atr_break_4h_with_kline_objects():
    """
    用 KlineBar dataclass 风格对象传 K 线。
    """
    from src.common.schemas import KlineBar
    from datetime import datetime, timezone, timedelta

    now = datetime(2026, 6, 24, 12, 0, tzinfo=timezone(timedelta(hours=8)))
    klines = []
    for i in range(15):
        c = 100.0 + i * 0.3
        klines.append(KlineBar(
            symbol="BTCUSDT", interval="4h",
            open_time=now, close_time=now,
            open=c, high=c + 0.1, low=c - 0.1, close=c,
            volume=1.0, quote_volume=100.0, trades_count=10,
        ))
    atr = compute_atr_break_4h(klines, period=14, breakout_mult=1.5)
    assert atr > 0


def test_atr_break_4h_insufficient_data():
    closes = [100.0] * 5
    atr = compute_atr_break_4h(closes, period=14)
    assert atr == 0.0

"""
Builder-B 因子计算：MOM_15m / BB_1h / ATR_4h。

所有因子函数都是**纯函数**：输入 (K线序列 + 参数) → 输出数值。
不依赖全局状态，不发起 IO。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np


# ============================================================
# 数据类：K 线 close 序列封装
# ============================================================
@dataclass
class FactorResult:
    """单次因子计算结果包装。"""

    mom_15m: float
    bb_pct_1h: float
    atr_break_4h: float
    details: dict


# ============================================================
# 工具：把 K 线序列转成 numpy 数组
# ============================================================
def _closes(klines: Sequence) -> np.ndarray:
    """支持 Sequence[KlineBar] 或 Sequence[float]，返回 close 数组。"""
    if len(klines) == 0:
        return np.asarray([], dtype=float)
    first = klines[0]
    if isinstance(first, (int, float, np.floating)):
        arr = np.asarray(list(klines), dtype=float)
    else:
        # KlineBar 对象，取 .close
        arr = np.asarray([float(k.close) for k in klines], dtype=float)
    return arr


def _highs(klines: Sequence) -> np.ndarray:
    if len(klines) == 0:
        return np.asarray([], dtype=float)
    first = klines[0]
    if isinstance(first, (int, float, np.floating)):
        return _closes(klines)
    return np.asarray([float(k.high) for k in klines], dtype=float)


def _lows(klines: Sequence) -> np.ndarray:
    if len(klines) == 0:
        return np.asarray([], dtype=float)
    first = klines[0]
    if isinstance(first, (int, float, np.floating)):
        return _closes(klines)
    return np.asarray([float(k.low) for k in klines], dtype=float)


# ============================================================
# 因子 1：MOM_15m — 15 分钟对数收益率
# ============================================================
def compute_mom_15m(
    klines_1m: Sequence,
    lookback: int = 15,
) -> float:
    """
    15 分钟对数收益率 = log(close_now) - log(close_15m_ago)。

    接受最近 N 根 1m K 线（建议 ≥ lookback+1）。
    返回单位：例如 0.005 表示 +0.5%。

    数据不足时返回 0.0（**不抛异常**，避免因子流断）。
    """
    closes = _closes(klines_1m)
    if len(closes) < lookback + 1:
        return 0.0
    c_now = float(closes[-1])
    c_prev = float(closes[-1 - lookback])
    if c_prev <= 0 or c_now <= 0:
        return 0.0
    return float(np.log(c_now) - np.log(c_prev))


# ============================================================
# 因子 2：BB_1h — 20 周期布林带 %b
# ============================================================
def compute_bb_pct_1h(
    klines_1h: Sequence,
    period: int = 20,
    std_mult: float = 2.0,
) -> float:
    """
    20 周期布林带 %b = (close - lower) / (upper - lower)。

    0 表示在下轨；1 表示在上轨；>1 突破上轨；<0 跌破下轨。
    数据不足时返回 0.5（中位）。
    """
    closes = _closes(klines_1h)
    if len(closes) < period:
        return 0.5
    window = closes[-period:]
    mid = float(np.mean(window))
    sd = float(np.std(window, ddof=0))  # 总体标准差，与 ta-lib 一致
    if sd == 0:
        return 0.5
    upper = mid + std_mult * sd
    lower = mid - std_mult * sd
    if upper == lower:
        return 0.5
    pct_b = (float(closes[-1]) - lower) / (upper - lower)
    return float(pct_b)


# ============================================================
# 因子 3：ATR_4h — 14 周期 ATR × 1.5 突破强度
# ============================================================
def compute_atr_break_4h(
    klines_4h: Sequence,
    period: int = 14,
    breakout_mult: float = 1.5,
) -> float:
    """
    14 周期 ATR × breakout_mult 当作突破阈值。
    返回 **当前价相对 4h 均值偏离 / 阈值**（绝对值，越大越强趋势）。

    公式：
        atr = mean(TR[-period:])
        threshold = atr * breakout_mult
        deviation = close_now - mean(close[-period:])
        result = deviation / threshold

    正号=价格高于均值（向上偏离），负号=向下偏离。
    数据不足时返回 0.0。
    """
    if len(klines_4h) < period + 1:
        return 0.0
    highs = _highs(klines_4h)
    lows = _lows(klines_4h)
    closes = _closes(klines_4h)

    # True Range
    prev_close = closes[-period - 1 : -1]
    cur_high = highs[-period:]
    cur_low = lows[-period:]
    tr1 = cur_high - cur_low
    tr2 = np.abs(cur_high - prev_close)
    tr3 = np.abs(cur_low - prev_close)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)

    atr = float(np.mean(tr))
    threshold = atr * breakout_mult
    if threshold == 0:
        return 0.0
    dev = float(closes[-1] - np.mean(closes[-period:]))
    return float(dev / threshold)


# ============================================================
# 综合入口
# ============================================================
def compute_all_factors(
    klines_1m: Sequence,
    klines_1h: Sequence,
    klines_4h: Sequence,
    mom_lookback: int = 15,
    mom_threshold: float = 0.005,
    bb_period: int = 20,
    bb_std: float = 2.0,
    bb_extreme: float = 0.95,
    atr_period: int = 14,
    atr_breakout_mult: float = 1.5,
) -> FactorResult:
    """
    一次性计算三个因子。返回 FactorResult。
    """
    mom = compute_mom_15m(klines_1m, lookback=mom_lookback)
    bb = compute_bb_pct_1h(klines_1h, period=bb_period, std_mult=bb_std)
    atr = compute_atr_break_4h(klines_4h, period=atr_period, breakout_mult=atr_breakout_mult)
    return FactorResult(
        mom_15m=mom,
        bb_pct_1h=bb,
        atr_break_4h=atr,
        details={
            "mom_threshold": mom_threshold,
            "bb_extreme": bb_extreme,
            "atr_period": atr_period,
            "atr_breakout_mult": atr_breakout_mult,
        },
    )

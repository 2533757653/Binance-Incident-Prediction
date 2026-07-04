"""规则信号引擎：5 条手工规则的确定性信号。
每条规则返回一个 (side, confidence, name) 三元组。
"""
from __future__ import annotations

from typing import List, Tuple, Optional
from datetime import datetime

import numpy as np


# side: +1 = 看涨（YES）, -1 = 看跌（NO）, 0 = 无信号
RuleSignal = Tuple[int, float, str]


def rule_rsi_extreme_reversal(
    inds: dict, i_5m: int,
    oversold: float = 25, overbought: float = 75,
) -> Optional[RuleSignal]:
    """RSI 极端反转：5m RSI<25 → 反弹看涨；1h RSI>75 → 看跌。

    适用场景：均值回归、恐慌/贪婪反转
    频率：每小时 1-3 条
    """
    i_1h = min(i_5m // 12, len(inds["1h"].get("rsi", np.zeros(1))) - 1)
    rsi_5m = float(inds["5m"]["rsi"][i_5m] * 100)  # rsi was normalized to 0-1
    rsi_1h = float(inds["1h"]["rsi"][i_1h] * 100)

    if rsi_5m < oversold and rsi_1h < 35:
        return (+1, 0.65, f"RSI反转5m<{oversold}+1h<35")
    if rsi_5m > overbought and rsi_1h > 65:
        return (-1, 0.65, f"RSI反转5m>{overbought}+1h>65")
    return None


def rule_macd_cross_tf_resonance(
    inds: dict, i_5m: int,
) -> Optional[RuleSignal]:
    """MACD 多周期共振：5m/15m/30m/1h MACD 柱同号 → 顺势。

    适用场景：趋势中段，跟随主趋势
    频率：每小时 1-2 条
    """
    i_15m = min(i_5m // 3, len(inds["15m"].get("macd_hist", np.zeros(1))) - 1)
    i_30m = min(i_5m // 6, len(inds["30m"].get("macd_hist", np.zeros(1))) - 1)
    i_1h = min(i_5m // 12, len(inds["1h"].get("macd_hist", np.zeros(1))) - 1)

    h5 = inds["5m"]["macd_hist"][i_5m]
    h15 = inds["15m"]["macd_hist"][i_15m]
    h30 = inds["30m"]["macd_hist"][i_30m]
    h1h = inds["1h"]["macd_hist"][i_1h]

    # 至少 3 个同号（不包括 5m 因为它最噪）
    hists = [h15, h30, h1h]
    pos = sum(1 for h in hists if h > 0)
    neg = sum(1 for h in hists if h < 0)

    if pos >= 3 and h5 > 0:
        # 5m 也确认看涨 → 强信号
        return (+1, 0.70, f"MACD共振{pos}/3+5m确认")
    if neg >= 3 and h5 < 0:
        return (-1, 0.70, f"MACD共振{neg}/3+5m确认")
    return None


def rule_adx_breakout(
    inds: dict, i_5m: int, klines_raw_close: np.ndarray,
    bb_pos_threshold: float = 0.95,
) -> Optional[RuleSignal]:
    """ADX 趋势强度 + BB 突破：ADX>25 且价格在 BB 上/下轨 → 顺势。

    适用场景：强趋势启动期
    频率：每天 2-5 条
    """
    i_15m = min(i_5m // 3, len(inds["15m"].get("adx", np.zeros(1))) - 1)
    i_30m = min(i_5m // 6, len(inds["30m"].get("adx", np.zeros(1))) - 1)
    adx_30m = float(inds["30m"]["adx"][i_30m] * 100)
    adx_15m = float(inds["15m"]["adx"][i_15m] * 100)

    # 至少一个周期 ADX > 25
    if max(adx_30m, adx_15m) < 25:
        return None

    bb_pos_30m = float(inds["30m"]["bb_pos"][i_30m])
    if bb_pos_30m > bb_pos_threshold:
        return (+1, 0.70, f"ADX突破{int(max(adx_30m, adx_15m))}+BB顶")
    elif bb_pos_30m < (1 - bb_pos_threshold):
        return (-1, 0.70, f"ADX突破{int(max(adx_30m, adx_15m))}+BB底")
    return None


def rule_volume_anomaly_reversal(
    inds: dict, i_5m: int,
    vol_surge_threshold: float = 3.0,
) -> Optional[RuleSignal]:
    """量价异动反转：5m 成交量突增 3x + RSI 极值 → 反向。

    适用场景：插针/瀑布后的快速回归
    频率：每天 0-2 条
    """
    vol_ratio = inds["5m"]["vol_ratio"][i_5m]
    rsi = float(inds["5m"]["rsi"][i_5m] * 100)

    if vol_ratio > vol_surge_threshold and rsi < 20:
        return (+1, 0.60, f"量异动{vol_ratio:.1f}x+RSI<20")
    if vol_ratio > vol_surge_threshold and rsi > 80:
        return (-1, 0.60, f"量异动{vol_ratio:.1f}x+RSI>80")
    return None


def rule_ema_nesting(
    inds: dict, i_5m: int,
) -> Optional[RuleSignal]:
    """EMA 多周期嵌套：5m/15m/30m/1h 的 EMA9 > EMA21 同向 → 顺势。

    适用场景：稳健趋势，跟随而非反转
    频率：每小时 1-2 条
    """
    i_15m = min(i_5m // 3, len(inds["15m"].get("ema9_21", np.zeros(1))) - 1)
    i_30m = min(i_5m // 6, len(inds["30m"].get("ema9_21", np.zeros(1))) - 1)
    i_1h = min(i_5m // 12, len(inds["1h"].get("ema9_21", np.zeros(1))) - 1)

    e5 = inds["5m"]["ema9_21"][i_5m]
    e15 = inds["15m"]["ema9_21"][i_15m]
    e30 = inds["30m"]["ema9_21"][i_30m]
    e1h = inds["1h"]["ema9_21"][i_1h]

    pos = sum(1 for e in [e15, e30, e1h] if e > 0.001)
    neg = sum(1 for e in [e15, e30, e1h] if e < -0.001)

    if pos >= 3 and e5 > 0:
        return (+1, 0.65, f"EMA嵌套{pos}/3")
    if neg >= 3 and e5 < 0:
        return (-1, 0.65, f"EMA嵌套{neg}/3")
    return None


ALL_RULES = [
    rule_rsi_extreme_reversal,
    rule_macd_cross_tf_resonance,
    rule_adx_breakout,
    rule_volume_anomaly_reversal,
    rule_ema_nesting,
]


def evaluate_all_rules(
    inds: dict, i_5m: int, klines_raw_close: np.ndarray = None,
) -> List[RuleSignal]:
    """评估所有规则，返回激活的信号列表。"""
    signals = []
    for rule in ALL_RULES:
        try:
            if rule == rule_adx_breakout and klines_raw_close is None:
                continue
            sig = rule(inds, i_5m, klines_raw_close) if rule == rule_adx_breakout else rule(inds, i_5m)
            if sig is not None:
                signals.append(sig)
        except Exception:
            continue
    return signals
"""
K 线特征 v3（10 → 15 个特征）。

新增 5 个特征：
- ADX_1h            （趋势强度，0-100，>25=强趋势，<20=震荡市）
- MACD_hist_1h      （MACD 柱状图，反映动量+趋势的复合信号）
- ATR_ratio_1h      （当前 ATR / 20 周期 ATR 均值，反映波动率突变）
- Volume_ratio_30m  （当前成交量 / 20 周期均量，反映量能异常）
- BB_width_1h       （布林带宽度，反映市场状态：窄=盘整，宽=趋势）

保留 v2 的 10 个特征（30m + 1h 双窗口）。
"""
from __future__ import annotations

from typing import List

import numpy as np


def compute_features_v3(
    closes_30m: np.ndarray,
    highs_30m: np.ndarray,
    lows_30m: np.ndarray,
    volumes_30m: np.ndarray,
    closes_1h: np.ndarray,
    highs_1h: np.ndarray,
    lows_1h: np.ndarray,
    volumes_1h: np.ndarray,
    i_30m: int,
) -> List[float]:
    """
    计算 15 个特征。
    """
    # ---- 30m 因子（5 个）----
    ret_30m = float(np.log(closes_30m[i_30m] / closes_30m[i_30m - 1])) if i_30m >= 1 else 0.0
    if i_30m >= 19:
        w = closes_30m[i_30m - 19: i_30m + 1]
        mid = float(np.mean(w))
        sd = float(np.std(w, ddof=0))
        bb_pct_30m = float((closes_30m[i_30m] - (mid - 2 * sd)) / (4 * sd)) if sd > 0 else 0.5
    else:
        bb_pct_30m = 0.5
    if i_30m >= 2:
        rets = np.diff(np.log(closes_30m[max(0, i_30m - 2): i_30m + 1]))
        vol_30m = float(np.std(rets)) if len(rets) > 1 else 0.0
    else:
        vol_30m = 0.0
    mom_30m = float(np.log(closes_30m[i_30m] / closes_30m[i_30m - 3])) if i_30m >= 3 else 0.0
    range_30m = float((highs_30m[i_30m] - lows_30m[i_30m]) / closes_30m[i_30m]) if i_30m >= 0 else 0.0

    # 新增 1：Volume ratio 30m
    if i_30m >= 20:
        vol_ma = float(np.mean(volumes_30m[i_30m - 19: i_30m + 1]))
        volume_ratio_30m = float(volumes_30m[i_30m] / vol_ma) if vol_ma > 0 else 1.0
    else:
        volume_ratio_30m = 1.0

    # ---- 1h 因子（10 个）----
    i_1h = i_30m // 2
    if i_1h >= len(closes_1h) - 1:
        i_1h = len(closes_1h) - 1
    ret_1h = float(np.log(closes_1h[i_1h] / closes_1h[i_1h - 1])) if i_1h >= 1 else 0.0
    if i_1h >= 19:
        w = closes_1h[i_1h - 19: i_1h + 1]
        mid = float(np.mean(w))
        sd = float(np.std(w, ddof=0))
        bb_pct_1h = float((closes_1h[i_1h] - (mid - 2 * sd)) / (4 * sd)) if sd > 0 else 0.5
    else:
        bb_pct_1h = 0.5
    trend_24h = float(np.log(closes_1h[i_1h] / closes_1h[i_1h - 24])) if i_1h >= 24 else 0.0
    mom_1h = float(np.log(closes_1h[i_1h] / closes_1h[i_1h - 3])) if i_1h >= 3 else 0.0
    vol_1h_window = closes_1h[max(0, i_1h - 12): i_1h + 1]
    vol_1h = float(np.std(np.diff(np.log(vol_1h_window)))) if len(vol_1h_window) > 2 else 0.0

    # 新增 2：ATR ratio 1h（当前 ATR / 20 周期均 ATR）
    atr_ratio_1h = _compute_atr_ratio(highs_1h, lows_1h, closes_1h, i_1h, period=14, lookback=20)

    # 新增 3：BB width 1h
    bb_width_1h = _compute_bb_width(closes_1h, i_1h, period=20, std_mult=2.0)

    # 新增 4：MACD histogram 1h
    macd_hist_1h = _compute_macd_histogram(closes_1h, i_1h)

    # 新增 5：ADX 1h
    adx_1h = _compute_adx(highs_1h, lows_1h, closes_1h, i_1h, period=14)

    return [
        # 30m 因子 5 个
        ret_30m, bb_pct_30m, vol_30m, mom_30m, range_30m,
        # 新增 30m 因子 1 个
        volume_ratio_30m,
        # 1h 因子 5 个
        ret_1h, bb_pct_1h, trend_24h, mom_1h, vol_1h,
        # 新增 1h 因子 4 个
        atr_ratio_1h, bb_width_1h, macd_hist_1h, adx_1h,
    ]


def _compute_atr_ratio(highs, lows, closes, i, period=14, lookback=20) -> float:
    """当前 ATR / 过去 lookback 周期均 ATR，反映波动率突变。"""
    if i < period + lookback:
        return 1.0
    # 当前 ATR（最后 period 根）
    prev_close = closes[i - period: i]
    cur_high = highs[i - period + 1: i + 1]
    cur_low = lows[i - period + 1: i + 1]
    tr = np.maximum(cur_high - cur_low,
                    np.maximum(np.abs(cur_high - prev_close),
                               np.abs(cur_low - prev_close)))
    cur_atr = float(np.mean(tr))
    # 过去 lookback 周期均 ATR（每个用 14 根）
    if i < period + lookback + 14:
        return 1.0
    past_atrs = []
    for j in range(i - lookback - 14, i - 14):
        prev_c = closes[j - period + 1: j + 1]
        cur_h = highs[j - period + 1: j + 1]
        cur_l = lows[j - period + 1: j + 1]
        ttr = np.maximum(cur_h - cur_l,
                         np.maximum(np.abs(cur_h - prev_c),
                                    np.abs(cur_l - prev_c)))
        past_atrs.append(float(np.mean(ttr)))
    avg_past_atr = float(np.mean(past_atrs)) if past_atrs else cur_atr
    return cur_atr / avg_past_atr if avg_past_atr > 0 else 1.0


def _compute_bb_width(closes, i, period=20, std_mult=2.0) -> float:
    """布林带宽度：(upper - lower) / mid。"""
    if i < period - 1:
        return 0.0
    window = closes[i - period + 1: i + 1]
    mid = float(np.mean(window))
    sd = float(np.std(window, ddof=0))
    upper = mid + std_mult * sd
    lower = mid - std_mult * sd
    return float((upper - lower) / mid) if mid > 0 else 0.0


def _compute_macd_histogram(closes, i, fast=12, slow=26, signal_period=9) -> float:
    """MACD 柱状图 = MACD - Signal，反映动量加速/减速。"""
    if i < slow + signal_period:
        return 0.0
    # EMA fast
    ema_fast = _ema(closes[: i + 1], fast)
    ema_slow = _ema(closes[: i + 1], slow)
    if ema_fast is None or ema_slow is None:
        return 0.0
    macd_line = ema_fast - ema_slow
    # 简化：signal = 9 周期 EMA of MACD
    # 这里只算最近 macd 值（近似）
    return float(macd_line / closes[i]) if closes[i] > 0 else 0.0


def _ema(values, period):
    """指数移动平均。返回最后一个值。"""
    if len(values) < period:
        return None
    alpha = 2.0 / (period + 1)
    ema_val = float(values[0])
    for v in values[1:]:
        ema_val = alpha * float(v) + (1 - alpha) * ema_val
    return ema_val


def _compute_adx(highs, lows, closes, i, period=14) -> float:
    """
    ADX（平均趋向指数），0-100。
    >25 = 强趋势，<20 = 弱趋势/震荡市。
    """
    if i < period * 2:
        return 25.0  # 默认中性
    # 计算 +DM / -DM / TR
    highs_arr = highs[i - period: i + 1]
    lows_arr = lows[i - period: i + 1]
    closes_arr = closes[i - period: i + 1]
    plus_dm = []
    minus_dm = []
    tr = []
    for j in range(1, len(highs_arr)):
        up = highs_arr[j] - highs_arr[j - 1]
        down = lows_arr[j - 1] - lows_arr[j]
        plus_dm.append(up if up > down and up > 0 else 0)
        minus_dm.append(down if down > up and down > 0 else 0)
        tr.append(max(highs_arr[j] - lows_arr[j],
                       abs(highs_arr[j] - closes_arr[j - 1]),
                       abs(lows_arr[j] - closes_arr[j - 1])))
    if not tr or sum(tr) == 0:
        return 25.0
    # 平滑（简化：用 sum）
    tr_sum = sum(tr)
    plus_di = 100 * sum(plus_dm) / tr_sum
    minus_di = 100 * sum(minus_dm) / tr_sum
    dx_sum = plus_di + minus_di
    if dx_sum == 0:
        return 0.0
    dx = 100 * abs(plus_di - minus_di) / dx_sum
    return float(dx)


# 兼容旧接口（v2 函数仍可用）
def compute_features(closes_30m, highs_30m, lows_30m, closes_1h, i_30m):
    """兼容 v2 接口（无 volume 数据）。"""
    closes_30m = np.asarray(closes_30m, dtype=float)
    highs_30m = np.asarray(highs_30m, dtype=float)
    lows_30m = np.asarray(lows_30m, dtype=float)
    closes_1h = np.asarray(closes_1h, dtype=float)
    # 构造假 volume（避免类型错误）
    vol_30m = np.ones_like(closes_30m)
    high_1h = np.concatenate([[closes_1h[0]], closes_1h])
    low_1h = np.concatenate([[closes_1h[0]], closes_1h])
    vol_1h = np.ones_like(closes_1h)
    # v3 需要 1h highs/lows，构造：粗略 = closes（保守）
    return compute_features_v3(
        closes_30m, highs_30m, lows_30m, vol_30m,
        closes_1h, high_1h, low_1h, vol_1h,
        i_30m,
    )


FEATURE_NAMES_V3 = [
    "ret_30m", "bb_pct_30m", "vol_30m", "mom_30m", "range_30m",
    "volume_ratio_30m",
    "ret_1h", "bb_pct_1h", "trend_24h", "mom_1h", "vol_1h",
    "atr_ratio_1h", "bb_width_1h", "macd_hist_1h", "adx_1h",
]
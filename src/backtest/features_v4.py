"""
事件合约 v4 full：30 个 ta 库技术指标（全副武装）。

特征类别（30）：
  30m (9): RSI, ROC, Williams%R, Stoch K/D, EMA cross, close/SMA20, BB width, ATR ratio
  1h (11): RSI, MACD(3), TSI, ADX, EMA cross(2), CCI, ATR, BB pos, CMF
  4h (2): EMA cross, ADX
  统计 (5): vol_1h, ret_mean_1h, skew, kurt, autocorr
  时间 (2): hour_norm, hour_block
  价格位置 (1): price_pos_30m

阈值：0.80/0.20（降低频率+提高胜率）
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

log = logging.getLogger("features.v4")


def _safe(v, default=0.0):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _safe_div(a, b, default=0.0):
    try:
        f = float(a) / float(b)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def compute_features_v4(
    closes_30m, highs_30m, lows_30m, volumes_30m,
    closes_1h, highs_1h, lows_1h, volumes_1h,
    closes_4h,  # unused in v4
    i_30m: int,
) -> List[float]:
    """
    v4 full：30 个 ta 库特征。
    仍保持高性能（整个序列一次构造 DataFrame，iloc 拿最后一刻的值）。
    """
    df30 = pd.DataFrame({"close": closes_30m, "high": highs_30m, "low": lows_30m, "volume": volumes_30m})
    df1h = pd.DataFrame({"close": closes_1h, "high": highs_1h, "low": lows_1h, "volume": volumes_1h})
    df4h = pd.DataFrame({"close": closes_4h, "high": closes_4h, "low": closes_4h, "volume": np.ones_like(closes_4h)})

    features = []

    # ======== 30m indicators (9) ========
    try:
        from ta.momentum import RSIIndicator, ROCIndicator, WilliamsRIndicator, StochasticOscillator
        from ta.trend import EMAIndicator, SMAIndicator
        from ta.volatility import BollingerBands, AverageTrueRange

        features.append(_safe(RSIIndicator(df30["close"], 14).rsi().iloc[i_30m] / 100))
        features.append(_safe(ROCIndicator(df30["close"], 10).roc().iloc[i_30m] / 100))
        features.append(_safe(WilliamsRIndicator(df30["high"], df30["low"], df30["close"], 14).williams_r().iloc[i_30m] / 100))
        so = StochasticOscillator(df30["high"], df30["low"], df30["close"], 14, 3)
        features.append(_safe(so.stoch().iloc[i_30m] / 100))
        features.append(_safe(so.stoch_signal().iloc[i_30m] / 100))
        ema9 = EMAIndicator(df30["close"], 9).ema_indicator()
        ema21 = EMAIndicator(df30["close"], 21).ema_indicator()
        features.append(_safe_div(ema9.iloc[i_30m] - ema21.iloc[i_30m], ema21.iloc[i_30m]))
        sma20 = SMAIndicator(df30["close"], 20).sma_indicator()
        features.append(_safe_div(df30["close"].iloc[i_30m] - sma20.iloc[i_30m], sma20.iloc[i_30m]))
        bb = BollingerBands(df30["close"], 20, 2)
        features.append(_safe_div(bb.bollinger_hband().iloc[i_30m] - bb.bollinger_lband().iloc[i_30m], sma20.iloc[i_30m]))
        features.append(_safe_div(AverageTrueRange(df30["high"], df30["low"], df30["close"], 14).average_true_range().iloc[i_30m], df30["close"].iloc[i_30m]))
    except Exception:
        features.extend([0.0] * 9)

    # ======== 1h indicators (11) ========
    try:
        from ta.momentum import RSIIndicator, TSIIndicator
        from ta.trend import MACD, ADXIndicator, EMAIndicator, CCIIndicator
        from ta.volatility import AverageTrueRange, BollingerBands
        from ta.volume import ChaikinMoneyFlowIndicator

        i1h = min(i_30m // 2, len(df1h) - 1)
        features.append(_safe(RSIIndicator(df1h["close"], 14).rsi().iloc[i1h] / 100))
        macd = MACD(df1h["close"])
        features.append(_safe_div(macd.macd_diff().iloc[i1h], df1h["close"].iloc[i1h]))
        features.append(_safe_div(macd.macd().iloc[i1h], df1h["close"].iloc[i1h]))
        features.append(_safe_div(macd.macd_signal().iloc[i1h], df1h["close"].iloc[i1h]))
        features.append(_safe(TSIIndicator(df1h["close"]).tsi().iloc[i1h] / 100))
        adx = ADXIndicator(df1h["high"], df1h["low"], df1h["close"], 14)
        features.append(_safe(adx.adx().iloc[i1h] / 100))
        ema12 = EMAIndicator(df1h["close"], 12).ema_indicator()
        ema26 = EMAIndicator(df1h["close"], 26).ema_indicator()
        ema50 = EMAIndicator(df1h["close"], 50).ema_indicator()
        features.append(_safe_div(ema12.iloc[i1h] - ema26.iloc[i1h], ema26.iloc[i1h]))
        features.append(_safe_div(ema26.iloc[i1h] - ema50.iloc[i1h], ema50.iloc[i1h]))
        features.append(_safe(CCIIndicator(df1h["high"], df1h["low"], df1h["close"], 20).cci().iloc[i1h] / 200))
        features.append(_safe_div(AverageTrueRange(df1h["high"], df1h["low"], df1h["close"], 14).average_true_range().iloc[i1h], df1h["close"].iloc[i1h]))
        bb1h = BollingerBands(df1h["close"], 20, 2)
        features.append(_safe_div(df1h["close"].iloc[i1h] - bb1h.bollinger_mavg().iloc[i1h], bb1h.bollinger_mavg().iloc[i1h]))
        features.append(_safe(ChaikinMoneyFlowIndicator(df1h["high"], df1h["low"], df1h["close"], df1h["volume"], 20).chaikin_money_flow().iloc[i1h]))
    except Exception:
        features.extend([0.0] * 12)
    try:
        from ta.trend import EMAIndicator, ADXIndicator
        i4h = min(i_30m // 8, len(df4h) - 1)
        e20 = EMAIndicator(df4h["close"], 20).ema_indicator().iloc[i4h]
        e50 = EMAIndicator(df4h["close"], 50).ema_indicator().iloc[i4h]
        features.append(_safe_div(e20 - e50, e50))
        features.append(_safe(ADXIndicator(df4h["high"], df4h["low"], df4h["close"], 14).adx().iloc[i4h] / 100))
    except Exception:
        features.extend([0.0] * 2)

    # ======== Statistical (5) ========
    try:
        i1h = min(i_30m // 2, len(closes_1h) - 1)
        rets = np.diff(np.log(closes_1h[max(0, i1h - 30): i1h + 1]))
        if len(rets) > 2:
            features.append(float(rets.std()))
            features.append(float(rets.mean()))
            features.append(float(pd.Series(rets).skew()))
            features.append(float(pd.Series(rets).kurtosis()))
            features.append(float(np.corrcoef(rets[:-1], rets[1:])[0, 1]) if len(rets) > 5 else 0.0)
        else:
            features.extend([0.0] * 5)
    except Exception:
        features.extend([0.0] * 5)

    # ======== Time (2) + Price position (1) ========
    features.append((i_30m % 24) / 24.0)
    features.append(((i_30m % 24) % 6) / 6.0)
    if i_30m >= 30:
        recent = closes_30m[i_30m - 29: i_30m + 1]
        features.append((closes_30m[i_30m] - recent.min()) / (recent.max() - recent.min() + 1e-9))
    else:
        features.append(0.5)

    return features


FEATURE_NAMES_V4 = (
    ["rsi_30m", "roc_30m", "williams_r_30m", "stoch_k_30m", "stoch_d_30m",
     "ema_cross_30m", "close_to_sma20_30m", "bb_width_30m", "atr_ratio_30m"]
    + ["rsi_1h", "macd_hist_1h", "macd_1h", "macd_signal_1h", "tsi_1h",
       "adx_1h", "ema_cross_12_26_1h", "ema_cross_26_50_1h",
       "cci_1h", "atr_ratio_1h", "bb_pos_1h", "cmf_1h"]
    + ["ema_cross_20_50_4h", "adx_4h"]
    + ["vol_1h", "ret_mean_1h", "skew_1h", "kurt_1h", "autocorr_1h"]
    + ["hour_norm", "hour_block", "price_pos_30m"]
)


def get_target(close_t0: float, close_t1: float) -> int:
    return 1 if close_t1 > close_t0 else 0
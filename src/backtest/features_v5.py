"""
v5 特征集：100 特征 × 5 个时间框架（5m/15m/30m/1h/4h）。
双目标：30m 方向 + 1h 方向。
主数据源：5m K 线，高周期由此聚合。
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("features.v5")

# ============================================================
# 时间框架聚合（通用版，替代旧版只支持 30m→高周期）
# ============================================================
_INTERVAL_MIN = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
_TZ_CN = timezone(timedelta(hours=8))


def build_aggregated_klines_v2(
    source_klines: list,
    source_interval: str,
    target_interval: str,
) -> list:
    """从短周期 K 线对象列表聚合到长周期。

    要求 source_klines 每个元素有 .open / .high / .low / .close / .volume /
    .quote_volume / .trades_count / .open_time / .close_time / .symbol / .interval 属性。
    """
    if source_interval not in _INTERVAL_MIN or target_interval not in _INTERVAL_MIN:
        raise ValueError(f"不支持的 interval: {source_interval} / {target_interval}")
    src_min = _INTERVAL_MIN[source_interval]
    tgt_min = _INTERVAL_MIN[target_interval]
    if tgt_min <= src_min:
        raise ValueError(f"target interval ({target_interval}) 必须大于 source ({source_interval})")
    ratio = tgt_min // src_min

    out: list = []
    sym = source_klines[0].symbol
    for i in range(0, len(source_klines) - ratio + 1, ratio):
        group = source_klines[i:i + ratio]
        # 用 KlineBar 构造器或简单的 namedtuple 构造
        from src.common.schemas import KlineBar
        out.append(KlineBar(
            symbol=sym,
            interval=target_interval,
            open_time=group[0].open_time,
            close_time=group[-1].close_time,
            open=group[0].open,
            high=max(k.high for k in group),
            low=min(k.low for k in group),
            close=group[-1].close,
            volume=sum(k.volume for k in group),
            quote_volume=sum(k.quote_volume for k in group),
            trades_count=sum(k.trades_count for k in group),
        ))
    return out


# ============================================================
# 从 KlineBar 列表提取 numpy 数组
# ============================================================
def _bars_to_arrays(klines: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    o = np.array([k.open for k in klines], dtype=float)
    c = np.array([k.close for k in klines], dtype=float)
    h = np.array([k.high for k in klines], dtype=float)
    l = np.array([k.low for k in klines], dtype=float)
    v = np.array([k.volume for k in klines], dtype=float)
    return o, c, h, l, v


# ============================================================
# 特征名称（100 个）
# ============================================================
FEATURE_NAMES_V5: List[str] = [
    # ---- 5m (16) ----
    "rsi_5m", "stoch_k_5m", "stoch_d_5m", "willr_5m", "roc_5m",
    "macd_5m", "macd_signal_5m", "macd_hist_5m",
    "ema9_21_5m", "adx_5m", "cci_5m",
    "bb_pos_5m", "bb_width_5m", "atr_ratio_5m",
    "vol_ratio_5m", "price_pos_5m",

    # ---- 15m (16) ----
    "rsi_15m", "stoch_k_15m", "stoch_d_15m", "willr_15m", "roc_15m",
    "macd_15m", "macd_signal_15m", "macd_hist_15m",
    "ema9_21_15m", "adx_15m", "cci_15m",
    "bb_pos_15m", "bb_width_15m", "atr_ratio_15m",
    "vol_ratio_15m", "price_pos_15m",

    # ---- 30m (18) ----
    "rsi_30m", "stoch_k_30m", "stoch_d_30m", "willr_30m", "roc_30m",
    "macd_30m", "macd_signal_30m", "macd_hist_30m",
    "ema9_21_30m", "adx_30m", "cci_30m",
    "bb_pos_30m", "bb_width_30m", "atr_ratio_30m",
    "vol_ratio_30m", "price_pos_30m", "candle_body_30m", "hl_vol_30m",

    # ---- 1h (18) ----
    "rsi_1h", "stoch_k_1h", "stoch_d_1h", "willr_1h", "roc_1h",
    "macd_1h", "macd_signal_1h", "macd_hist_1h",
    "ema9_21_1h", "adx_1h", "cci_1h",
    "bb_pos_1h", "bb_width_1h", "atr_ratio_1h",
    "vol_ratio_1h", "price_pos_1h", "candle_body_1h", "hl_vol_1h",

    # ---- 4h (10) ----
    "rsi_4h", "roc_4h",
    "macd_4h", "macd_signal_4h", "macd_hist_4h",
    "ema20_50_4h", "adx_4h",
    "bb_pos_4h", "bb_width_4h", "price_pos_4h",

    # ---- Statistical (8) ----
    "ret_vol_5m", "ret_vol_1h",
    "ret_skew_1h", "ret_kurt_1h", "autocorr_1h",
    "vol_surge_5m", "vol_surge_30m", "range_expansion",

    # ---- Cross-TF (10) ----
    "rsi_div_5m_1h", "rsi_div_15m_1h",
    "macd_align_5m_30m", "adx_ratio_5m_1h",
    "trend_agree", "bb_squeeze",
    "vol_cascade", "mom_clash",
    "ema_nesting", "multi_tf_macd_dir",

    # ---- Time (4) ----
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
]

_FEAT_COUNT = len(FEATURE_NAMES_V5)  # 100


# ============================================================
# 辅助函数
# ============================================================
def _safe(v, default=0.0):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _safe_div(a, b, default=0.0):
    try:
        return float(a) / float(b) if float(b) != 0 else default
    except Exception:
        return default


# ============================================================
# 指标预计算（每个时间框架一次性向量化计算全序列）
# ============================================================
def _build_tf_indicators(
    o: np.ndarray, c: np.ndarray, h: np.ndarray, l: np.ndarray, v: np.ndarray,
    tf: str,
) -> Dict[str, np.ndarray]:
    """对一个时间框架预计算所有技术指标，返回 {name: array}。"""
    n = len(c)
    if n < 30:
        return {}

    open_s = pd.Series(o, dtype=float)
    close = pd.Series(c, dtype=float)
    high = pd.Series(h, dtype=float)
    low = pd.Series(l, dtype=float)
    volume = pd.Series(v, dtype=float)

    ind: Dict[str, np.ndarray] = {}
    try:
        from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
        from ta.trend import MACD, ADXIndicator, EMAIndicator, CCIIndicator
        from ta.volatility import BollingerBands, AverageTrueRange
        from ta.volume import ChaikinMoneyFlowIndicator

        # RSI (14)
        ind["rsi"] = RSIIndicator(close, 14).rsi().fillna(50).values / 100.0

        # Stochastic (14,3,3)
        so = StochasticOscillator(high, low, close, 14, 3)
        ind["stoch_k"] = so.stoch().fillna(50).values / 100.0
        ind["stoch_d"] = so.stoch_signal().fillna(50).values / 100.0

        # Williams %R (14)
        ind["willr"] = WilliamsRIndicator(high, low, close, 14).williams_r().fillna(-50).values / 100.0

        # ROC (10)
        ind["roc"] = ROCIndicator(close, 10).roc().fillna(0).values / 100.0

        # MACD (12, 26, 9)
        macd_obj = MACD(close, 12, 26, 9)
        macd_val = macd_obj.macd().fillna(0).values
        macd_sig = macd_obj.macd_signal().fillna(0).values
        macd_hist = macd_obj.macd_diff().fillna(0).values
        ind["macd"] = macd_val / close.values
        ind["macd_signal"] = macd_sig / close.values
        ind["macd_hist"] = macd_hist / close.values

        # EMA crosses
        ema9 = EMAIndicator(close, 9).ema_indicator().fillna(close).values
        ema21 = EMAIndicator(close, 21).ema_indicator().fillna(close).values
        ind["ema9_21"] = np.where(ema21 != 0, (ema9 - ema21) / ema21, 0)

        # ADX (14)
        ind["adx"] = ADXIndicator(high, low, close, 14).adx().fillna(20).values / 100.0

        # CCI (20)
        ind["cci"] = CCIIndicator(high, low, close, 20).cci().fillna(0).values / 200.0

        # Bollinger Bands (20, 2)
        bb = BollingerBands(close, 20, 2)
        bb_h = bb.bollinger_hband().fillna(close * 1.05).values
        bb_l = bb.bollinger_lband().fillna(close * 0.95).values
        bb_m = bb.bollinger_mavg().fillna(close).values
        ind["bb_pos"] = np.where((bb_h - bb_l) != 0, (close.values - bb_l) / (bb_h - bb_l), 0.5)
        ind["bb_width"] = np.where(bb_m != 0, (bb_h - bb_l) / bb_m, 0)

        # ATR ratio (14)
        atr = AverageTrueRange(high, low, close, 14).average_true_range().fillna(0).values
        ind["atr_ratio"] = np.where(close.values != 0, atr / close.values, 0)

        # Volume ratio (vs 20-period MA)
        vol_ma = volume.rolling(20, min_periods=1).mean().values
        ind["vol_ratio"] = np.where(vol_ma != 0, volume.values / vol_ma, 1.0)

        # Price position (in 30-bar range)
        if n >= 30:
            rmin = pd.Series(l).rolling(30, min_periods=1).min().values
            rmax = pd.Series(h).rolling(30, min_periods=1).max().values
            ind["price_pos"] = np.where(rmax - rmin > 0, (close.values - rmin) / (rmax - rmin), 0.5)
        else:
            ind["price_pos"] = np.full(n, 0.5)

        # CMF (20)
        ind["cmf"] = ChaikinMoneyFlowIndicator(high, low, close, volume, 20) \
            .chaikin_money_flow().fillna(0).values

        # Candle body ratio: |close - open| / (high - low), default 0.5
        candle_range = h - l
        ind["candle_body"] = np.where(
            candle_range > 0,
            np.abs(close.values - open_s.values) / candle_range,
            0.5,
        )

        # HL volatility
        ind["hl_vol"] = np.where(close.values != 0, (h - l) / close.values, 0)

        # For 4h, compute ema20_50
        ema20 = EMAIndicator(close, 20).ema_indicator().fillna(close).values
        ema50 = EMAIndicator(close, 50).ema_indicator().fillna(close).values
        ind["ema20_50"] = np.where(ema50 != 0, (ema20 - ema50) / ema50, 0)

    except Exception as e:
        log.warning("[features_v5] %s indicator build failed: %s, filling zeros", tf, e)
        # Fill with safe defaults
        for key in ["rsi", "stoch_k", "stoch_d", "willr", "roc", "macd", "macd_signal",
                     "macd_hist", "ema9_21", "adx", "cci", "bb_pos", "bb_width",
                     "atr_ratio", "vol_ratio", "price_pos", "cmf", "candle_body",
                     "hl_vol", "ema20_50"]:
            if key not in ind:
                ind[key] = np.zeros(n)

    return ind


# ============================================================
# 主特征构建（一次性预计算所有时间框架指标）
# ============================================================
def build_all_indicators(
    o5: np.ndarray, c5: np.ndarray, h5: np.ndarray, l5: np.ndarray, v5: np.ndarray,
    o15: np.ndarray, c15: np.ndarray, h15: np.ndarray, l15: np.ndarray, v15: np.ndarray,
    o30: np.ndarray, c30: np.ndarray, h30: np.ndarray, l30: np.ndarray, v30: np.ndarray,
    o1h: np.ndarray, c1h: np.ndarray, h1h: np.ndarray, l1h: np.ndarray, v1h: np.ndarray,
    o4h: np.ndarray, c4h: np.ndarray, h4h: np.ndarray, l4h: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    """预计算所有时间框架的所有指标。

    Returns:
        {"5m": {ind_name: array}, "15m": {...}, "30m": {...}, "1h": {...}, "4h": {...}}
    """
    indicators: Dict[str, Dict[str, np.ndarray]] = {}

    log.info("[features_v5] building 5m indicators (%d bars)...", len(c5))
    indicators["5m"] = _build_tf_indicators(o5, c5, h5, l5, v5, "5m")

    log.info("[features_v5] building 15m indicators (%d bars)...", len(c15))
    indicators["15m"] = _build_tf_indicators(o15, c15, h15, l15, v15, "15m")

    log.info("[features_v5] building 30m indicators (%d bars)...", len(c30))
    indicators["30m"] = _build_tf_indicators(o30, c30, h30, l30, v30, "30m")

    log.info("[features_v5] building 1h indicators (%d bars)...", len(c1h))
    indicators["1h"] = _build_tf_indicators(o1h, c1h, h1h, l1h, v1h, "1h")

    # 4h 的 open 和 volume 用合理代理
    o4h_dummy = o4h  # open of 4h bar
    v4h_dummy = np.ones_like(c4h)  # volume 聚合会丢失，用 ones
    log.info("[features_v5] building 4h indicators (%d bars)...", len(c4h))
    indicators["4h"] = _build_tf_indicators(o4h_dummy, c4h, h4h, l4h, v4h_dummy, "4h")

    return indicators


# ============================================================
# 统计特征（跨时间框架）
# ============================================================
def _compute_statistical_features(
    c5: np.ndarray, c1h: np.ndarray, v5: np.ndarray, v30: np.ndarray,
    h30: np.ndarray, l30: np.ndarray,
    i_5m: int, i_1h: int,
) -> List[float]:
    """计算 8 个统计/微观结构特征。"""
    feats = []

    # ret_vol_5m: 5m returns std (20-period)
    if i_5m >= 20:
        rets_5m = np.diff(np.log(c5[max(0, i_5m - 20): i_5m + 1]))
        feats.append(_safe(np.std(rets_5m)))
    else:
        feats.append(0.0)

    # ret_vol_1h: 1h returns std (20-period)
    if i_1h >= 20:
        rets_1h = np.diff(np.log(c1h[max(0, i_1h - 20): i_1h + 1]))
        feats.append(_safe(np.std(rets_1h)))
    else:
        feats.append(0.0)

    # ret_skew_1h
    if i_1h >= 20:
        rets_1h = np.diff(np.log(c1h[max(0, i_1h - 20): i_1h + 1]))
        if len(rets_1h) > 2:
            feats.append(_safe(float(pd.Series(rets_1h).skew())))
        else:
            feats.append(0.0)
    else:
        feats.append(0.0)

    # ret_kurt_1h
    if i_1h >= 20:
        rets_1h = np.diff(np.log(c1h[max(0, i_1h - 20): i_1h + 1]))
        if len(rets_1h) > 3:
            feats.append(_safe(float(pd.Series(rets_1h).kurtosis())))
        else:
            feats.append(0.0)
    else:
        feats.append(0.0)

    # autocorr_1h
    if i_1h >= 20:
        rets_1h = np.diff(np.log(c1h[max(0, i_1h - 20): i_1h + 1]))
        if len(rets_1h) > 5:
            try:
                ac = float(np.corrcoef(rets_1h[:-1], rets_1h[1:])[0, 1])
                feats.append(_safe(ac))
            except Exception:
                feats.append(0.0)
        else:
            feats.append(0.0)
    else:
        feats.append(0.0)

    # vol_surge_5m: current volume / median volume (20-period)
    if i_5m >= 20:
        vol_med = np.median(v5[max(0, i_5m - 20): i_5m])
        feats.append(_safe_div(v5[i_5m], vol_med, 1.0))
    else:
        feats.append(1.0)

    # vol_surge_30m
    i_30m = min(i_5m // 6, len(v30) - 1)
    if i_30m >= 20:
        vol_med = np.median(v30[max(0, i_30m - 20): i_30m])
        feats.append(_safe_div(v30[i_30m], vol_med, 1.0))
    else:
        feats.append(1.0)

    # range_expansion: current 30m range / avg 30m range (20-period)
    i_30m = min(i_5m // 6, len(h30) - 1) if len(h30) > 0 else 0
    # h30, l30 are passed as h30_arr, l30_arr
    if i_30m >= 20 and len(h30) > 20:
        ranges = h30[max(0, i_30m - 20): i_30m] - l30[max(0, i_30m - 20): i_30m]
        avg_range = np.mean(ranges) if len(ranges) > 0 else 1.0
        cur_range = h30[i_30m] - l30[i_30m] if i_30m < len(h30) else 0
        feats.append(_safe_div(cur_range, avg_range, 1.0))
    else:
        feats.append(1.0)

    return feats


# ============================================================
# 跨时间框架特征
# ============================================================
def _compute_cross_tf_features(
    inds: Dict[str, Dict[str, np.ndarray]],
    i_5m: int,
) -> List[float]:
    """计算 10 个跨时间框架特征。"""
    feats = []

    i_15m = min(i_5m // 3, len(inds["15m"].get("rsi", np.zeros(1))) - 1)
    i_30m = min(i_5m // 6, len(inds["30m"].get("rsi", np.zeros(1))) - 1)
    i_1h = min(i_5m // 12, len(inds["1h"].get("rsi", np.zeros(1))) - 1)

    try:
        # rsi_div_5m_1h: 短周期 RSI - 长周期 RSI
        rsi5 = inds["5m"]["rsi"][i_5m]
        rsi1h = inds["1h"]["rsi"][i_1h]
        feats.append(_safe(rsi5 - rsi1h))

        # rsi_div_15m_1h
        rsi15 = inds["15m"]["rsi"][i_15m]
        feats.append(_safe(rsi15 - rsi1h))

        # macd_align_5m_30m: MACD 柱方向是否一致
        macd5_hist = inds["5m"]["macd_hist"][i_5m]
        macd30_hist = inds["30m"]["macd_hist"][i_30m]
        feats.append(1.0 if macd5_hist * macd30_hist > 0 else -1.0)

        # adx_ratio_5m_1h
        adx5 = inds["5m"]["adx"][i_5m]
        adx1h = inds["1h"]["adx"][i_1h]
        feats.append(_safe_div(adx5, adx1h + 0.01, 1.0))

        # trend_agree: 各周期 EMA 交叉方向一致度（-1 ~ +1）
        ema_signs = []
        for tf_key in ["5m", "15m", "30m", "1h"]:
            idx_map = {"5m": i_5m, "15m": i_15m, "30m": i_30m, "1h": i_1h}
            idx = min(idx_map[tf_key], len(inds[tf_key].get("ema9_21", np.zeros(1))) - 1)
            ema_val = inds[tf_key]["ema9_21"][idx]
            ema_signs.append(1 if ema_val > 0 else (-1 if ema_val < 0 else 0))
        feats.append(sum(ema_signs) / max(1, len(ema_signs)))

        # bb_squeeze: 多周期 BB 宽度平均（越小=即将突破）
        bb_widths = []
        for tf_key, idx in [("5m", i_5m), ("15m", i_15m), ("30m", i_30m), ("1h", i_1h)]:
            idx = min(idx, len(inds[tf_key].get("bb_width", np.zeros(1))) - 1)
            bb_widths.append(inds[tf_key]["bb_width"][idx])
        feats.append(_safe(np.mean(bb_widths)))

        # vol_cascade: 5m/30m 量比（短周期放量 = 异动）
        vol5_ratio = inds["5m"]["vol_ratio"][i_5m]
        vol30_ratio = inds["30m"]["vol_ratio"][i_30m]
        feats.append(_safe_div(vol5_ratio, vol30_ratio + 0.01, 1.0))

        # mom_clash: 多周期动量方向冲突度
        roc_vals = []
        for tf_key, idx in [("5m", i_5m), ("15m", i_15m), ("30m", i_30m), ("1h", i_1h)]:
            idx = min(idx, len(inds[tf_key].get("roc", np.zeros(1))) - 1)
            roc_vals.append(inds[tf_key]["roc"][idx])
        pos_count = sum(1 for r in roc_vals if r > 0)
        neg_count = sum(1 for r in roc_vals if r < 0)
        feats.append((pos_count - neg_count) / max(1, len(roc_vals)))

        # ema_nesting: 检查 EMA 是否正常嵌套（ema9 > ema21 > ema50）
        # 用各周期的 ema9_21 正负 + 4h 的 ema20_50 来判断
        i_4h = min(i_5m // 48, len(inds["4h"].get("ema20_50", np.zeros(1))) - 1)
        ema4h_dir = 1 if inds["4h"]["ema20_50"][i_4h] > 0.001 else (-1 if inds["4h"]["ema20_50"][i_4h] < -0.001 else 0)
        ema1h_dir = 1 if inds["1h"]["ema9_21"][i_1h] > 0.001 else (-1 if inds["1h"]["ema9_21"][i_1h] < -0.001 else 0)
        feats.append(1.0 if ema1h_dir == ema4h_dir else -1.0)

        # multi_tf_macd_dir: 统计多少周期的 MACD 柱 > 0
        macd_count = 0
        for tf_key, idx in [("5m", i_5m), ("15m", i_15m), ("30m", i_30m), ("1h", i_1h), ("4h", i_4h)]:
            idx = min(idx, len(inds[tf_key].get("macd_hist", np.zeros(1))) - 1)
            if inds[tf_key]["macd_hist"][idx] > 0:
                macd_count += 1
        feats.append(macd_count / 5.0)
    except Exception as e:
        log.debug("[features_v5] cross-tf feature error: %s", e)
        feats.extend([0.0] * (10 - len(feats)))

    # Pad to exactly 10
    while len(feats) < 10:
        feats.append(0.0)
    return feats[:10]


# ============================================================
# 时间特征
# ============================================================
def _compute_time_features(open_time_5m: Optional[datetime] = None) -> List[float]:
    """计算 4 个时间特征。"""
    if open_time_5m is None:
        return [0.0, 1.0, 0.0, 1.0]  # defaults (00:00 Monday)

    hour = open_time_5m.hour + open_time_5m.minute / 60.0
    weekday = open_time_5m.weekday()  # 0=Mon, 6=Sun

    return [
        float(np.sin(2 * np.pi * hour / 24)),
        float(np.cos(2 * np.pi * hour / 24)),
        float(np.sin(2 * np.pi * weekday / 7)),
        float(np.cos(2 * np.pi * weekday / 7)),
    ]


# ============================================================
# 主特征计算（从预计算指标中提取单一样本）
# ============================================================
def compute_features_v5(
    inds: Dict[str, Dict[str, np.ndarray]],
    c5: np.ndarray, v5: np.ndarray,
    c1h: np.ndarray, v30: np.ndarray,
    h30: np.ndarray, l30: np.ndarray,
    i_5m: int,
    open_time_5m: Optional[datetime] = None,
) -> List[float]:
    """从预计算指标中提取第 i_5m 个样本的 100 个特征。

    Args:
        inds: build_all_indicators() 的输出
        c5, v5, c1h, v30: 原始价格/量数组（用于统计特征）
        i_5m: 当前 5m 索引
        open_time_5m: 当前 5m K 线的开盘时间（用于时间特征）

    Returns:
        长度 100 的 float 列表
    """
    i_15m = min(i_5m // 3, len(inds["15m"].get("rsi", np.zeros(1))) - 1)
    i_30m = min(i_5m // 6, len(inds["30m"].get("rsi", np.zeros(1))) - 1)
    i_1h = min(i_5m // 12, len(inds["1h"].get("rsi", np.zeros(1))) - 1)
    i_4h = min(i_5m // 48, len(inds["4h"].get("rsi", np.zeros(1))) - 1)

    def _get(tf: str, key: str, idx: int) -> float:
        arr = inds.get(tf, {}).get(key)
        if arr is None or idx < 0 or idx >= len(arr):
            return 0.0
        return _safe(arr[idx])

    features: List[float] = []

    # ======== 5m (16 features) ========
    features.append(_get("5m", "rsi", i_5m))
    features.append(_get("5m", "stoch_k", i_5m))
    features.append(_get("5m", "stoch_d", i_5m))
    features.append(_get("5m", "willr", i_5m))
    features.append(_get("5m", "roc", i_5m))
    features.append(_get("5m", "macd", i_5m))
    features.append(_get("5m", "macd_signal", i_5m))
    features.append(_get("5m", "macd_hist", i_5m))
    features.append(_get("5m", "ema9_21", i_5m))
    features.append(_get("5m", "adx", i_5m))
    features.append(_get("5m", "cci", i_5m))
    features.append(_get("5m", "bb_pos", i_5m))
    features.append(_get("5m", "bb_width", i_5m))
    features.append(_get("5m", "atr_ratio", i_5m))
    features.append(_get("5m", "vol_ratio", i_5m))
    features.append(_get("5m", "price_pos", i_5m))

    # ======== 15m (16 features) ========
    features.append(_get("15m", "rsi", i_15m))
    features.append(_get("15m", "stoch_k", i_15m))
    features.append(_get("15m", "stoch_d", i_15m))
    features.append(_get("15m", "willr", i_15m))
    features.append(_get("15m", "roc", i_15m))
    features.append(_get("15m", "macd", i_15m))
    features.append(_get("15m", "macd_signal", i_15m))
    features.append(_get("15m", "macd_hist", i_15m))
    features.append(_get("15m", "ema9_21", i_15m))
    features.append(_get("15m", "adx", i_15m))
    features.append(_get("15m", "cci", i_15m))
    features.append(_get("15m", "bb_pos", i_15m))
    features.append(_get("15m", "bb_width", i_15m))
    features.append(_get("15m", "atr_ratio", i_15m))
    features.append(_get("15m", "vol_ratio", i_15m))
    features.append(_get("15m", "price_pos", i_15m))

    # ======== 30m (18 features) ========
    features.append(_get("30m", "rsi", i_30m))
    features.append(_get("30m", "stoch_k", i_30m))
    features.append(_get("30m", "stoch_d", i_30m))
    features.append(_get("30m", "willr", i_30m))
    features.append(_get("30m", "roc", i_30m))
    features.append(_get("30m", "macd", i_30m))
    features.append(_get("30m", "macd_signal", i_30m))
    features.append(_get("30m", "macd_hist", i_30m))
    features.append(_get("30m", "ema9_21", i_30m))
    features.append(_get("30m", "adx", i_30m))
    features.append(_get("30m", "cci", i_30m))
    features.append(_get("30m", "bb_pos", i_30m))
    features.append(_get("30m", "bb_width", i_30m))
    features.append(_get("30m", "atr_ratio", i_30m))
    features.append(_get("30m", "vol_ratio", i_30m))
    features.append(_get("30m", "price_pos", i_30m))
    features.append(_get("30m", "candle_body", i_30m))
    features.append(_get("30m", "hl_vol", i_30m))

    # ======== 1h (18 features) ========
    features.append(_get("1h", "rsi", i_1h))
    features.append(_get("1h", "stoch_k", i_1h))
    features.append(_get("1h", "stoch_d", i_1h))
    features.append(_get("1h", "willr", i_1h))
    features.append(_get("1h", "roc", i_1h))
    features.append(_get("1h", "macd", i_1h))
    features.append(_get("1h", "macd_signal", i_1h))
    features.append(_get("1h", "macd_hist", i_1h))
    features.append(_get("1h", "ema9_21", i_1h))
    features.append(_get("1h", "adx", i_1h))
    features.append(_get("1h", "cci", i_1h))
    features.append(_get("1h", "bb_pos", i_1h))
    features.append(_get("1h", "bb_width", i_1h))
    features.append(_get("1h", "atr_ratio", i_1h))
    features.append(_get("1h", "vol_ratio", i_1h))
    features.append(_get("1h", "price_pos", i_1h))
    features.append(_get("1h", "candle_body", i_1h))
    features.append(_get("1h", "hl_vol", i_1h))

    # ======== 4h (10 features) ========
    features.append(_get("4h", "rsi", i_4h))
    features.append(_get("4h", "roc", i_4h))
    features.append(_get("4h", "macd", i_4h))
    features.append(_get("4h", "macd_signal", i_4h))
    features.append(_get("4h", "macd_hist", i_4h))
    features.append(_get("4h", "ema20_50", i_4h))
    features.append(_get("4h", "adx", i_4h))
    features.append(_get("4h", "bb_pos", i_4h))
    features.append(_get("4h", "bb_width", i_4h))
    features.append(_get("4h", "price_pos", i_4h))

    # ======== Statistical (8) ========
    features.extend(_compute_statistical_features(c5, c1h, v5, v30, h30, l30, i_5m, i_1h))

    # ======== Cross-TF (10) ========
    features.extend(_compute_cross_tf_features(inds, i_5m))

    # ======== Time (4) ========
    features.extend(_compute_time_features(open_time_5m))

    # Sanity check
    if len(features) != _FEAT_COUNT:
        log.warning("[features_v5] feature count mismatch: got %d, expected %d",
                     len(features), _FEAT_COUNT)
        while len(features) < _FEAT_COUNT:
            features.append(0.0)

    return features[:_FEAT_COUNT]


# ============================================================
# 双目标
# ============================================================
def get_target_30m(close_now: float, close_6_bars_later: float) -> int:
    """30m 目标：6 根 5m K 线后涨(1)还是跌(0)。"""
    return 1 if close_6_bars_later > close_now else 0


def get_target_1h(close_now: float, close_12_bars_later: float) -> int:
    """1h 目标：12 根 5m K 线后涨(1)还是跌(0)。"""
    return 1 if close_12_bars_later > close_now else 0

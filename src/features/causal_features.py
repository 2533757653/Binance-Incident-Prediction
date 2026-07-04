"""
因果特征引擎（无前视泄露 / 训练==实盘）。

修复要点（对比旧 features_v5）：
1. 高周期 K 线按【真实时钟】对齐聚合（epoch 分桶），与交易所真实 K 线一致。
2. 任一 5m 时刻 i，只引用【上一根已彻底收盘】的高周期 K 线
   （causal_idx = 当前桶之前最后一根完整桶），绝不碰"当前未走完的 K 线"。
3. 训练矩阵与实盘单点用【同一套装配逻辑】，从根本上消除 train/serve 偏移。

特征顺序、名称完全沿用 FEATURE_NAMES_V5（100 个），便于无缝替换。
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest.features_v5 import (
    FEATURE_NAMES_V5, _safe, _safe_div, build_all_indicators,
)

_TF_MIN = {"15m": 15, "30m": 30, "1h": 60, "4h": 240}
_FEAT_COUNT = len(FEATURE_NAMES_V5)


# ════════════════════════════════════════════════════════════════
# 时钟对齐聚合
# ════════════════════════════════════════════════════════════════
def _epoch_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _aligned_tf(open5_ms, o5, h5, l5, c5, v5, tf_min):
    """按真实时钟把 5m 聚合到高周期；返回 (o,h,l,c,v, buckets)。
    buckets[k] = 该高周期 K 线所属的 epoch 桶号（用于因果索引）。"""
    width = tf_min * 60_000
    bucket = open5_ms // width
    uniq, starts = np.unique(bucket, return_index=True)  # 升序，starts 为各桶首位
    ends = np.append(starts[1:], len(bucket)) - 1
    o = o5[starts]
    c = c5[ends]
    h = np.maximum.reduceat(h5, starts)
    l = np.minimum.reduceat(l5, starts)
    v = np.add.reduceat(v5, starts)
    return o, h, l, c, v, uniq


def _causal_idx(open5_ms, tf_min, tf_buckets):
    """对每个 5m bar，返回"上一根已收盘高周期 K 线"在聚合数组中的下标。
    = 当前桶号在 tf_buckets 中 left 插入位 - 1（即严格早于当前桶的最后一根完整桶）。"""
    width = tf_min * 60_000
    cur_bucket = open5_ms // width
    idx = np.searchsorted(tf_buckets, cur_bucket, side="left") - 1
    return np.clip(idx, 0, len(tf_buckets) - 1)


# ════════════════════════════════════════════════════════════════
# 单样本装配（训练 & 实盘共用）
# ════════════════════════════════════════════════════════════════
def _assemble(inds, c5, v5, tf, cmap, i, open_time):
    """组装第 i 个 5m bar 的 100 维特征（全部使用因果下标）。
    tf:   {"15m":(o,c,h,l,v),...} 聚合数组
    cmap: {"15m":idxarr,...} 各高周期的因果下标数组
    """
    i15, i30, i1h, i4h = (int(cmap["15m"][i]), int(cmap["30m"][i]),
                          int(cmap["1h"][i]), int(cmap["4h"][i]))

    def g(tf_key, key, idx):
        arr = inds.get(tf_key, {}).get(key)
        if arr is None or idx < 0 or idx >= len(arr):
            return 0.0
        return _safe(arr[idx])

    f: List[float] = []
    # 5m (16)
    for k in ("rsi", "stoch_k", "stoch_d", "willr", "roc", "macd", "macd_signal",
              "macd_hist", "ema9_21", "adx", "cci", "bb_pos", "bb_width",
              "atr_ratio", "vol_ratio", "price_pos"):
        f.append(g("5m", k, i))
    # 15m (16)
    for k in ("rsi", "stoch_k", "stoch_d", "willr", "roc", "macd", "macd_signal",
              "macd_hist", "ema9_21", "adx", "cci", "bb_pos", "bb_width",
              "atr_ratio", "vol_ratio", "price_pos"):
        f.append(g("15m", k, i15))
    # 30m (18)
    for k in ("rsi", "stoch_k", "stoch_d", "willr", "roc", "macd", "macd_signal",
              "macd_hist", "ema9_21", "adx", "cci", "bb_pos", "bb_width",
              "atr_ratio", "vol_ratio", "price_pos", "candle_body", "hl_vol"):
        f.append(g("30m", k, i30))
    # 1h (18)
    for k in ("rsi", "stoch_k", "stoch_d", "willr", "roc", "macd", "macd_signal",
              "macd_hist", "ema9_21", "adx", "cci", "bb_pos", "bb_width",
              "atr_ratio", "vol_ratio", "price_pos", "candle_body", "hl_vol"):
        f.append(g("1h", k, i1h))
    # 4h (10)
    for k in ("rsi", "roc", "macd", "macd_signal", "macd_hist",
              "ema20_50", "adx", "bb_pos", "bb_width", "price_pos"):
        f.append(g("4h", k, i4h))

    # 统计 (8) —— 全部用因果下标
    c1h = tf["1h"][1]
    h30, l30, v30 = tf["30m"][2], tf["30m"][3], tf["30m"][4]
    f.extend(_stat_feats(c5, v5, c1h, v30, h30, l30, i, i1h, i30))
    # 跨周期 (10)
    f.extend(_cross_feats(inds, i, i15, i30, i1h, i4h))
    # 时间 (4)
    f.extend(_time_feats(open_time))

    if len(f) != _FEAT_COUNT:
        f = (f + [0.0] * _FEAT_COUNT)[:_FEAT_COUNT]
    return f


def _stat_feats(c5, v5, c1h, v30, h30, l30, i5, i1h, i30):
    out = []
    if i5 >= 20:
        r = np.diff(np.log(c5[max(0, i5 - 20): i5 + 1])); out.append(_safe(np.std(r)))
    else:
        out.append(0.0)
    if i1h >= 20:
        r1 = np.diff(np.log(c1h[max(0, i1h - 20): i1h + 1]))
        out.append(_safe(np.std(r1)))
        out.append(_safe(float(pd.Series(r1).skew())) if len(r1) > 2 else 0.0)
        out.append(_safe(float(pd.Series(r1).kurtosis())) if len(r1) > 3 else 0.0)
        if len(r1) > 5:
            try:
                out.append(_safe(float(np.corrcoef(r1[:-1], r1[1:])[0, 1])))
            except Exception:
                out.append(0.0)
        else:
            out.append(0.0)
    else:
        out.extend([0.0, 0.0, 0.0, 0.0])
    # vol_surge_5m
    out.append(_safe_div(v5[i5], np.median(v5[max(0, i5 - 20): i5]), 1.0) if i5 >= 20 else 1.0)
    # vol_surge_30m（因果 i30）
    out.append(_safe_div(v30[i30], np.median(v30[max(0, i30 - 20): i30]), 1.0) if i30 >= 20 else 1.0)
    # range_expansion（因果 i30）
    if i30 >= 20 and len(h30) > 20:
        rng = h30[max(0, i30 - 20): i30] - l30[max(0, i30 - 20): i30]
        out.append(_safe_div(h30[i30] - l30[i30], np.mean(rng) if len(rng) else 1.0, 1.0))
    else:
        out.append(1.0)
    return out


def _cross_feats(inds, i5, i15, i30, i1h, i4h):
    f = []

    def gv(tf, key, idx):
        a = inds.get(tf, {}).get(key)
        return _safe(a[idx]) if (a is not None and 0 <= idx < len(a)) else 0.0

    rsi5, rsi15, rsi1h = gv("5m", "rsi", i5), gv("15m", "rsi", i15), gv("1h", "rsi", i1h)
    f.append(rsi5 - rsi1h)
    f.append(rsi15 - rsi1h)
    f.append(1.0 if gv("5m", "macd_hist", i5) * gv("30m", "macd_hist", i30) > 0 else -1.0)
    f.append(_safe_div(gv("5m", "adx", i5), gv("1h", "adx", i1h) + 0.01, 1.0))
    signs = []
    for tf, idx in (("5m", i5), ("15m", i15), ("30m", i30), ("1h", i1h)):
        ev = gv(tf, "ema9_21", idx)
        signs.append(1 if ev > 0 else (-1 if ev < 0 else 0))
    f.append(sum(signs) / max(1, len(signs)))
    f.append(_safe(np.mean([gv(tf, "bb_width", idx) for tf, idx in
                            (("5m", i5), ("15m", i15), ("30m", i30), ("1h", i1h))])))
    f.append(_safe_div(gv("5m", "vol_ratio", i5), gv("30m", "vol_ratio", i30) + 0.01, 1.0))
    rocs = [gv(tf, "roc", idx) for tf, idx in (("5m", i5), ("15m", i15), ("30m", i30), ("1h", i1h))]
    f.append((sum(1 for r in rocs if r > 0) - sum(1 for r in rocs if r < 0)) / max(1, len(rocs)))
    e4 = gv("4h", "ema20_50", i4h); e1 = gv("1h", "ema9_21", i1h)
    d4 = 1 if e4 > 0.001 else (-1 if e4 < -0.001 else 0)
    d1 = 1 if e1 > 0.001 else (-1 if e1 < -0.001 else 0)
    f.append(1.0 if d1 == d4 else -1.0)
    mc = sum(1 for tf, idx in (("5m", i5), ("15m", i15), ("30m", i30), ("1h", i1h), ("4h", i4h))
             if gv(tf, "macd_hist", idx) > 0)
    f.append(mc / 5.0)
    return f[:10]


def _time_feats(ot: Optional[datetime]):
    if ot is None:
        return [0.0, 1.0, 0.0, 1.0]
    hour = ot.hour + ot.minute / 60.0
    wd = ot.weekday()
    return [float(np.sin(2 * np.pi * hour / 24)), float(np.cos(2 * np.pi * hour / 24)),
            float(np.sin(2 * np.pi * wd / 7)), float(np.cos(2 * np.pi * wd / 7))]


# ════════════════════════════════════════════════════════════════
# 公共入口
# ════════════════════════════════════════════════════════════════
def _prep(k5):
    o5 = np.array([b.open for b in k5], float)
    h5 = np.array([b.high for b in k5], float)
    l5 = np.array([b.low for b in k5], float)
    c5 = np.array([b.close for b in k5], float)
    v5 = np.array([b.volume for b in k5], float)
    open5_ms = np.array([_epoch_ms(b.open_time) for b in k5], dtype=np.int64)
    tf, cmap = {}, {}
    for name, m in _TF_MIN.items():
        o, h, l, c, v, buckets = _aligned_tf(open5_ms, o5, h5, l5, c5, v5, m)
        tf[name] = (o, c, h, l, v)
        cmap[name] = _causal_idx(open5_ms, m, buckets)
    inds = build_all_indicators(
        o5, c5, h5, l5, v5,
        *tf["15m"], *tf["30m"], *tf["1h"],
        tf["4h"][0], tf["4h"][1], tf["4h"][2], tf["4h"][3])
    return o5, c5, v5, tf, cmap, inds


def compute_matrix(k5, warmup_5m: int = 60, min_lookahead_5m: int = 12):
    """构建训练/回测特征矩阵。返回 (X, y_30m, y_1h, names, open_times)。"""
    o5, c5, v5, tf, cmap, inds = _prep(k5)
    n = len(k5)
    rng = range(warmup_5m, n - min_lookahead_5m)
    X = np.zeros((len(rng), _FEAT_COUNT), float)
    y30 = np.zeros(len(rng), int)
    y1h = np.zeros(len(rng), int)
    times = []
    for j, i in enumerate(rng):
        X[j] = _assemble(inds, c5, v5, tf, cmap, i, k5[i].open_time)
        y30[j] = 1 if c5[i + 6] > c5[i] else 0
        y1h[j] = 1 if c5[i + 12] > c5[i] else 0
        times.append(k5[i].open_time)
    return X, y30, y1h, list(FEATURE_NAMES_V5), times


def compute_live(k5):
    """实盘：返回最新【已收盘】5m bar 的特征向量（与训练同一装配）。"""
    o5, c5, v5, tf, cmap, inds = _prep(k5)
    i = len(k5) - 1  # 调用方只传已收盘 bar
    x = _assemble(inds, c5, v5, tf, cmap, i, k5[i].open_time)
    return np.array(x, float), list(FEATURE_NAMES_V5)

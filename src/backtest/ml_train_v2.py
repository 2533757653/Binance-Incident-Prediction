"""
XGBoost 训练 v2：90 天 + 30m/1h 双窗口 + 多特征。

特征（共 10 个）：
  - 30m K 线因子（5 个）：
    * ret_30m    (30 分钟对数收益率)
    * bb_pct_30m (20 周期布林带 %b，用 30m 序列)
    * vol_30m    (1h 窗口波动率)
    * mom_30m    (3 根 30m K 线动量)
    * range_30m  (最近 30m high-low / close)
  - 1h K 线因子（5 个）：
    * ret_1h
    * bb_pct_1h
    * trend_24h  (24h 趋势)
    * mom_1h     (3 根 1h 动量)
    * vol_1h     (1h K 线波动率)

目标（事件窗口 = 30m）：
  y=1 if YES 赢（close ≥ strike），0 if YES 输

切分：前 75% 训练，后 25% 验证（≈ 22.5 天测试）
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.backtest.real_data import (
    build_aggregated_klines,
    build_events_from_klines,
    fetch_klines_paginated,
)

log = logging.getLogger(__name__)


FEATURE_NAMES = [
    "ret_30m", "bb_pct_30m", "vol_30m", "mom_30m", "range_30m",
    "ret_1h", "bb_pct_1h", "trend_24h", "mom_1h", "vol_1h",
]


def compute_features(
    closes_30m: np.ndarray,
    highs_30m: np.ndarray,
    lows_30m: np.ndarray,
    closes_1h: np.ndarray,
    i_30m: int,
) -> List[float]:
    """计算 10 个特征。"""
    # 30m 特征
    ret_30m = float(np.log(closes_30m[i_30m] / closes_30m[i_30m - 1])) if i_30m >= 1 else 0.0
    # bb 20 周期（用 30m 序列）
    if i_30m >= 19:
        w = closes_30m[i_30m - 19: i_30m + 1]
        mid = float(np.mean(w))
        sd = float(np.std(w, ddof=0))
        if sd > 0:
            bb_pct_30m = float((closes_30m[i_30m] - (mid - 2 * sd)) / (4 * sd))
        else:
            bb_pct_30m = 0.5
    else:
        bb_pct_30m = 0.5
    # vol_30m：最近 2 根 30m 的对数收益率 std
    if i_30m >= 2:
        rets = np.diff(np.log(closes_30m[max(0, i_30m - 2): i_30m + 1]))
        vol_30m = float(np.std(rets)) if len(rets) > 1 else 0.0
    else:
        vol_30m = 0.0
    # mom_30m：3 根 30m 累计
    mom_30m = float(np.log(closes_30m[i_30m] / closes_30m[i_30m - 3])) if i_30m >= 3 else 0.0
    # range_30m
    if i_30m >= 0:
        range_30m = float((highs_30m[i_30m] - lows_30m[i_30m]) / closes_30m[i_30m])
    else:
        range_30m = 0.0

    # 1h 特征（用 i_30m 对应的 1h 索引）
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

    return [
        ret_30m, bb_pct_30m, vol_30m, mom_30m, range_30m,
        ret_1h, bb_pct_1h, trend_24h, mom_1h, vol_1h,
    ]


def build_training_dataset(
    symbol: str = "BTCUSDT",
    days: int = 90,
    event_minutes: int = 30,
    strike_offset_pct: float = 0.02,
    split_ratio: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    返回 X_train, X_test, y_train, y_test, feature_names。
    """
    log.info("[ml_train] %s: 拉 %d 天 30m K 线 ...", symbol, days)
    k30 = fetch_klines_paginated(symbol, "30m", days=days)
    log.info("[ml_train] %s: 30m K 线 %d 根", symbol, len(k30))
    k1h = build_aggregated_klines(k30, "1h")
    log.info("[ml_train] %s: 1h K 线 %d 根", symbol, len(k1h))

    closes_30m = np.array([k.close for k in k30], dtype=float)
    highs_30m = np.array([k.high for k in k30], dtype=float)
    lows_30m = np.array([k.low for k in k30], dtype=float)
    closes_1h = np.array([k.close for k in k1h], dtype=float)

    # 构造事件：每根 30m K 线 = 一个 30m 事件
    import random
    rng = random.Random(hash(symbol + str(event_minutes)) & 0xFFFFFFFF)

    X_all, y_all, indices = [], [], []
    # 跳过前 30 根（数据不够）和最后 1 根（无 close 后结果）
    for i in range(30, len(k30) - 1):
        try:
            features = compute_features(closes_30m, highs_30m, lows_30m, closes_1h, i)
        except Exception as e:
            log.debug("[ml_train] features fail at i=%d: %s", i, e)
            continue
        # 目标：close vs strike（随机 strike 在 ±2%）
        offset = (rng.random() * 2 - 1) * strike_offset_pct
        strike = k30[i].open * (1.0 + offset)
        yes_won = k30[i + 1].close >= strike  # 30m 后 close
        X_all.append(features)
        y_all.append(1 if yes_won else 0)
        indices.append(i)

    X_all = np.array(X_all, dtype=float)
    y_all = np.array(y_all, dtype=int)

    # 切分
    split_idx = int(len(X_all) * split_ratio)
    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]
    log.info("[ml_train] train: %d | test: %d | yes_ratio: %.3f",
             len(X_train), len(X_test), y_all.mean())
    return X_train, X_test, y_train, y_test, FEATURE_NAMES


def train_xgboost(
    X_train, y_train, X_test, y_test,
    *,
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    random_state: int = 42,
):
    """训练 XGBoost。"""
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=2,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    train_acc = (model.predict(X_train) == y_train).mean()
    test_acc = (model.predict(X_test) == y_test).mean()
    return model, train_acc, test_acc


def save_model(model, feature_names: List[str], path: str | Path) -> Path:
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": feature_names}, path)
    return path
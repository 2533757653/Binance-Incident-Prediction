"""
XGBoost v3 训练：90 天 + 15 个 K 线特征。

新特征（5 个）：
- volume_ratio_30m   （当前成交量 / 20 周期均量）
- atr_ratio_1h       （当前 ATR / 20 周期均 ATR，反映波动率突变）
- bb_width_1h        （布林带宽度，盘整/趋势状态）
- macd_hist_1h       （MACD 柱状图，动量+趋势）
- adx_1h             （趋势强度，0-100）
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.backtest.features_v3 import compute_features_v3, FEATURE_NAMES_V3
from src.backtest.real_data import build_aggregated_klines, fetch_klines_paginated

log = logging.getLogger("ml.v3")


def build_training_dataset_v3(
    symbol: str = "BTCUSDT",
    days: int = 90,
    split_ratio: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    用 v3 特征（15 个）构造训练/测试集。

    返回 X_train, X_test, y_train, y_test, feature_names。
    """
    log.info("[v3] %s: 拉 %d 天 30m K 线 ...", symbol, days)
    k30 = fetch_klines_paginated(symbol, "30m", days=days)
    log.info("[v3] %s: 30m K 线 %d 根", symbol, len(k30))
    k1h = build_aggregated_klines(k30, "1h")
    log.info("[v3] %s: 1h K 线 %d 根", symbol, len(k1h))

    closes_30m = np.array([k.close for k in k30], dtype=float)
    highs_30m = np.array([k.high for k in k30], dtype=float)
    lows_30m = np.array([k.low for k in k30], dtype=float)
    volumes_30m = np.array([k.volume for k in k30], dtype=float)

    closes_1h = np.array([k.close for k in k1h], dtype=float)
    highs_1h = np.array([k.high for k in k1h], dtype=float)
    lows_1h = np.array([k.low for k in k1h], dtype=float)
    volumes_1h = np.array([k.volume for k in k1h], dtype=float)

    import random
    rng = random.Random(hash(symbol + "v3") & 0xFFFFFFFF)

    X_all, y_all = [], []
    # 跳过前 50 根（数据不够 ADX 等指标）和最后 1 根
    for i in range(50, len(k30) - 1):
        try:
            features = compute_features_v3(
                closes_30m, highs_30m, lows_30m, volumes_30m,
                closes_1h, highs_1h, lows_1h, volumes_1h,
                i_30m=i,
            )
        except Exception as e:
            log.debug("[v3] features fail at i=%d: %s", i, e)
            continue
        # 目标：close vs strike（随机 strike 在 ±2%）
        offset = (rng.random() * 2 - 1) * 0.02
        strike = k30[i].open * (1.0 + offset)
        yes_won = k30[i + 1].close >= strike
        X_all.append(features)
        y_all.append(1 if yes_won else 0)

    X_all = np.array(X_all, dtype=float)
    y_all = np.array(y_all, dtype=int)

    split_idx = int(len(X_all) * split_ratio)
    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]
    log.info("[v3] train: %d | test: %d | yes_ratio: %.3f",
             len(X_train), len(X_test), y_all.mean())
    return X_train, X_test, y_train, y_test, FEATURE_NAMES_V3


def train_xgboost(X_train, y_train, X_test, y_test, *,
                    n_estimators: int = 300, max_depth: int = 4,
                    learning_rate: float = 0.05, random_state: int = 42):
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="logloss",
        random_state=random_state, n_jobs=2,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    train_acc = (model.predict(X_train) == y_train).mean()
    test_acc = (model.predict(X_test) == y_test).mean()
    return model, train_acc, test_acc


def save_model(model, feature_names, path):
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": feature_names}, path)
    return path
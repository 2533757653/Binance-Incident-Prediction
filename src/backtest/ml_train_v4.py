"""
XGBoost v4 训练：事件合约二元方向预测（涨/跌）+ 39 个因子。

目标：1 if close_t+30m > close_t else 0（不看幅度）
因子：30m (11) + 1h (16) + 4h (2) + 统计 (5) + 时间 (2) + 价格位置 (3) = 39
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.backtest.features_v4 import FEATURE_NAMES_V4, compute_features_v4, get_target
from src.backtest.real_data import build_aggregated_klines, fetch_klines_paginated

log = logging.getLogger("ml.v4")


def build_training_dataset_v4(
    symbol: str = "BTCUSDT",
    days: int = 90,
    event_minutes: int = 30,
    split_ratio: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    用 v4 特征（39 个）构造训练/测试集。

    目标：close_t+event_minutes > close_t → 1（涨），否则 0（跌）。
    """
    log.info("[v4] %s: 拉 %d 天 30m K 线 ...", symbol, days)
    k30 = fetch_klines_paginated(symbol, "30m", days=days)
    log.info("[v4] %s: 30m K 线 %d 根", symbol, len(k30))
    k1h = build_aggregated_klines(k30, "1h")
    k4h = build_aggregated_klines(k30, "4h")
    log.info("[v4] %s: 1h K 线 %d 根, 4h K 线 %d 根", symbol, len(k1h), len(k4h))

    closes_30m = np.array([k.close for k in k30], dtype=float)
    highs_30m = np.array([k.high for k in k30], dtype=float)
    lows_30m = np.array([k.low for k in k30], dtype=float)
    volumes_30m = np.array([k.volume for k in k30], dtype=float)

    closes_1h = np.array([k.close for k in k1h], dtype=float)
    highs_1h = np.array([k.high for k in k1h], dtype=float)
    lows_1h = np.array([k.low for k in k1h], dtype=float)
    volumes_1h = np.array([k.volume for k in k1h], dtype=float)

    closes_4h = np.array([k.close for k in k4h], dtype=float)

    X_all, y_all = [], []
    # 跳过前 60 根（数据不够最长的指标）和最后 1 根（无 close_t+1）
    for i in range(60, len(k30) - 1):
        try:
            features = compute_features_v4(
                closes_30m, highs_30m, lows_30m, volumes_30m,
                closes_1h, highs_1h, lows_1h, volumes_1h,
                closes_4h,
                i_30m=i,
            )
        except Exception as e:
            log.debug("[v4] features fail at i=%d: %s", i, e)
            continue
        # 目标：30m 后 close > 当前 close
        y = get_target(closes_30m[i], closes_30m[i + 1])
        X_all.append(features)
        y_all.append(y)

    X_all = np.array(X_all, dtype=float)
    y_all = np.array(y_all, dtype=int)

    split_idx = int(len(X_all) * split_ratio)
    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]
    log.info("[v4] train: %d | test: %d | up_ratio: %.3f",
             len(X_train), len(X_test), y_all.mean())
    return X_train, X_test, y_train, y_test, FEATURE_NAMES_V4


def train_xgboost(X_train, y_train, X_test, y_test, *,
                    n_estimators: int = 400, max_depth: int = 5,
                    learning_rate: float = 0.05, random_state: int = 42):
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=0.8, colsample_bytree=0.7,
        objective="binary:logistic", eval_metric="logloss",
        random_state=random_state, n_jobs=2,
        reg_alpha=0.1, reg_lambda=1.0,
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
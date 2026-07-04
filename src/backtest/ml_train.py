"""
XGBoost 训练：从真实 K 线反推事件合约结果，预测 YES 中奖概率。

特征：
  - mom_15m       (15 根 K 线对数收益率)
  - bb_pct_1h     (布林带 %b)
  - atr_break_4h  (ATR 突破强度)
  - hour_of_day   (0~23，捕捉日内模式)
  - vol_1h        (最近 1h 波动率)
  - trend_24h     (24h 趋势强度，close/open-1)
  - ret_1h        (上一根 K 线收益率)
  - spread        (事件盘口价差)

目标：
  - y=1 if YES 赢，0 if YES 输
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from src.backtest.real_data import (
    build_4h_aggregated_klines,
    fetch_and_cache_klines,
    build_events_from_klines,
    load_binance_klines,
)
from src.signals.factors import (
    compute_atr_break_4h,
    compute_bb_pct_1h,
    compute_mom_15m,
)

log = logging.getLogger(__name__)


def build_training_dataset(
    symbol: str = "BTCUSDT",
    n_hours: int = 720,
    strike_offset_pct: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    从真实 K 线反推事件合约 → 构造训练样本。
    返回 (X, y, feature_names)。
    """
    k1h = fetch_and_cache_klines(symbol, interval="1h", limit=n_hours)
    k4h = build_4h_aggregated_klines(k1h)
    events = build_events_from_klines(k1h, symbol=symbol, direction="ABOVE", strike_offset_pct=strike_offset_pct)
    log.info("[ml] %s events=%d", symbol, len(events))

    X, y = [], []
    feature_names = ["mom_15m", "bb_pct_1h", "atr_break_4h", "hour_of_day", "vol_1h", "trend_24h", "ret_1h", "spread"]

    closes = np.array([k.close for k in k1h], dtype=float)
    for i, ev in enumerate(events):
        if i < 30:
            continue  # 数据不够
        # 特征
        k1m_seg = k1h[max(0, i - 15):i + 1]   # 用 1h 替代 1m
        k1h_seg = k1h[max(0, i - 19):i + 1]
        k4h_seg = k4h[max(0, i // 4 - 14):max(0, i // 4) + 1]
        if len(k1h_seg) < 20 or len(k4h_seg) < 14:
            continue
        try:
            mom = compute_mom_15m(k1m_seg, lookback=15)
            bb = compute_bb_pct_1h(k1h_seg, period=20, std_mult=2.0)
            atr = compute_atr_break_4h(k4h_seg, period=14, breakout_mult=1.0)
        except Exception:
            continue
        hour = k1h[i].open_time.hour
        # 最近 1h 波动率（用最近 24 根 close 计算）
        recent = closes[max(0, i - 24):i + 1]
        vol_1h = float(np.std(np.diff(np.log(recent)))) if len(recent) > 1 else 0.0
        # 24h 趋势
        trend_24h = (closes[i] / closes[max(0, i - 24)] - 1.0) if i >= 24 else 0.0
        # 上一根 K 线收益
        ret_1h = (closes[i] / closes[i - 1] - 1.0) if i >= 1 else 0.0
        spread = ev.spread

        # 目标：YES 是否赢
        settle_price = k1h[i].close
        yes_won = settle_price >= ev.strike_price

        X.append([mom, bb, atr, hour, vol_1h, trend_24h, ret_1h, spread])
        y.append(1 if yes_won else 0)

    return np.array(X, dtype=float), np.array(y, dtype=int), feature_names


def train_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
):
    """训练 XGBoost 二分类。"""
    from xgboost import XGBClassifier
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
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
    return model, X_train, X_test, y_train, y_test, train_acc, test_acc


def save_model(model, feature_names: List[str], path: str | Path) -> Path:
    """保存模型 + 特征名。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({"model": model, "feature_names": feature_names}, path)
    return path


def load_model(path: str | Path):
    """加载模型。"""
    import joblib
    return joblib.load(path)
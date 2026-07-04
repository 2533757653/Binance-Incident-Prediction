#!/usr/bin/env python
"""数据泄露检测：用同一组特征 + 三个不同 label 测试胜率。
- 真实 label: c5[i+6] vs c5[i]
- 未来 label: c5[i+12] vs c5[i+6]（1h 后的"6 根后涨/跌"）
- 噪声 label: random

如果三组胜率差不多 → 有数据泄露
如果真实 label > 未来 label ≈ 噪声 → 没有泄露
"""
import os, sys, logging

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import joblib
from datetime import datetime, timezone, timedelta
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("leak.test")

PROOF_DIR = "proofs/iter-15"


def fetch_5m_data(sym):
    from src.backtest.real_data import fetch_klines_paginated
    proxy = os.environ.get("HORIZON_PROXY", "http://127.0.0.1:7897")
    os.environ["HORIZON_PROXY"] = proxy
    return fetch_klines_paginated(sym, "5m", days=90)


def main():
    from src.backtest.features_v5 import (
        build_aggregated_klines_v2, _bars_to_arrays,
        build_all_indicators, compute_features_v5,
        get_target_30m,
        FEATURE_NAMES_V5,
    )

    sym = "BTCUSDT"
    k5 = fetch_5m_data(sym)
    log.info("  %d 5m bars", len(k5))
    k15 = build_aggregated_klines_v2(k5, "5m", "15m")
    k30 = build_aggregated_klines_v2(k5, "5m", "30m")
    k1h = build_aggregated_klines_v2(k5, "5m", "1h")
    k4h = build_aggregated_klines_v2(k5, "5m", "4h")

    o5, c5, h5, l5, v5 = _bars_to_arrays(k5)
    o15, c15, h15, l15, v15 = _bars_to_arrays(k15)
    o30, c30, h30, l30, v30 = _bars_to_arrays(k30)
    o1h, c1h, h1h, l1h, v1h = _bars_to_arrays(k1h)
    o4h, c4h, h4h, l4h, _ = _bars_to_arrays(k4h)

    inds = build_all_indicators(
        o5, c5, h5, l5, v5,
        o15, c15, h15, l15, v15,
        o30, c30, h30, l30, v30,
        o1h, c1h, h1h, l1h, v1h,
        o4h, c4h, h4h, l4h,
    )

    WARMUP_5M, MIN_LOOKAHEAD_5M = 60, 12
    valid_range = range(WARMUP_5M, len(k5) - MIN_LOOKAHEAD_5M)
    n_samples = len(valid_range)

    X = np.zeros((n_samples, len(FEATURE_NAMES_V5)), dtype=float)
    y_real = np.zeros(n_samples, dtype=int)    # c5[i+6] vs c5[i]
    y_future = np.zeros(n_samples, dtype=int)  # c5[i+12] vs c5[i+6]
    y_noise = np.zeros(n_samples, dtype=int)   # 随机

    np.random.seed(42)
    for idx, i5 in enumerate(valid_range):
        ot = k5[i5].open_time if hasattr(k5[i5], 'open_time') else None
        X[idx] = compute_features_v5(inds, c5, v5, c1h, v30, h30, l30, i5, open_time_5m=ot)
        y_real[idx] = 1 if c5[i5 + 6] > c5[i5] else 0
        y_future[idx] = 1 if c5[i5 + 12] > c5[i5 + 6] else 0
        y_noise[idx] = np.random.randint(0, 2)

    log.info("  X=%s", X.shape)
    log.info("  y_real up=%.3f, y_future up=%.3f, y_noise up=%.3f",
             y_real.mean(), y_future.mean(), y_noise.mean())

    # Load top features
    xgb_data = joblib.load(os.path.join(PROOF_DIR, f"{sym}_xgb_v5_30m_pruned.joblib"))
    top_idx = np.array(xgb_data["feature_indices"])
    X_sel = X[:, top_idx]

    # 5-fold CV with three labels
    kf = KFold(n_splits=5, shuffle=False)

    log.info("=" * 60)
    log.info("数据泄露检测：同一特征 + 三个不同 label")
    log.info("=" * 60)

    for label_name, y in [("real (c5[i+6] > c5[i])", y_real),
                          ("future (c5[i+12] > c5[i+6])", y_future),
                          ("noise (random)", y_noise)]:
        wrs = []
        sigs_total = 0
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_sel)):
            base = xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7,
                objective="binary:logistic", random_state=42, n_jobs=2,
            )
            model = CalibratedClassifierCV(base, method='isotonic', cv=5, n_jobs=1)
            model.fit(X_sel[train_idx], y[train_idx])
            probs = model.predict_proba(X_sel[test_idx])[:, 1]

            n_sig = n_cor = 0
            for i in range(len(test_idx)):
                prob = float(probs[i])
                if prob >= 0.55:
                    side = 1
                elif prob <= 0.45:
                    side = 0
                else:
                    continue
                actual = y[test_idx[i]]
                correct = (side == 1 and actual == 1) or (side == 0 and actual == 0)
                n_sig += 1
                if correct:
                    n_cor += 1
            wr = n_cor / n_sig if n_sig else 0
            wrs.append(wr)
            sigs_total += n_sig

        log.info("  %-35s | 平均胜率 %.1f%% | 信号量 %d", label_name, np.mean(wrs) * 100, sigs_total)
    log.info("=" * 60)
    log.info("判断：")
    log.info("  real ≈ future ≈ noise → 严重泄露，模型啥也学不到")
    log.info("  real > future ≈ noise → 正常（模型真的学到模式）")


if __name__ == "__main__":
    main()
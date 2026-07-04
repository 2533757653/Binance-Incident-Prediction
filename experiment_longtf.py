#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验：给 1h 目标额外加入 12h/1d/1w 周期特征，是否提升？
只读现有模块（causal_features 的 _aligned_tf/_causal_idx + features_v5 的指标构建），不改任何底层文件。
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging; logging.disable(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
import numpy as np
from sklearn.metrics import roc_auc_score
from src.backtest.real_data import fetch_klines_paginated
from src.features.causal_features import compute_matrix, _aligned_tf, _causal_idx
from src.backtest.features_v5 import _build_tf_indicators
from train_honest import new_model, winrate

WARMUP, H, DAYS = 60, 12, 365
EXTRA = {"12h": 720, "1d": 1440, "1w": 10080}
KEYS = ["rsi", "roc", "macd", "macd_hist", "ema9_21", "bb_pos", "price_pos", "adx"]


def extra_cols(k5, n_rows):
    o5 = np.array([b.open for b in k5], float); h5 = np.array([b.high for b in k5], float)
    l5 = np.array([b.low for b in k5], float); c5 = np.array([b.close for b in k5], float)
    v5 = np.array([b.volume for b in k5], float)
    oms = np.array([int(b.open_time.timestamp() * 1000) for b in k5], dtype=np.int64)
    idxs = WARMUP + np.arange(n_rows)
    out = {}
    for name, m in EXTRA.items():
        o, h, l, c, v, buckets = _aligned_tf(oms, o5, h5, l5, c5, v5, m)
        ind = _build_tf_indicators(o, c, h, l, v, name)
        cm = _causal_idx(oms, m, buckets)[idxs]
        cols = [(ind[k][cm] if k in ind else np.zeros(n_rows)) for k in KEYS]
        out[name] = np.column_stack(cols)
    return out


def wf(X, y, n_folds=5):
    N = len(X); block = N // (n_folds + 1); aucs, w62 = [], []; s62 = 0
    for k in range(1, n_folds + 1):
        tr = k * block; te = min(tr + block, N)
        m = new_model(); m.fit(X[:tr], y[:tr])
        p = m.predict_proba(X[tr:te])[:, 1]; yt = y[tr:te]
        if len(np.unique(yt)) > 1:
            aucs.append(roc_auc_score(yt, p))
        w, s = winrate(p, yt, 0.62); w62.append(w) if s else None; s62 += s
    return (np.mean(aucs) if aucs else .5, np.mean(w62) if w62 else 0, s62)


def main():
    for sym in ["BTCUSDT", "ETHUSDT"]:
        print("=" * 60); print(f"  {sym}（1年, 1h目标, walk-forward）"); print("=" * 60)
        k5 = fetch_klines_paginated(sym, "5m", days=DAYS)
        X, _, y1h, names, times = compute_matrix(k5, min_lookahead_5m=H)
        ex = extra_cols(k5, len(X))
        sets = {
            "基线(100特征)": X,
            "+12h": np.column_stack([X, ex["12h"]]),
            "+12h+1d": np.column_stack([X, ex["12h"], ex["1d"]]),
            "+12h+1d+1w": np.column_stack([X, ex["12h"], ex["1d"], ex["1w"]]),
        }
        print(f"  {'特征集':<16}{'特征数':>7}{'AUC':>8}{'胜率@0.62':>11}{'信号':>8}")
        base_auc = None
        for tag, Xs in sets.items():
            auc, w, s = wf(Xs, y1h)
            if base_auc is None:
                base_auc = auc
            flag = "" if tag == "基线(100特征)" else (f"{C_UP if auc>base_auc+0.002 else (C_DN if auc<base_auc-0.002 else C_EQ)}")
            print(f"  {tag:<16}{Xs.shape[1]:>7}{auc:>8.3f}{w*100:>10.1f}%{s:>8}  {flag}")
        print("  判断：AUC/胜率相对基线明显上升才算有用；持平或下降=灌水/过拟合。")


C_UP, C_DN, C_EQ = "↑有提升", "↓变差", "≈没变"
if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""实验：不同预测周期(30m~1d)哪个更可预测？
关键指标：
- AUC：模型真实区分力，0.5=瞎猜，>0.55才算有点本事（不受涨跌偏向影响）
- 基准：市场本身"涨"的比例(取多数类)，用来识别"顺风≠会预测"
- edge：阈值胜率 - 基准，真正超越"无脑押多数方"的部分
"""
import os, sys, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.disable(logging.INFO)

import numpy as np
from sklearn.metrics import roc_auc_score
from src.backtest.real_data import fetch_klines_paginated
from src.features.causal_features import compute_matrix
from train_honest import new_model, winrate

WARMUP = 60
MAXH = 288
HORIZONS = [(6, "30m"), (12, "1h"), (24, "2h"), (48, "4h"), (96, "8h"), (144, "12h"), (288, "1d")]
DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 90


def wf(X, y, n_folds=5):
    N = len(X); block = N // (n_folds + 1)
    aucs, wr55, sig55, wr60, sig60 = [], [], 0, [], 0
    for k in range(1, n_folds + 1):
        tr = k * block; te = min(tr + block, N)
        m = new_model(); m.fit(X[:tr], y[:tr])
        p = m.predict_proba(X[tr:te])[:, 1]; yt = y[tr:te]
        if len(np.unique(yt)) > 1:
            aucs.append(roc_auc_score(yt, p))
        w, s = winrate(p, yt, 0.55); wr55.append(w) if s else None; sig55 += s
        w, s = winrate(p, yt, 0.60); wr60.append(w) if s else None; sig60 += s
    return (np.mean(aucs) if aucs else .5,
            np.mean(wr55) if wr55 else 0, sig55,
            np.mean(wr60) if wr60 else 0, sig60)


def main():
    for sym in ["BTCUSDT", "ETHUSDT"]:
        print("=" * 66)
        print(f"  {sym}  （{DAYS}天 5m 数据）")
        print("=" * 66)
        k5 = fetch_klines_paginated(sym, "5m", days=DAYS)
        c5 = np.array([b.close for b in k5], float)
        X, _, _, names, _ = compute_matrix(k5, min_lookahead_5m=MAXH)
        N = len(X); idx = WARMUP + np.arange(N)
        print(f"  {'周期':<6}{'基准涨%':>9}{'AUC':>8}{'胜率@0.55':>11}{'胜率@0.60':>11}{'edge@0.60':>11}")
        for h, lab in HORIZONS:
            y = (c5[idx + h] > c5[idx]).astype(int)
            base = max(y.mean(), 1 - y.mean())
            auc, w55, s55, w60, s60 = wf(X, y)
            edge = (w60 - base) * 100
            tag = "←有料" if auc > 0.55 else ""
            print(f"  {lab:<6}{base*100:>8.1f}%{auc:>8.3f}{w55*100:>10.1f}%{w60*100:>10.1f}%{edge:>+9.1f}pp {tag}")
        print("  解读：AUC≈0.50 = 没有真本事；edge≈0或负 = 高胜率只是顺风(基准高)，非预测力。")


if __name__ == "__main__":
    main()

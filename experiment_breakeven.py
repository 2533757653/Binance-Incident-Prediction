#!/usr/bin/env python
"""盈亏平衡分析（盈亏比 0.85）+ 提高置信门槛能否把胜率推过保本线。
盈亏比 b=0.85：赢 +0.85，输 -1。
保本胜率 p* = 1/(1+b) = 1/1.85 = 54.05%
每笔期望 EV = p*(1+b) - 1 = 1.85p - 1
"""
import os, sys, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.disable(logging.INFO)

import numpy as np
from src.backtest.real_data import fetch_klines_paginated
from src.features.causal_features import compute_matrix
from train_honest import new_model, winrate

B = 0.85
P_STAR = 1 / (1 + B)
THRS = [0.55, 0.58, 0.60, 0.62, 0.65, 0.70]
WARMUP = 60
H = 12  # 1h = 12 根 5m


def wf_thresholds(X, y, n_folds=5):
    N = len(X); block = N // (n_folds + 1)
    acc = {t: {"w": [], "s": 0} for t in THRS}
    for k in range(1, n_folds + 1):
        tr = k * block; te = min(tr + block, N)
        m = new_model(); m.fit(X[:tr], y[:tr])
        p = m.predict_proba(X[tr:te])[:, 1]; yt = y[tr:te]
        for t in THRS:
            w, s = winrate(p, yt, t)
            if s:
                acc[t]["w"].append(w)
            acc[t]["s"] += s
    return {t: (np.mean(v["w"]) if v["w"] else 0, v["s"]) for t, v in acc.items()}


def main():
    print("=" * 64)
    print(f"  事件合约 1h 盈亏分析  |  盈亏比 b={B}")
    print(f"  保本胜率 p* = 1/(1+{B}) = {P_STAR*100:.2f}%")
    print(f"  每笔期望 EV = 1.85×胜率 − 1   （胜率每+1pp ≈ EV +1.85%）")
    print("=" * 64)
    for sym in ["BTCUSDT", "ETHUSDT"]:
        k5 = fetch_klines_paginated(sym, "5m", days=365)
        c5 = np.array([b.close for b in k5], float)
        X, _, _, _, _ = compute_matrix(k5, min_lookahead_5m=H)
        idx = WARMUP + np.arange(len(X))
        y = (c5[idx + H] > c5[idx]).astype(int)
        res = wf_thresholds(X, y)
        print(f"\n  {sym}（1年, walk-forward）")
        print(f"    {'门槛':<6}{'胜率':>8}{'vs保本':>9}{'EV/笔':>9}{'信号数':>9}  判定")
        for t in THRS:
            wr, s = res[t]
            ev = (1.85 * wr - 1) * 100
            gap = (wr - P_STAR) * 100
            verdict = "赚" if ev > 0.3 else ("保本" if ev > -0.3 else "亏")
            print(f"    {t:<6}{wr*100:>7.1f}%{gap:>+8.1f}pp{ev:>+8.2f}%{s:>9}  {verdict}")
    print("\n  注：高门槛=只在最有把握时下注，信号变少；胜率若仍上不去54%，说明没救。")


if __name__ == "__main__":
    main()

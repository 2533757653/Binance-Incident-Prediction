#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验：BTC/ETH 信号协同能否提升胜率。
对比：各自单独出信号 vs 只在【两者同向】时才出信号(协同过滤)。
1年数据 / walk-forward / 1h 目标。保本 54%。
"""
import os, sys, io, logging
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.getLogger("httpx").setLevel(logging.WARNING); logging.disable(logging.INFO)

import numpy as np
from src.backtest.real_data import fetch_klines_paginated
from src.features.causal_features import compute_matrix
from train_honest import new_model

P_STAR = 1 / 1.85
THRS = [0.58, 0.62, 0.66, 0.70]


def wf_collect(Xb, yb, Xe, ye, n_folds=5):
    N = len(Xb); block = N // (n_folds + 1)
    PB, PE, YB, YE = [], [], [], []
    for k in range(1, n_folds + 1):
        tr = k * block; te = min(tr + block, N)
        mb = new_model(); mb.fit(Xb[:tr], yb[:tr])
        me = new_model(); me.fit(Xe[:tr], ye[:tr])
        PB.append(mb.predict_proba(Xb[tr:te])[:, 1]); YB.append(yb[tr:te])
        PE.append(me.predict_proba(Xe[tr:te])[:, 1]); YE.append(ye[tr:te])
    return (np.concatenate(PB), np.concatenate(PE), np.concatenate(YB), np.concatenate(YE))


def wr(mask_up, mask_dn, y):
    n = mask_up.sum() + mask_dn.sum()
    if n == 0:
        return 0.0, 0
    correct = (y[mask_up] == 1).sum() + (y[mask_dn] == 0).sum()
    return correct / n, int(n)


def main():
    print("加载数据+训练中(约4分钟)...")
    kb = fetch_klines_paginated("BTCUSDT", "5m", days=365)
    ke = fetch_klines_paginated("ETHUSDT", "5m", days=365)
    Xb, _, yb, _, tb = compute_matrix(kb)
    Xe, _, ye, _, te = compute_matrix(ke)
    # 按时间戳对齐
    ie = {int(t.timestamp()): j for j, t in enumerate(te)}
    pairs = [(i, ie[int(t.timestamp())]) for i, t in enumerate(tb) if int(t.timestamp()) in ie]
    bi = np.array([p[0] for p in pairs]); ei = np.array([p[1] for p in pairs])
    Xb, yb = Xb[bi], yb[bi]; Xe, ye = Xe[ei], ye[ei]
    print(f"对齐样本 {len(Xb)}")

    pb, pe, yb, ye = wf_collect(Xb, yb, Xe, ye)

    print("\n" + "=" * 66)
    print(f"  BTC/ETH 协同实验（保本 {P_STAR*100:.1f}%，1h，1年walk-forward）")
    print("=" * 66)
    print(f"  {'门槛':<6}{'BTC单独':>14}{'BTC(ETH同向)':>16}{'ETH单独':>14}{'ETH(BTC同向)':>16}")
    for t in THRS:
        bt_up, bt_dn = pb >= t, pb <= 1 - t
        et_up, et_dn = pe >= t, pe <= 1 - t
        # 单独
        sb = wr(bt_up, bt_dn, yb)
        se = wr(et_up, et_dn, ye)
        # 协同：两者同向
        cons_up = bt_up & et_up
        cons_dn = bt_dn & et_dn
        cb = wr(cons_up, cons_dn, yb)   # 协同时 BTC 命中
        ce = wr(cons_up, cons_dn, ye)   # 协同时 ETH 命中
        def f(x): return f"{x[0]*100:.1f}%({x[1]})"
        print(f"  {t:<6}{f(sb):>14}{f(cb):>16}{f(se):>14}{f(ce):>16}")
    print("=" * 66)
    print("  括号内=信号数。协同列若胜率明显更高，说明'两者同向'确实是更强信号。")


if __name__ == "__main__":
    main()

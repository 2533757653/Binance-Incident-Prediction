#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验：日线次日涨跌预测，「恐慌贪婪指数」到底加不加分？

诚实消融：同一套日线数据 / 同一 walk-forward 切分，对比
  BASE  = 价格+量能+时间 (px_/vol_/tm_)          —— 纯价格系
  +FnG  = BASE + 恐慌贪婪指数族 (fng_)            —— 加价格无关信息
关键指标 AUC（区分力，0.5=瞎猜）+ 阈值胜率（含信号数）。
数据尽量长（~7 年日线），BTC/ETH 共用同一条 F&G 序列。可复现。
"""
import io
import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.disable(logging.INFO)

import numpy as np
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.backtest.real_data import fetch_klines_paginated
from src.features.daily_features import compute_daily_matrix

_TZ = timezone(timedelta(hours=8))
DAYS = 2600
THRS = [0.52, 0.55, 0.58, 0.60]
OUT = os.path.join(os.path.dirname(__file__), "proofs", "iter-19")
os.makedirs(OUT, exist_ok=True)


def new_model():
    # 日线样本少 → 强正则、浅树，防过拟合
    return XGBClassifier(
        n_estimators=220, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=3.0,
        min_child_weight=6, objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=4)


def winrate(p, y, thr):
    ns = nc = 0
    for pi, a in zip(p, y):
        if pi >= thr:
            s = 1
        elif pi <= 1 - thr:
            s = 0
        else:
            continue
        ns += 1
        nc += int(s == a)
    return (nc / ns if ns else 0.0), ns


def walk_forward(X, y, n_folds=6):
    n = len(X)
    block = n // (n_folds + 1)
    aucs, agg = [], {t: {"c": 0, "n": 0} for t in THRS}
    for k in range(1, n_folds + 1):
        tr, te = k * block, min((k + 1) * block, n)
        m = new_model()
        m.fit(X[:tr], y[:tr])
        p = m.predict_proba(X[tr:te])[:, 1]
        yt = y[tr:te]
        if len(np.unique(yt)) > 1:
            aucs.append(roc_auc_score(yt, p))
        for t in THRS:
            wr, ns = winrate(p, yt, t)
            agg[t]["c"] += int(round(wr * ns))
            agg[t]["n"] += ns
    auc = float(np.mean(aucs)) if aucs else 0.5
    wr = {t: (agg[t]["c"] / agg[t]["n"] if agg[t]["n"] else 0.0, agg[t]["n"]) for t in THRS}
    return auc, wr


def run_symbol(sym, summary):
    print("=" * 70)
    k = fetch_klines_paginated(sym, "1d", days=DAYS)
    Xdf, y, dates = compute_daily_matrix(k, use_fng=True)
    cols = list(Xdf.columns)
    base_cols = [c for c in cols if not c.startswith("fng_")]
    fng_cols = [c for c in cols if c.startswith("fng_")]
    d0, d1 = dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")
    print(f"  {sym}  日线样本 {len(y)}  ({d0} ~ {d1})  次日涨占比 {y.mean()*100:.1f}%")
    print(f"  特征：BASE {len(base_cols)} 个（价格+量能+时间） / +FnG 再加 {len(fng_cols)} 个恐慌贪婪族")

    Xb = Xdf[base_cols].to_numpy(float)
    Xf = Xdf[cols].to_numpy(float)
    auc_b, wr_b = walk_forward(Xb, y)
    auc_f, wr_f = walk_forward(Xf, y)

    print(f"\n  {'配置':<10}{'AUC':>8}" + "".join(f"{'胜率@'+str(t):>13}" for t in THRS))
    def line(tag, auc, wr):
        cells = "".join(f"{wr[t][0]*100:>7.1f}%({wr[t][1]:>3}){'':1}" for t in THRS)
        return f"  {tag:<10}{auc:>8.3f}   {cells}"
    print(line("BASE", auc_b, wr_b))
    print(line("+FnG", auc_f, wr_f))
    dauc = auc_f - auc_b
    print(f"  → 加 F&G 后 AUC 变化：{dauc:+.3f}   "
          f"({'↑ 有增益' if dauc > 0.003 else ('↓ 变差' if dauc < -0.003 else '≈ 基本没变')})")

    # 特征重要度：F&G 族排在哪
    m = new_model(); m.fit(Xf, y)
    imp = sorted(zip(cols, m.feature_importances_), key=lambda z: -z[1])
    top = [(c, round(float(w), 4)) for c, w in imp[:12]]
    n_fng_in_top = sum(1 for c, _ in top if c.startswith("fng_"))
    print(f"  重要度 Top12 里 F&G 族占 {n_fng_in_top} 个：", ", ".join(c for c, _ in top[:12]))

    summary["results"][sym] = {
        "samples": len(y), "range": [d0, d1], "up_rate": round(float(y.mean()), 4),
        "n_base": len(base_cols), "n_fng": len(fng_cols),
        "BASE": {"auc": round(auc_b, 4), "wr": {str(t): [round(wr_b[t][0], 4), wr_b[t][1]] for t in THRS}},
        "+FnG": {"auc": round(auc_f, 4), "wr": {str(t): [round(wr_f[t][0], 4), wr_f[t][1]] for t in THRS}},
        "delta_auc": round(dauc, 4),
        "top_features": top,
    }


def main():
    print("加载日线 + F&G，训练消融中（约 1~2 分钟）...")
    summary = {"ts": datetime.now(_TZ).isoformat(), "target": "次日涨跌(1d)",
               "thresholds": THRS, "results": {}}
    for sym in ["BTCUSDT", "ETHUSDT"]:
        run_symbol(sym, summary)
    with open(os.path.join(OUT, "fng_experiment.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 70)
    print("  说明：AUC>0.53 才算有真区分力；日线样本少，阈值胜率括号内=信号数，")
    print("  信号越少误差带越大。结果已存 proofs/iter-19/fng_experiment.json")


if __name__ == "__main__":
    main()

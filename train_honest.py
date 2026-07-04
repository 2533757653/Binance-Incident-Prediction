#!/usr/bin/env python
"""诚实训练 + Walk-Forward 回测（因果特征，无泄露）。
- 数据：Bybit/OKX/Gate（无 Binance）
- 验证：时间顺序 walk-forward（过去训练 → 未来测试）
- 产出：每币种 30m/1h 模型 + 真实胜率报告
"""
import os, sys, json, time, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("train.honest")

import numpy as np
import joblib
from datetime import datetime, timezone, timedelta
from xgboost import XGBClassifier

from src.backtest.real_data import fetch_klines_paginated
from src.features.causal_features import compute_matrix, FEATURE_NAMES_V5

_TZ = timezone(timedelta(hours=8))
OUT = os.path.join(os.path.dirname(__file__), "proofs", "honest")
os.makedirs(OUT, exist_ok=True)
DAYS = 365          # 用满 1 年数据训练，更稳健（避免单一行情过拟合）
THRS = [0.55, 0.60, 0.65, 0.70]


def new_model():
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
        objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=4)


def winrate(probs, y, thr):
    ns = nc = 0
    for p, a in zip(probs, y):
        if p >= thr: s = 1
        elif p <= 1 - thr: s = 0
        else: continue
        ns += 1; nc += int(s == a)
    return (nc / ns if ns else 0.0), ns


def walk_forward(X, y, n_folds=5):
    """过去训练→未来测试，扩张窗口。返回各阈值平均胜率+总信号。"""
    n = len(X)
    block = n // (n_folds + 1)
    agg = {t: {"wr": [], "sig": 0} for t in THRS}
    for k in range(1, n_folds + 1):
        tr_end = k * block
        te_end = min(tr_end + block, n)
        m = new_model()
        m.fit(X[:tr_end], y[:tr_end])
        p = m.predict_proba(X[tr_end:te_end])[:, 1]
        yt = y[tr_end:te_end]
        for t in THRS:
            wr, ns = winrate(p, yt, t)
            if ns:
                agg[t]["wr"].append(wr)
            agg[t]["sig"] += ns
    return {t: (float(np.mean(v["wr"])) if v["wr"] else 0.0, v["sig"]) for t, v in agg.items()}


def main():
    summary = {"ts": datetime.now(_TZ).isoformat(), "features": len(FEATURE_NAMES_V5), "results": {}}
    for sym in ["BTCUSDT", "ETHUSDT"]:
        log.info("=" * 56)
        log.info("=== %s ===", sym)
        k5 = fetch_klines_paginated(sym, "5m", days=DAYS)
        t0 = time.time()
        X, y30, y1h, names, times = compute_matrix(k5)
        log.info("特征矩阵 %s  (%.0fs)  up30=%.3f up1h=%.3f",
                 X.shape, time.time() - t0, y30.mean(), y1h.mean())

        summary["results"][sym] = {"n_samples": len(X)}
        for tgt, y in [("30m", y30), ("1h", y1h)]:
            wf = walk_forward(X, y)
            log.info("  [%s 目标] Walk-Forward 真实胜率：", tgt)
            for t in THRS:
                wr, sig = wf[t]
                log.info("     阈值%.2f → 胜率 %.1f%%  信号 %d 次", t, wr * 100, sig)

            # 最终模型：前 85% 训练，最后 15% 留作样本外（仅记录，不调参）
            split = int(len(X) * 0.85)
            m = new_model(); m.fit(X[:split], y[:split])
            p_oos = m.predict_proba(X[split:])[:, 1]
            oos = {f"{t}": list(winrate(p_oos, y[split:], t)) for t in THRS}

            joblib.dump({"model": m, "feature_names": names, "target": tgt,
                         "symbol": sym, "walk_forward": {str(t): wf[t] for t in THRS}},
                        os.path.join(OUT, f"{sym}_{tgt}.joblib"))
            summary["results"][sym][tgt] = {
                "walk_forward": {str(t): wf[t] for t in THRS},
                "oos_last15pct": oos,
            }
    with open(os.path.join(OUT, "honest_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 终版对照表
    print("\n" + "=" * 60)
    print("  诚实回测结果（因果特征 / Walk-Forward / 真实可复现）")
    print("=" * 60)
    print(f"  {'币种/目标':14s}{'门槛0.55':>10s}{'门槛0.65':>10s}{'门槛0.70':>10s}")
    for sym in ["BTCUSDT", "ETHUSDT"]:
        for tgt in ["30m", "1h"]:
            r = summary["results"][sym][tgt]["walk_forward"]
            def cell(t):
                wr, sig = r[str(t)]; return f"{wr*100:.1f}%({sig//1000}k)"
            print(f"  {sym+'/'+tgt:14s}{cell(0.55):>10s}{cell(0.65):>10s}{cell(0.70):>10s}")
    print("=" * 60)
    print(f"  盈亏比0.85 → 保本胜率 54.05%。门槛≥0.65 才进盈利区。")
    print(f"  对照：泄露版曾吹 74~80%（假）。50%=抛硬币。")


if __name__ == "__main__":
    main()

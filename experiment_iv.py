#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验：期权隐含波动率(DVOL) + 方差风险溢价，对日线预测加不加分？

诚实消融（同 F&G 那套方法），在 DVOL 有历史的时段(2021-03 至今)：
  BASE   价格+量能+时间
  +IV    再加 隐含波动率族(iv_)：水平/变化/z分/分位 + **方差风险溢价(IV−realized)**
  +FnG   再加 恐慌贪婪族
  +ALL   两者都加
另测：方差风险溢价分档 → 未来收益（IV 相对已实现波动"贵不贵"是否预示后市）。
可复现。DVOL=Deribit 加密版 VIX。
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging; logging.disable(logging.INFO); logging.getLogger("httpx").setLevel(logging.WARNING)
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from src.backtest.real_data import fetch_klines_paginated
from src.features.daily_features import compute_daily_matrix
from src.data.deribit_vol import fetch_dvol, align_dvol
from experiment_fng import new_model, walk_forward, THRS  # 该模块导入时会把 stdout 设为 UTF-8

_TZ = timezone(timedelta(hours=8))
OUT = os.path.join(os.path.dirname(__file__), "proofs", "iter-19")
os.makedirs(OUT, exist_ok=True)
CCY = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}


def iv_features(dates, hv20, ccy):
    """基于 DVOL 构建 iv_ 特征，行对齐 dates。hv20=20日已实现日波动(来自 matrix)。"""
    iv_raw = align_dvol(dates, fetch_dvol(ccy, days=2000), ccy)  # 年化%，前段可能 NaN
    iv = pd.Series(iv_raw, dtype=float)
    ivf = iv / 100.0
    rv_annual = pd.Series(hv20, dtype=float) * np.sqrt(365.0)  # 已实现波动年化
    df = pd.DataFrame(index=range(len(dates)))
    df["iv_level"] = ivf
    df["iv_chg5"] = ivf.diff(5)
    df["iv_z20"] = (iv - iv.rolling(20).mean()) / (iv.rolling(20).std() + 1e-9)
    df["iv_pct90"] = iv.rolling(90).apply(
        lambda a: float(np.mean(a[:-1] < a[-1])) if len(a) > 1 else 0.5, raw=True)
    df["iv_vrp"] = ivf - rv_annual              # ★ 方差风险溢价
    df["iv_vrp_z"] = (df["iv_vrp"] - df["iv_vrp"].rolling(20).mean()) / (df["iv_vrp"].rolling(20).std() + 1e-9)
    df["iv_regime"] = np.sign(iv - iv.rolling(60).mean())
    valid = ~iv.isna().to_numpy()
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0), valid


def run(sym, summary):
    print("=" * 72)
    k = fetch_klines_paginated(sym, "1d", days=2600)
    Xdf, y, dates = compute_daily_matrix(k, use_fng=True)
    cols = list(Xdf.columns)
    base = [c for c in cols if c[:3] in ("px_", "vol") or c.startswith("tm_")]
    fng = [c for c in cols if c.startswith("fng_")]
    ivdf, valid = iv_features(dates, Xdf["px_hv20"].to_numpy(), CCY[sym])

    # 限定在 DVOL 有历史的行
    Xall = pd.concat([Xdf.reset_index(drop=True), ivdf.reset_index(drop=True)], axis=1)
    Xall, yv, dts = Xall[valid].reset_index(drop=True), y[valid], [d for d, v in zip(dates, valid) if v]
    ivcols = list(ivdf.columns)
    d0, d1 = dts[0].strftime("%Y-%m-%d"), dts[-1].strftime("%Y-%m-%d")
    print(f"  {sym}  DVOL时段样本 {len(yv)} ({d0}~{d1})  次日涨占比 {yv.mean()*100:.1f}%")

    sets = {"BASE": base, "+IV": base + ivcols, "+FnG": base + fng, "+ALL": base + fng + ivcols}
    print(f"\n  {'配置':<8}{'AUC':>8}{'胜率@0.55':>12}{'胜率@0.58':>12}")
    res = {}
    for tag, use in sets.items():
        auc, wr = walk_forward(Xall[use].to_numpy(float), yv)
        res[tag] = {"auc": round(auc, 4), "wr55": [round(wr[0.55][0], 4), wr[0.55][1]],
                    "wr58": [round(wr[0.58][0], 4), wr[0.58][1]]}
        print(f"  {tag:<8}{auc:>8.3f}{wr[0.55][0]*100:>10.1f}%({wr[0.55][1]:>3}){wr[0.58][0]*100:>9.1f}%({wr[0.58][1]:>3})")
    dauc = res["+IV"]["auc"] - res["BASE"]["auc"]
    print(f"  → 加 IV 后 AUC 变化：{dauc:+.3f}  ({'↑ 有增益' if dauc>0.003 else ('↓ 变差' if dauc<-0.003 else '≈ 基本没变')})")

    # 方差风险溢价分档 → 未来收益
    c = np.array([b.close for b in k], float)
    tms = [b.open_time for b in k]
    ivraw = align_dvol(tms, fetch_dvol(CCY[sym], 2000), CCY[sym]) / 100.0
    ret = pd.Series(c).pct_change()
    rv = ret.rolling(20).std().to_numpy() * np.sqrt(365.0)
    vrp = ivraw - rv
    print("  方差风险溢价(IV−realized)分档 → 未来10天平均涨幅：")
    vp_rec = {}
    m = ~np.isnan(vrp)
    q = np.nanquantile(vrp[m], [1/3, 2/3])
    for name, lo, hi in [("低(IV便宜)", -1e9, q[0]), ("中", q[0], q[1]), ("高(IV贵/恐慌)", q[1], 1e9)]:
        idx = [i for i in range(len(c) - 10) if m[i] and lo <= vrp[i] < hi]
        fr = [c[i + 10] / c[i] - 1 for i in idx]
        mean = float(np.mean(fr)) * 100 if fr else 0.0
        print(f"     {name:<12} n={len(fr):>4}  10天 {mean:+.2f}%")
        vp_rec[name] = {"n": len(fr), "fwd10_pct": round(mean, 3)}

    summary["results"][sym] = {"samples": len(yv), "range": [d0, d1], "ablation": res,
                               "delta_auc_iv": round(dauc, 4), "vrp_buckets": vp_rec}


def main():
    print("加载日线 + DVOL(Deribit) + F&G，消融中（约 1~2 分钟）...")
    summary = {"ts": datetime.now(_TZ).isoformat(), "target": "次日涨跌(1d)", "results": {}}
    for sym in ["BTCUSDT", "ETHUSDT"]:
        run(sym, summary)
    with open(os.path.join(OUT, "iv_experiment.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 72)
    print("  判据：AUC 相对 BASE 明显上升(>0.005)才算 IV 有用；否则=又一个价格下游冗余。")
    print("  结果存 proofs/iter-19/iv_experiment.json")


if __name__ == "__main__":
    main()

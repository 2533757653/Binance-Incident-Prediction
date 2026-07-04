#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验：恐慌贪婪指数的「真本事」在哪个持有周期？（反向/择时视角）

上一个实验证明 F&G 对*次日*涨跌没用。但 F&G 是慢变的**反向情绪**指标，
它的经典用法是「别人恐惧我贪婪」——在极端恐慌后持有*一段时间*才见效。
本实验直接量化：按 F&G 分档，看未来 1/3/5/10/20/30 天的收益，是否
「越恐慌→后市越涨」。纯统计、无 ML、无过拟合、可复现。
"""
import io, os, sys, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging; logging.disable(logging.INFO); logging.getLogger("httpx").setLevel(logging.WARNING)
import numpy as np
from datetime import datetime, timezone, timedelta
from src.backtest.real_data import fetch_klines_paginated
from src.data.fear_greed import fetch_fng, align_fng

_TZ = timezone(timedelta(hours=8))
DAYS = 2600
HORIZONS = [1, 3, 5, 10, 20, 30]
OUT = os.path.join(os.path.dirname(__file__), "proofs", "iter-19")
os.makedirs(OUT, exist_ok=True)

# F&G 分档（标准区间）
BUCKETS = [("极端恐慌≤25", 0, 25), ("恐慌26-45", 26, 45),
           ("中性46-54", 46, 54), ("贪婪55-74", 55, 74), ("极端贪婪≥75", 75, 100)]


def load(sym):
    k = fetch_klines_paginated(sym, "1d", days=DAYS)
    c = np.array([b.close for b in k], float)
    t = [b.open_time for b in k]
    fng = align_fng(t, fetch_fng())
    return c, fng, t


def fwd_ret(c, i, h):
    return c[i + h] / c[i] - 1 if i + h < len(c) else np.nan


def main():
    print("加载日线 + F&G ...")
    data = {s: load(s) for s in ["BTCUSDT", "ETHUSDT"]}
    # 合并池（BTC+ETH），样本更稳
    summary = {"ts": datetime.now(_TZ).isoformat(), "horizons": HORIZONS, "pool": {}, "per_symbol": {}}

    # ---- 池化：每个(币,日)一条样本 ----
    rows = []  # (fng_value, {h: fwd_ret})
    for s, (c, fng, t) in data.items():
        for i in range(len(c)):
            fr = {h: fwd_ret(c, i, h) for h in HORIZONS}
            rows.append((fng[i], fr))

    print("\n" + "=" * 84)
    print("  F&G 分档 → 未来各持有天数的【平均收益%】（BTC+ETH 合并，2019-12 至今）")
    print("=" * 84)
    header = f"  {'F&G 档位':<14}{'样本':>6}" + "".join(f"{str(h)+'天':>10}" for h in HORIZONS)
    print(header)
    base = {}
    allrows = rows
    for h in HORIZONS:
        vals = [r[1][h] for r in allrows if not np.isnan(r[1][h])]
        base[h] = float(np.mean(vals))
    for name, lo, hi in BUCKETS:
        sub = [r for r in allrows if lo <= r[0] <= hi]
        cells, rec = "", {}
        for h in HORIZONS:
            vv = [r[1][h] for r in sub if not np.isnan(r[1][h])]
            mean = float(np.mean(vv)) * 100 if vv else 0.0
            pos = float(np.mean([1 for x in vv if x > 0])) if vv else 0.0
            cells += f"{mean:>9.2f}%"
            rec[h] = {"mean_pct": round(mean, 3), "pos_rate": round(np.mean([x > 0 for x in vv]) if vv else 0, 3), "n": len(vv)}
        print(f"  {name:<14}{len(sub):>6}{cells}")
        summary["pool"][name] = rec
    print("  " + "-" * 80)
    print(f"  {'全样本基准':<14}{len(allrows):>6}" + "".join(f"{base[h]*100:>9.2f}%" for h in HORIZONS))
    summary["pool"]["_baseline"] = {h: round(base[h] * 100, 3) for h in HORIZONS}

    # 极端恐慌相对基准的超额
    print("\n  ★ 极端恐慌(≤25) 相对基准的超额收益（正=反向做多有效）：")
    ef = summary["pool"]["极端恐慌≤25"]
    exc = {h: round(ef[h]["mean_pct"] - base[h] * 100, 2) for h in HORIZONS}
    print("    " + "  ".join(f"{h}天:{exc[h]:+.2f}%" for h in HORIZONS))
    summary["excess_extreme_fear"] = exc

    # 极端贪婪相对基准
    eg = summary["pool"]["极端贪婪≥75"]
    excg = {h: round(eg[h]["mean_pct"] - base[h] * 100, 2) for h in HORIZONS}
    print("  ★ 极端贪婪(≥75) 相对基准的超额收益（负=该减仓/反向做空）：")
    print("    " + "  ".join(f"{h}天:{excg[h]:+.2f}%" for h in HORIZONS))
    summary["excess_extreme_greed"] = excg

    with open(os.path.join(OUT, "fng_horizon.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 84)
    print("  结论看两点：①极端恐慌行是否随天数走高且超额转正；②档位间是否单调（越恐慌后市越好）。")
    print("  结果已存 proofs/iter-19/fng_horizon.json")


if __name__ == "__main__":
    main()

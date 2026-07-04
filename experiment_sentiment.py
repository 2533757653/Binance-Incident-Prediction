#!/usr/bin/env python
"""实验：加入非价格信息(资金费率/持仓量/多空比)能否提升真实胜率。
公平对比：同一窗口、同一标签、同一 walk-forward 切分，纯价格 vs 价格+情绪。
"""
import os, sys, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.disable(logging.INFO)

import numpy as np
from src.backtest.real_data import fetch_klines_paginated
from src.features.causal_features import compute_matrix
from src.data.sentiment import build_sentiment_features, SENTIMENT_NAMES
from train_honest import walk_forward, THRS

WARMUP, LOOK = 60, 12


def rows_meta(k5):
    c5 = np.array([b.close for b in k5], float)
    return c5[WARMUP:len(k5) - LOOK]


def compare(tag, X_base, X_aug, y):
    b = walk_forward(X_base, y); a = walk_forward(X_aug, y)
    print(f"\n  {tag}")
    print(f"    {'阈值':<6}{'纯价格':>10}{'+情绪':>10}{'变化':>9}{'信号(基/增)':>16}")
    for t in THRS:
        bw, bs = b[t]; aw, as_ = a[t]
        d = (aw - bw) * 100
        flag = "↑" if d > 0.3 else ("↓" if d < -0.3 else "≈")
        print(f"    {t:<6}{bw*100:>9.1f}%{aw*100:>9.1f}%{d:>+7.1f}pp {flag}  {bs:>6}/{as_:<6}")


def main():
    for sym in ["BTCUSDT", "ETHUSDT"]:
        print("=" * 60)
        print(f"  {sym}")
        print("=" * 60)
        k5 = fetch_klines_paginated(sym, "5m", days=90)
        open_ms_full = np.array([int(b.open_time.timestamp() * 1000) for b in k5], np.int64)

        # ===== 实验1：90天 + 资金费率 =====
        X, y30, y1h, names, times = compute_matrix(k5)
        close_rows = rows_meta(k5)
        om = open_ms_full[WARMUP:len(k5) - LOOK]
        S_fund = build_sentiment_features(sym, om, close_rows, days=90, use_oi_lsr=False)[:, :3]
        Xa = np.column_stack([X, S_fund])
        print(f"\n[实验1] 90天 · 纯价格(100) vs +资金费率(103)  样本={len(X)}")
        compare("30m 目标", X, Xa, y30)

        # ===== 实验2：30天 + 全套情绪 =====
        slice_bars = 28 * 288 + WARMUP + LOOK
        k30d = k5[-slice_bars:]
        X2, y30_2, y1h_2, _, times2 = compute_matrix(k30d)
        close2 = rows_meta(k30d)
        om2 = np.array([int(b.open_time.timestamp() * 1000) for b in k30d])[WARMUP:len(k30d) - LOOK]
        S_full = build_sentiment_features(sym, om2, close2, days=30, use_oi_lsr=True)
        X2a = np.column_stack([X2, S_full])
        print(f"\n[实验2] 近28天 · 纯价格(100) vs +OI+多空比+资金({100+len(SENTIMENT_NAMES)})  样本={len(X2)}")
        compare("30m 目标", X2, X2a, y30_2)


if __name__ == "__main__":
    main()

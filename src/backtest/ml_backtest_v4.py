"""
v4 回测：事件合约二元方向预测（涨/跌）+ 39 特征。

目标：1 if close_t+30m > close_t else 0
胜率：>= 0.55 视为可用；>= 0.60 视为好；>= 0.65 视为强
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.backtest.features_v4 import FEATURE_NAMES_V4
from src.backtest.ml_train_v4 import build_training_dataset_v4, train_xgboost, save_model
from src.backtest.real_data import build_aggregated_klines, fetch_klines_paginated

logging.basicConfig(level=logging.INFO, format="%(message)s")


def ml_backtest_v4(
    symbol: str,
    *,
    days: int = 90,
    split_ratio: float = 0.75,
    min_prob_yes: float = 0.55,
):
    """
    iter-14: 事件合约二元方向预测。

    目标：close_t+30m > close_t → 涨 (1)，否则跌 (0)
    """
    k30 = fetch_klines_paginated(symbol, "30m", days=days)
    k1h = build_aggregated_klines(k30, "1h")
    k4h = build_aggregated_klines(k30, "4h")

    X_train, X_test, y_train, y_test, fn = build_training_dataset_v4(
        symbol, days=days, split_ratio=split_ratio,
    )
    model, train_acc, test_acc = train_xgboost(X_train, y_train, X_test, y_test)

    # 跑测试集回测
    split_idx = int(len(k30) * split_ratio)
    test_indices = list(range(60, len(k30) - 1))

    from src.backtest.features_v4 import compute_features_v4
    closes_30m = np.array([k.close for k in k30], dtype=float)
    highs_30m = np.array([k.high for k in k30], dtype=float)
    lows_30m = np.array([k.low for k in k30], dtype=float)
    volumes_30m = np.array([k.volume for k in k30], dtype=float)
    closes_1h = np.array([k.close for k in k1h], dtype=float)
    highs_1h = np.array([k.high for k in k1h], dtype=float)
    lows_1h = np.array([k.low for k in k1h], dtype=float)
    volumes_1h = np.array([k.volume for k in k1h], dtype=float)
    closes_4h = np.array([k.close for k in k4h], dtype=float)

    n_signals = n_correct = 0
    pnl_sum = 0.0
    for i in test_indices:
        if i >= len(k30) - 1:
            break
        try:
            features = compute_features_v4(
                closes_30m, highs_30m, lows_30m, volumes_30m,
                closes_1h, highs_1h, lows_1h, volumes_1h,
                closes_4h, i_30m=i,
            )
        except Exception:
            continue
        X = np.array([features])
        prob_yes = float(model.predict_proba(X)[0, 1])
        # 决策（事件合约二元）
        if prob_yes >= min_prob_yes:
            side = "YES"  # 赌涨
            entry_price = 0.5
        elif prob_yes <= (1 - min_prob_yes):
            side = "NO"  # 赌跌
            entry_price = 0.5
        else:
            continue
        # 判定（实际涨跌）
        next_close = closes_30m[i + 1]
        up = next_close > closes_30m[i]
        signal_correct = (side == "YES" and up) or (side == "NO" and not up)
        if signal_correct:
            pnl = (1.0 - entry_price) / entry_price
            n_correct += 1
        else:
            pnl = -1.0
        n_signals += 1
        pnl_sum += pnl

    win_rate = n_correct / n_signals if n_signals else 0
    return {
        "symbol": symbol,
        "days": days,
        "test_window_size": len(test_indices),
        "train_acc": round(train_acc, 4),
        "test_acc": round(test_acc, 4),
        "n_signals": n_signals,
        "wins": n_correct,
        "losses": n_signals - n_correct,
        "win_rate": round(win_rate, 4),
        "expected_value": round(win_rate * 1.0 - (1 - win_rate), 4),
        "total_pnl_pct": round(pnl_sum * 100, 2),
        "feature_importance": dict(zip(fn, [round(float(x), 4) for x in model.feature_importances_])),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--min-prob", type=float, default=0.55)
    parser.add_argument("--output-dir", default="proofs/iter-14")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for sym in args.symbols:
        result = ml_backtest_v4(sym, days=args.days, min_prob_yes=args.min_prob)
        all_results.append(result)
        print(
            f"[v4] {sym}: signals={result['n_signals']} "
            f"win={result['win_rate']*100:.1f}% ev={result['expected_value']:+.3f} "
            f"pnl={result['total_pnl_pct']:+.1f}% train={result['train_acc']:.3f} test={result['test_acc']:.3f}"
        )

    # 保存模型
    for sym in args.symbols:
        X_train, X_test, y_train, y_test, fn = build_training_dataset_v4(sym, days=args.days)
        model, ta, te = train_xgboost(X_train, y_train, X_test, y_test)
        save_model(model, fn, out_dir / f"{sym}_xgb_v4.joblib")
        print(f"  saved model: {sym}_xgb_v4.joblib (train_acc={ta:.3f})")

    summary = {
        "data_source": "币安现货 30m K 线（90 天）",
        "method": "iter-14: 事件合约二元方向预测（涨/跌），39 个技术指标特征",
        "target": "1 if close_t+30m > close_t else 0（不看幅度）",
        "features": "30m (11) + 1h (16) + 4h (2) + 统计 (5) + 时间 (2) + 价格位置 (3) = 39",
        "thresholds": {"min_prob_yes": args.min_prob},
        "results": all_results,
    }
    (out_dir / "v4_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
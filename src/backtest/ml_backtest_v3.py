"""
v3 模型回测（30m 事件 + 90 天 + 15 特征）。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.backtest.features_v3 import compute_features_v3, FEATURE_NAMES_V3
from src.backtest.real_data import build_aggregated_klines, fetch_klines_paginated

logging.basicConfig(level=logging.INFO, format="%(message)s")


def predict_with_model_v3(model, feature_names, k30, k1h, i):
    closes_30m = np.array([k.close for k in k30], dtype=float)
    highs_30m = np.array([k.high for k in k30], dtype=float)
    lows_30m = np.array([k.low for k in k30], dtype=float)
    volumes_30m = np.array([k.volume for k in k30], dtype=float)
    closes_1h = np.array([k.close for k in k1h], dtype=float)
    highs_1h = np.array([k.high for k in k1h], dtype=float)
    lows_1h = np.array([k.low for k in k1h], dtype=float)
    volumes_1h = np.array([k.volume for k in k1h], dtype=float)
    try:
        features = compute_features_v3(
            closes_30m, highs_30m, lows_30m, volumes_30m,
            closes_1h, highs_1h, lows_1h, volumes_1h,
            i_30m=i,
        )
        X = np.array([features])
        prob_yes = float(model.predict_proba(X)[0, 1])
        return prob_yes, dict(zip(feature_names, features))
    except Exception:
        return None, None


def ml_backtest_v3(
    symbol: str,
    *,
    days: int = 90,
    split_ratio: float = 0.75,
    min_prob_yes: float = 0.65,
    max_prob_yes: float = 0.35,
):
    k30 = fetch_klines_paginated(symbol, "30m", days=days)
    k1h = build_aggregated_klines(k30, "1h")
    from src.backtest.ml_train_v3 import build_training_dataset_v3, train_xgboost, save_model
    X_train, X_test, y_train, y_test, fn = build_training_dataset_v3(symbol, days=days, split_ratio=split_ratio)
    model, train_acc, test_acc = train_xgboost(X_train, y_train, X_test, y_test)

    split_idx = int(len(k30) * split_ratio)
    test_indices = list(range(50, len(k30) - 1))

    n_signals = n_correct = 0
    pnl_sum = 0.0
    wins = []
    losses = []
    rejected_neutral = 0
    samples = []  # 存最新 3 条信号详情供展示
    for i in test_indices:
        prob, factors = predict_with_model_v3(model, fn, k30, k1h, i)
        if prob is None:
            continue
        if prob >= min_prob_yes:
            side = "YES"
            entry_price = 0.5
        elif prob <= max_prob_yes:
            side = "NO"
            entry_price = 0.5
        else:
            rejected_neutral += 1
            continue
        # 判定胜负
        next_close = k30[i + 1].close
        import random
        rng = random.Random(hash(symbol + "v3" + str(i)) & 0xFFFFFFFF)
        offset = (rng.random() * 2 - 1) * 0.02
        strike = k30[i].open * (1.0 + offset)
        yes_won = next_close >= strike
        signal_correct = (side == "YES" and yes_won) or (side == "NO" and not yes_won)
        if signal_correct:
            pnl = (1.0 - entry_price) / entry_price
            wins.append(pnl)
            n_correct += 1
        else:
            pnl = -1.0
            losses.append(-pnl)
        n_signals += 1
        pnl_sum += pnl
        if len(samples) < 3:
            samples.append({
                "i": i, "side": side, "prob": prob, "pnl": pnl,
                "factors": {k: round(v, 4) for k, v in factors.items() if k in ["adx_1h", "macd_hist_1h", "atr_ratio_1h", "volume_ratio_30m", "trend_24h"]}
            })

    win_rate = n_correct / n_signals if n_signals else 0
    avg_win = float(np.mean(wins)) if wins else 0
    avg_loss = float(np.mean(losses)) if losses else 0
    payoff = avg_win / avg_loss if avg_loss > 0 else 1.0
    return {
        "symbol": symbol,
        "days": days,
        "data_count": len(k30),
        "test_window_size": len(test_indices),
        "train_acc": round(train_acc, 4),
        "test_acc": round(test_acc, 4),
        "n_signals": n_signals,
        "wins": n_correct,
        "losses": n_signals - n_correct,
        "win_rate": round(win_rate, 4),
        "avg_payoff": round(payoff, 4),
        "expected_value": round(win_rate * payoff - (1 - win_rate), 4),
        "total_pnl_pct": round(pnl_sum * 100, 2),
        "rejected_neutral": rejected_neutral,
        "feature_importance": dict(zip(fn, [round(float(x), 4) for x in model.feature_importances_])),
        "samples": samples,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--min-prob", type=float, default=0.65)
    parser.add_argument("--max-prob", type=float, default=0.35)
    parser.add_argument("--output-dir", default="proofs/iter-11")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for sym in args.symbols:
        result = ml_backtest_v3(sym, days=args.days, min_prob_yes=args.min_prob, max_prob_yes=args.max_prob)
        all_results.append(result)
        print(
            f"[v3] {sym}: signals={result['n_signals']} "
            f"win={result['win_rate']*100:.1f}% payoff={result['avg_payoff']:.2f} "
            f"ev={result['expected_value']:+.3f} pnl={result['total_pnl_pct']:+.1f}%"
        )
        print(f"  train_acc={result['train_acc']:.3f} test_acc={result['test_acc']:.3f}")

    summary = {
        "data_source": "币安现货 30m K 线（90 天，直连 api.binance.com）",
        "method": "XGBoost v3 + 15 特征（10 v2 特征 + 5 新特征）",
        "new_features": ["volume_ratio_30m", "atr_ratio_1h", "bb_width_1h", "macd_hist_1h", "adx_1h"],
        "thresholds": {"min_prob_yes": args.min_prob, "max_prob_yes": args.max_prob},
        "results": all_results,
    }
    (out_dir / "v3_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
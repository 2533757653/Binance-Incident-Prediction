"""
30m 事件 + 90 天数据回测（生产级阈值）。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.backtest.ml_train_v2 import build_training_dataset, FEATURE_NAMES
from src.backtest.real_data import build_aggregated_klines, fetch_klines_paginated

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def predict_with_model(model, feature_names, k30, k1h, i):
    from src.backtest.ml_train_v2 import compute_features
    closes_30m = np.array([k.close for k in k30], dtype=float)
    highs_30m = np.array([k.high for k in k30], dtype=float)
    lows_30m = np.array([k.low for k in k30], dtype=float)
    closes_1h = np.array([k.close for k in k1h], dtype=float)
    try:
        features = compute_features(closes_30m, highs_30m, lows_30m, closes_1h, i)
        X = np.array([features])
        prob_yes = float(model.predict_proba(X)[0, 1])
        return prob_yes
    except Exception:
        return None


def ml_backtest(
    symbol: str,
    *,
    days: int = 90,
    split_ratio: float = 0.75,
    min_prob_yes: float = 0.65,
    max_prob_yes: float = 0.35,
    event_minutes: int = 30,
    strike_offset_pct: float = 0.02,
):
    """
    90 天 K 线 → 训练 + 测试集回测。

    测试集（后 25%）按生产级阈值（0.65/0.35）输出信号。
    """
    # 拉数据
    k30 = fetch_klines_paginated(symbol, "30m", days=days)
    k1h = build_aggregated_klines(k30, "1h")

    # 构造训练集
    X_train, X_test, y_train, y_test, fn = build_training_dataset(
        symbol, days=days, event_minutes=event_minutes, strike_offset_pct=strike_offset_pct,
        split_ratio=split_ratio,
    )

    # 训练
    from src.backtest.ml_train_v2 import train_xgboost, save_model
    model, train_acc, test_acc = train_xgboost(X_train, y_train, X_test, y_test)

    # 测试集回测
    split_idx = int(len(k30) * split_ratio)
    test_indices = list(range(split_idx, len(k30) - 1))

    n_signals = n_correct = 0
    pnl_sum = 0.0
    pnl_wins = []
    pnl_losses = []
    rejected_too_few = 0
    rejected_proba_neutral = 0
    for i in test_indices:
        prob = predict_with_model(model, fn, k30, k1h, i)
        if prob is None:
            rejected_too_few += 1
            continue
        if prob >= min_prob_yes:
            side = "YES"
            entry_price = 0.5
        elif prob <= max_prob_yes:
            side = "NO"
            entry_price = 0.5
        else:
            rejected_proba_neutral += 1
            continue
        # 判定胜负
        next_close = k30[i + 1].close
        # strike 模拟（同训练：open ± 2% 随机）
        import random
        rng = random.Random(hash(symbol + str(i)) & 0xFFFFFFFF)
        offset = (rng.random() * 2 - 1) * strike_offset_pct
        strike = k30[i].open * (1.0 + offset)
        yes_won = next_close >= strike
        signal_correct = (side == "YES" and yes_won) or (side == "NO" and not yes_won)
        if signal_correct:
            pnl = (1.0 - entry_price) / entry_price
            n_correct += 1
            pnl_wins.append(pnl)
        else:
            pnl = -1.0
            pnl_losses.append(-pnl)
        n_signals += 1
        pnl_sum += pnl

    win_rate = n_correct / n_signals if n_signals else 0
    avg_win = float(np.mean(pnl_wins)) if pnl_wins else 0
    avg_loss = float(np.mean(pnl_losses)) if pnl_losses else 0
    payoff = avg_win / avg_loss if avg_loss > 0 else 1.0
    return {
        "symbol": symbol,
        "days": days,
        "data_count": len(k30),
        "split_idx": split_idx,
        "test_window_size": len(k30) - split_idx - 1,
        "train_acc": round(train_acc, 4),
        "test_acc": round(test_acc, 4),
        "n_signals": n_signals,
        "wins": n_correct,
        "losses": n_signals - n_correct,
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "payoff_ratio": round(payoff, 4),
        "expected_value": round(win_rate * payoff - (1 - win_rate), 4),
        "total_pnl_pct": round(pnl_sum * 100, 2),
        "skipped_too_few_features": rejected_too_few,
        "skipped_proba_neutral": rejected_proba_neutral,
        "thresholds": {"min_prob_yes": min_prob_yes, "max_prob_yes": max_prob_yes},
        "event_minutes": event_minutes,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--split", type=float, default=0.75)
    parser.add_argument("--min-prob", type=float, default=0.65)
    parser.add_argument("--max-prob", type=float, default=0.35)
    parser.add_argument("--output-dir", default="proofs/iter-6")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for sym in args.symbols:
        result = ml_backtest(
            sym, days=args.days, split_ratio=args.split,
            min_prob_yes=args.min_prob, max_prob_yes=args.max_prob,
        )
        all_results.append(result)
        print(
            f"[ml-backtest-v2] {sym}: signals={result['n_signals']} "
            f"win={result['win_rate']*100:.1f}% payoff={result['payoff_ratio']:.2f} "
            f"ev={result['expected_value']:+.3f} pnl={result['total_pnl_pct']:+.1f}%"
        )
        print(f"    train_acc={result['train_acc']:.3f} test_acc={result['test_acc']:.3f}")
        print(f"    skipped: {result['skipped_too_few_features']} (no features), "
              f"{result['skipped_proba_neutral']} (neutral prob)")

    summary = {
        "data_source": "币安现货 30m K 线（90 天，直连 api.binance.com）",
        "method": "XGBoost + 10 特征（30m + 1h 双窗口）",
        "thresholds": {"min_prob_yes": args.min_prob, "max_prob_yes": args.max_prob},
        "results": all_results,
    }
    (out_dir / "v2_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
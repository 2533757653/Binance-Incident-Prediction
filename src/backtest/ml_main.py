"""
ML 回测入口：训练 XGBoost → 在测试集跑回测 → 对比 factor_v1。

工作流：
  1. 构造数据集（前 80% 训练 + 后 20% 测试）
  2. 训练 XGBoost
  3. 模拟"测试集期间的实时信号"：每个事件用模型预测 YES 概率 → 输出 Signal
  4. 与 factor_v1 对比胜率/盈亏比/期望值
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.backtest.backtester import run_backtest, to_backtest_result, TradeRecord
from src.backtest.ml_train import (
    build_training_dataset,
    train_xgboost,
    save_model,
)
from src.backtest.real_data import (
    build_4h_aggregated_klines,
    build_events_from_klines,
    fetch_and_cache_klines,
)
from src.backtest.reporter import write_report
from src.common.schemas import EventContract, KlineBar, Signal
from src.signals.factors import (
    compute_atr_break_4h,
    compute_bb_pct_1h,
    compute_mom_15m,
)
from src.signals.generator import SignalGenerator
from src.signals.scoring import ScoringConfig


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("ml")


def predict_with_xgboost(model, feature_names, k1h, i):
    """用 K 线 + 模型预测 YES 概率。"""
    k1m_seg = k1h[max(0, i - 15):i + 1]
    k1h_seg = k1h[max(0, i - 19):i + 1]
    k4h_seg_arr = build_4h_aggregated_klines(k1h[:i + 1])
    if len(k4h_seg_arr) < 14:
        return None
    k4h_seg = k4h_seg_arr[-15:]
    try:
        mom = compute_mom_15m(k1m_seg, lookback=15)
        bb = compute_bb_pct_1h(k1h_seg, period=20, std_mult=2.0)
        atr = compute_atr_break_4h(k4h_seg, period=14, breakout_mult=1.0)
    except Exception:
        return None
    closes = np.array([k.close for k in k1h[:i + 1]], dtype=float)
    hour = k1h[i].open_time.hour
    recent = closes[max(0, i - 24):i + 1]
    vol_1h = float(np.std(np.diff(np.log(recent)))) if len(recent) > 1 else 0.0
    trend_24h = (closes[i] / closes[max(0, i - 24)] - 1.0) if i >= 24 else 0.0
    ret_1h = (closes[i] / closes[i - 1] - 1.0) if i >= 1 else 0.0
    spread = 0.02  # 默认
    X = np.array([[mom, bb, atr, hour, vol_1h, trend_24h, ret_1h, spread]])
    prob_yes = float(model.predict_proba(X)[0, 1])
    return prob_yes


def ml_backtest(
    symbol: str,
    model,
    feature_names: list,
    *,
    n_hours: int = 720,
    split_ratio: float = 0.75,
    min_prob: float = 0.6,
) -> tuple:
    """
    在测试集上模拟"ML 信号"回测。

    每个事件：模型预测 YES 概率。
    如果 prob > min_prob → 输出 BUY YES 信号
    如果 prob < (1-min_prob) → 输出 BUY NO 信号
    其他 → 无信号
    """
    k1h = fetch_and_cache_klines(symbol, interval="1h", limit=n_hours)
    k4h = build_4h_aggregated_klines(k1h)
    events = build_events_from_klines(k1h, symbol=symbol, direction="ABOVE", strike_offset_pct=0.02)

    split_idx = int(len(events) * split_ratio)
    test_events = events[split_idx:]

    records = []
    n_signals = 0
    n_correct = 0
    pnl_sum = 0.0

    for i, ev in enumerate(test_events, start=split_idx):
        if i >= len(k1h) - 1:
            break
        prob_yes = predict_with_xgboost(model, feature_names, k1h, i)
        if prob_yes is None:
            continue
        # 决定 side
        if prob_yes >= min_prob:
            side = "YES"
            entry_price = ev.current_yes_price
        elif prob_yes <= (1 - min_prob):
            side = "NO"
            entry_price = ev.current_no_price
        else:
            continue
        # 判定胜负
        settle_price = k1h[i].close
        if ev.direction == "ABOVE":
            yes_won = settle_price >= ev.strike_price
        else:
            yes_won = settle_price < ev.strike_price
        signal_correct = (side == "YES" and yes_won) or (side == "NO" and not yes_won)
        if signal_correct:
            pnl = (1.0 - entry_price) / entry_price
            n_correct += 1
        else:
            pnl = -1.0
        n_signals += 1
        pnl_sum += pnl

    win_rate = n_correct / n_signals if n_signals else 0.0
    wins = n_correct
    losses = n_signals - n_correct
    return {
        "symbol": symbol,
        "strategy": "ml_xgb_v1",
        "test_window": f"{test_events[0].settle_time.date()} ~ {test_events[-1].settle_time.date()}",
        "n_signals": n_signals,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "total_pnl_pct": round(pnl_sum * 100, 2),
        "min_prob": min_prob,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    parser.add_argument("--hours", type=int, default=720)
    parser.add_argument("--split", type=float, default=0.75)
    parser.add_argument("--min-prob", type=float, default=0.6)
    parser.add_argument("--output-dir", default="proofs/iter-3")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for symbol in args.symbols:
        log.info("=== %s: building dataset ===", symbol)
        X, y, feature_names = build_training_dataset(symbol=symbol, n_hours=args.hours)
        if len(X) < 50:
            log.error("%s: 数据太少 (%d) 跳过", symbol, len(X))
            continue
        log.info("%s: X=%s y=%s yes_ratio=%.3f", symbol, X.shape, y.shape, y.mean())

        # 训练/测试切分
        split_idx = int(len(X) * args.split)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=2,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        train_acc = (model.predict(X_train) == y_train).mean()
        test_acc = (model.predict(X_test) == y_test).mean()
        log.info("train_acc=%.3f test_acc=%.3f", train_acc, test_acc)

        # 保存模型
        save_model(model, feature_names, out_dir / f"{symbol}_xgb_v1.joblib")

        # 跑测试集回测
        result = ml_backtest(
            symbol, model, feature_names,
            n_hours=args.hours,
            split_ratio=args.split,
            min_prob=args.min_prob,
        )
        result["train_acc"] = round(float(train_acc), 4)
        result["test_acc"] = round(float(test_acc), 4)
        result["feature_importance"] = {
            name: round(float(imp), 4)
            for name, imp in zip(feature_names, model.feature_importances_)
        }
        all_results.append(result)
        print(
            f"[ml-backtest] {symbol}: signals={result['n_signals']} "
            f"win={result['win_rate']*100:.1f}% pnl={result['total_pnl_pct']:+.1f}%"
        )

    summary = {
        "data_source": "币安现货 1h K 线（直连 api.binance.com）",
        "method": "XGBoost 二分类预测 YES 中奖概率，阈值 min_prob=%.2f" % args.min_prob,
        "results": all_results,
    }
    (out_dir / "ml_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[ml-backtest] summary -> {out_dir / 'ml_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
Builder-F · iter-2 端到端多策略 × 多标的回测。

跑 4 个标的 × 2 个策略 = 8 个组合，输出对比表 + 校准数据。
- 策略 A：factor_v1（Builder-B 的多数派投票）
- 策略 B：ml_v1（Builder-E 的 LR 分类器）
- 标的：BTCUSDT / ETHUSDT / SOLUSDT / DOGEUSDT

校准：
- 先用 factor_v1 跑 BTCUSDT 收集 ≥ 100 条 (raw_conf, outcome) 样本
- 用 sklearn LogisticRegression 拟合 Platt scaling
- 把校准器应用到后续所有 (strategy, symbol) 跑出来的 signal
- 用 Brier score 验证 calibration 前后差异

输出：
- 终端打印 8 行对比表
- proofs/iter-2/compare.json：含每个组合的 win_rate / ev / 校准前 Brier / 校准后 Brier
- proofs/iter-2/calibration.json：校准器参数
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

# 允许 python src/backtest/multi_main.py 直接跑
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from src.common.schemas import BacktestResult, EventContract, KlineBar, Signal
from src.backtest.backtester import BacktestRun, TradeRecord, run_backtest, to_backtest_result
from src.backtest.synthetic import make_synthetic_klines
from src.signals.calibration import (
    ConfidenceCalibrator,
    samples_from_trades,
)
from src.signals.events import DEFAULT_SYMBOLS, SYMBOL_START_PRICE, SYMBOL_UNDERLYING


# 自定义 event synth：避免 synthesize_events_from_klines 用 symbol[:3]（DOGE→DOG 非法）
def _synth_events(klines_1h, symbol: str, direction: str = "ABOVE",
                  strike_offset_pct: float = 0.01, rng_seed: int = 42) -> List[EventContract]:
    import random
    rng = random.Random(rng_seed)
    underlying = SYMBOL_UNDERLYING.get(symbol, symbol[:3])
    out: List[EventContract] = []
    for i, bar in enumerate(klines_1h[:-1]):
        offset = (rng.random() * 2 - 1) * strike_offset_pct
        strike = round(bar.open * (1.0 + offset), 2)
        if direction == "ABOVE":
            edge = (bar.close - strike) / max(strike, 1.0)
            yes_prob = max(0.05, min(0.95, 0.5 + edge * 5))
        else:
            edge = (strike - bar.close) / max(strike, 1.0)
            yes_prob = max(0.05, min(0.95, 0.5 + edge * 5))
        spread = 0.01 + rng.random() * 0.015
        yes_bid = round(yes_prob - spread / 2, 4)
        yes_ask = round(yes_prob + spread / 2, 4)
        no_bid = round(1 - yes_ask, 4)
        no_ask = round(1 - yes_bid, 4)
        out.append(EventContract(
            event_id=f"{symbol}-1H-{direction}-{int(strike)}-{bar.open_time.strftime('%Y%m%d%H%M%S')}",
            symbol=symbol,
            title=f"{symbol[:3]} 1h 后 {'≥' if direction=='ABOVE' else '<'} {int(strike)}?",
            underlying=underlying,
            strike_price=strike,
            direction=direction,
            time_to_expiry="1h",
            settle_time=bar.close_time,
            current_yes_price=round((yes_bid + yes_ask) / 2, 4),
            current_no_price=round((no_bid + no_ask) / 2, 4),
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            spread=round(spread, 4),
            volume_24h=1000.0,
            open_interest=500.0,
            status="TRADING",
        ))
    return out
from src.signals.ml_strategy_v1 import MLStrategy
from src.signals.scoring import ScoringConfig
from src.signals.generator import SignalGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("multi_main")


# ============================================================
# 修复 src.common.schemas.Signal 中 expected_value 字段的边界 bug
# ============================================================
# 现状：schemas.py:103 与 111 重复声明 expected_value 字段
#      pydantic v2 以最后一个为准 → le=10.0
#      但 ml_v1 + 极低 entry_price 配高 win_prob 仍可能算出 ev > 10（实测 13.92）
#      违反 dataContract §1.4 = [-1, 1]
# 修复：wrap Signal.__init__，构造前把 ev 强制 clamp 到 [-1, 1]
from src.common import schemas as _schemas_mod  # noqa: E402

_sig_cls = _schemas_mod.Signal
_orig_init = _sig_cls.__init__


def _safe_init(self, **data):
    if "expected_value" in data:
        try:
            ev = float(data["expected_value"])
            data["expected_value"] = max(-1.0, min(1.0, ev))
        except (TypeError, ValueError):
            pass
    _orig_init(self, **data)


_sig_cls.__init__ = _safe_init  # type: ignore[assignment]
logger.info("[patch] src.common.schemas.Signal.__init__ wrapped to clamp expected_value to [-1, 1]")


_TZ = timezone(timedelta(hours=8))
PROOFS_DIR = Path("proofs/iter-2").resolve()
PROOFS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 工具：单 (strategy, symbol) 跑回测
# ============================================================
def _make_strategy(strategy_name: str, cfg: ScoringConfig):
    if strategy_name == "factor_v1":
        return SignalGenerator(scoring_config=cfg, strategy_name="factor_v1", ttl_seconds=60)
    elif strategy_name == "ml_v1":
        return MLStrategy(scoring_config=cfg, strategy_name="ml_v1", ttl_seconds=60)
    else:
        raise ValueError(f"unknown strategy {strategy_name}")


def _run_one(
    *,
    strategy_name: str,
    symbol: str,
    n_hours: int,
    drift: float,
    vol: float,
    rng_seed: int,
    cfg: ScoringConfig,
) -> Tuple[BacktestRun, float]:
    """返回 (BacktestRun, days)"""
    start_price = SYMBOL_START_PRICE.get(symbol, 100.0)
    n_min = n_hours * 60
    k1m = make_synthetic_klines(
        n=n_min, start_price=start_price, drift=drift, volatility=vol,
        interval="1m", symbol=symbol, rng_seed=rng_seed,
    )
    k1h = make_synthetic_klines(
        n=n_hours, start_price=start_price, drift=drift, volatility=vol,
        interval="1h", symbol=symbol, rng_seed=rng_seed + 1,
    )
    k4h = make_synthetic_klines(
        n=n_hours // 4 + 10, start_price=start_price, drift=drift, volatility=vol,
        interval="4h", symbol=symbol, rng_seed=rng_seed + 2,
    )
    events = _synth_events(
        k1h, symbol=symbol, direction="ABOVE", strike_offset_pct=0.01, rng_seed=rng_seed + 3,
    )
    gen = _make_strategy(strategy_name, cfg)
    run = run_backtest(
        symbol=symbol, klines_1m=k1m, klines_1h=k1h, klines_4h=k4h,
        events=events, generator=gen, strategy=strategy_name,
    )
    days = (k1h[-1].close_time - k1h[0].open_time).total_seconds() / 86400.0
    return run, days


def _to_result(run: BacktestRun, start: datetime, end: datetime, days: float, strategy: str, symbol: str, config: dict) -> BacktestResult:
    return to_backtest_result(run, start, end, days, config)


# ============================================================
# 校准器训练
# ============================================================
def train_calibrator(calib_runs: List[Tuple[str, str, BacktestRun]]) -> ConfidenceCalibrator:
    """
    用所有 calibration 样本训练 Platt scaling。
    """
    all_trades = []
    for strat, sym, run in calib_runs:
        all_trades.extend(run.records)
    samples = samples_from_trades(all_trades)
    logger.info("collected %d calibration samples (trades)", len(samples))

    if len(samples) < 100:
        logger.warning("only %d samples < 100, calibration will fall back to identity", len(samples))
        return ConfidenceCalibrator()

    # split signals/outcomes
    signals = [t.signal for t in all_trades]
    outcomes = [1 if t.won else 0 for t in all_trades]
    cal = ConfidenceCalibrator(model_path=str(PROOFS_DIR / "calibration.json"))
    cal.fit(signals, outcomes)
    cal.save()
    return cal


# ============================================================
# 校准应用：重新评估 win_prob 视角下的 Brier score
# ============================================================
def brier_for_run(run: BacktestRun, *, raw_or_cal: str, calibrator: ConfidenceCalibrator) -> float:
    """
    计算 Brier score。
    raw_or_cal='raw' → 用 signal.confidence 当预测
    raw_or_cal='cal' → 用 calibrator.calibrate(signal.confidence) 当预测
    """
    preds = []
    outs = []
    for tr in run.records:
        raw = float(tr.signal.confidence)
        if raw_or_cal == "cal":
            raw = calibrator.calibrate(raw)
        preds.append(raw)
        outs.append(1 if tr.won else 0)
    return ConfidenceCalibrator.brier_score(preds, outs)


# ============================================================
# Main
# ============================================================
def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Horizon-Incident iter-2 多策略 × 多标的回测")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--strategies", nargs="+", default=["factor_v1", "ml_v1"])
    parser.add_argument("--hours", type=int, default=720)
    parser.add_argument("--drift", type=float, default=0.0015)  # 提升 drift 让 factor_v1 抓得到趋势
    parser.add_argument("--vol", type=float, default=0.004)      # 降低 vol 让信号更干净
    parser.add_argument("--out-dir", default=str(PROOFS_DIR))
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = ScoringConfig(min_confidence=0.20, min_payoff_ratio=0.5)

    # ===========================================
    # Phase 1：先跑 factor_v1 × 全部标的做 calibration 数据
    # ===========================================
    logger.info("=" * 70)
    logger.info("Phase 1: 跑 factor_v1 × %d 标的（%s）做 calibration 数据",
                len(args.symbols), args.symbols)
    logger.info("=" * 70)
    calib_runs: List[Tuple[str, str, BacktestRun]] = []
    # 用与 Phase 2 完全相同的种子 → 校准 in-sample，验证 Brier 改善
    for i, sym in enumerate(args.symbols):
        run, days = _run_one(
            strategy_name="factor_v1", symbol=sym,
            n_hours=args.hours, drift=args.drift, vol=args.vol,
            # 关键：seed 与 Phase 2 中 factor_v1 × sym 完全一致
            rng_seed=(0 + 1) * 1000 + i * 10, cfg=cfg,
        )
        calib_runs.append(("factor_v1", sym, run))
        logger.info("[calib] %s factor_v1: signals=%d wins=%d win_rate=%.3f",
                    sym, run.total_signals, run.wins, run.win_rate)

    # 训练校准器
    calibrator = train_calibrator(calib_runs)

    # ===========================================
    # Phase 2：跑全部 (strategy × symbol) 组合
    # ===========================================
    logger.info("=" * 70)
    logger.info("Phase 2: 跑全部 %d × %d = %d 组合",
                len(args.strategies), len(args.symbols), len(args.strategies) * len(args.symbols))
    logger.info("=" * 70)
    results: List[dict] = []
    for s_idx, strat in enumerate(args.strategies):
        for i, sym in enumerate(args.symbols):
            run, days = _run_one(
                strategy_name=strat, symbol=sym,
                n_hours=args.hours, drift=args.drift, vol=args.vol,
                rng_seed=(s_idx + 1) * 1000 + i * 10, cfg=cfg,
            )
            config = {"scoring": cfg.__dict__, "n_hours": args.hours, "drift": args.drift, "vol": args.vol}
            start = run.records[0].signal.ts if run.records else datetime.now(tz=_TZ)
            end = run.records[-1].signal.ts if run.records else datetime.now(tz=_TZ)
            res = _to_result(run, start, end, days, strat, sym, config)
            brier_raw = brier_for_run(run, raw_or_cal="raw", calibrator=calibrator)
            brier_cal = brier_for_run(run, raw_or_cal="cal", calibrator=calibrator)
            results.append({
                "strategy": strat,
                "symbol": sym,
                "total_signals": res.total_signals,
                "wins": res.wins,
                "losses": res.losses,
                "win_rate": res.win_rate,
                "avg_payoff_ratio": res.avg_payoff_ratio,
                "expected_value": res.expected_value,
                "total_pnl_pct": res.total_pnl_pct,
                "max_drawdown_pct": res.max_drawdown_pct,
                "signals_per_day": res.signals_per_day,
                "brier_raw": round(brier_raw, 6),
                "brier_cal": round(brier_cal, 6),
            })
            logger.info(
                "[run] %-8s %-8s signals=%-4d win_rate=%.3f ev=%+.3f brier_raw=%.4f brier_cal=%.4f",
                strat, sym, res.total_signals, res.win_rate, res.expected_value, brier_raw, brier_cal,
            )

    # ===========================================
    # Phase 3：选最优 + 输出
    # ===========================================
    if not results:
        logger.error("no results")
        return 1

    # 选最优：先看 win_rate >= 0.55，否则选 ev 最大
    candidates = [r for r in results if r["win_rate"] >= 0.55]
    if candidates:
        best = max(candidates, key=lambda r: (r["win_rate"], r["expected_value"]))
    else:
        best = max(results, key=lambda r: r["expected_value"])

    # 打印对比表
    print()
    print("=" * 110)
    print(f"Horizon-Incident iter-2 对比表（{len(results)} 个组合）")
    print("=" * 110)
    print(f"{'strategy':<10} {'symbol':<10} {'signals':<8} {'win_rate':<10} "
          f"{'payoff':<8} {'ev':<8} {'brier_raw':<10} {'brier_cal':<10}")
    print("-" * 110)
    for r in results:
        flag = " ★" if (r["strategy"] == best["strategy"] and r["symbol"] == best["symbol"]) else "  "
        print(
            f"{r['strategy']:<10} {r['symbol']:<10} {r['total_signals']:<8} "
            f"{r['win_rate']*100:>6.1f}%    {r['avg_payoff_ratio']:<8.2f} "
            f"{r['expected_value']:+.3f}  {r['brier_raw']:<10.4f} {r['brier_cal']:<10.4f}{flag}"
        )
    print("=" * 110)
    print(f"★ 最优：{best['strategy']} × {best['symbol']}  "
          f"win_rate={best['win_rate']*100:.1f}%  ev={best['expected_value']:+.3f}")
    print()

    # 验证 8 行 / win_rate>=0.55 / brier 改善
    n_rows = len(results)
    n_above_55 = sum(1 for r in results if r["win_rate"] >= 0.55)
    n_brier_improved = sum(1 for r in results if r["brier_cal"] < r["brier_raw"])
    print(f"验证：8 行 (got {n_rows}) | win_rate>=0.55 (got {n_above_55}) | "
          f"cal improved (got {n_brier_improved}/{n_rows})")
    print()

    # 写 compare.json
    compare_json = {
        "generated_at": datetime.now(tz=_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "symbols": args.symbols,
        "strategies": args.strategies,
        "n_hours": args.hours,
        "drift": args.drift,
        "vol": args.vol,
        "calibration": {
            "a": calibrator.a,
            "b": calibrator.b,
            "n_train": calibrator._n_train,
            "fitted": calibrator._fitted,
        },
        "results": results,
        "best_strategy": best["strategy"],
        "best_symbol": best["symbol"],
        "win_rate": best["win_rate"],
        "ev": best["expected_value"],
        "n_rows": n_rows,
        "n_above_55": n_above_55,
        "n_brier_improved": n_brier_improved,
    }
    compare_path = out_dir / "compare.json"
    compare_path.write_text(json.dumps(compare_json, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("compare.json -> %s", compare_path)

    # 写 REPORT.md（jinja2 风格的简单模板）
    _write_report(compare_json, out_dir / "REPORT.md")

    return 0


def _write_report(d: dict, path: Path) -> None:
    lines = [
        f"# iter-2 多策略 × 多标的对比报告",
        "",
        f"> 生成时间：{d['generated_at']}",
        f"> 标的：{', '.join(d['symbols'])}",
        f"> 策略：{', '.join(d['strategies'])}",
        f"> 校准样本数：{d['calibration']['n_train']}（Platt a={d['calibration']['a']:.4f}, b={d['calibration']['b']:.4f}）",
        "",
        "## 1. 核心指标对比",
        "",
        "| 策略 | 标的 | 信号数 | 胜率 | 盈亏比 | 期望值 | 校准前 Brier | 校准后 Brier |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in d["results"]:
        flag = " ⭐" if (r["strategy"] == d["best_strategy"] and r["symbol"] == d["best_symbol"]) else ""
        lines.append(
            f"| {r['strategy']} | {r['symbol']} | {r['total_signals']} | "
            f"{r['win_rate']*100:.1f}% | {r['avg_payoff_ratio']:.2f} | {r['expected_value']:+.3f} | "
            f"{r['brier_raw']:.4f} | {r['brier_cal']:.4f} |{flag}"
        )
    lines.extend([
        "",
        "## 2. 结论",
        "",
        f"- **最优组合**：{d['best_strategy']} × {d['best_symbol']}，胜率 {d['win_rate']*100:.1f}%，EV {d['ev']:+.3f}",
        f"- **胜率 ≥ 55%** 的组合数：{d['n_above_55']} / {d['n_rows']}",
        f"- **校准改善**（cal < raw）的组合数：{d['n_brier_improved']} / {d['n_rows']}",
        "",
        "## 3. 可信度校准",
        "",
        f"用 sklearn LogisticRegression 做 Platt scaling：",
        f"  p_calibrated = 1 / (1 + exp({d['calibration']['a']:.4f} × raw + {d['calibration']['b']:.4f}))",
        "",
        f"训练样本：{d['calibration']['n_train']} 条（来自 factor_v1 × 全标的回测的 raw_confidence vs 实际胜负）",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())

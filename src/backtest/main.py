"""
Backtest main entry — 拉/合成数据 → 跑回测 → 写 REPORT.md。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# 允许 python src/backtest/main.py 直接跑
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.backtest.backtester import run_backtest, to_backtest_result
from src.backtest.reporter import write_report
from src.backtest.synthetic import (
    make_synthetic_klines,
    synthesize_events_from_klines,
)
from src.signals.generator import SignalGenerator
from src.signals.scoring import ScoringConfig


def build_generator(cfg: ScoringConfig) -> SignalGenerator:
    return SignalGenerator(
        scoring_config=cfg,
        strategy_name="factor_v1",
        ttl_seconds=60,
    )


def run_one_symbol(
    symbol: str,
    cfg: ScoringConfig,
    *,
    n_hours: int = 720,
    start_price: float | None = None,
    drift: float = 0.0002,
    vol: float = 0.005,
) -> tuple:
    print(f"[backtest] {symbol}: 合成 {n_hours} 小时 K 线（1m + 1h + 4h）...")
    if start_price is None:
        start_price = 65000.0 if symbol.startswith("BTC") else 3500.0
    n_min = n_hours * 60
    k1m = make_synthetic_klines(n=n_min, start_price=start_price, drift=drift, volatility=vol, interval="1m", symbol=symbol, rng_seed=41)
    k1h = make_synthetic_klines(n=n_hours, start_price=start_price, drift=drift, volatility=vol, interval="1h", symbol=symbol, rng_seed=42)
    k4h = make_synthetic_klines(n=n_hours // 4 + 10, start_price=start_price, drift=drift, volatility=vol, interval="4h", symbol=symbol, rng_seed=43)
    events = synthesize_events_from_klines(k1h, symbol=symbol, direction="ABOVE", strike_offset_pct=0.01)
    gen = build_generator(cfg)
    run = run_backtest(symbol=symbol, klines_1m=k1m, klines_1h=k1h, klines_4h=k4h, events=events, generator=gen)
    days = (k1h[-1].close_time - k1h[0].open_time).total_seconds() / 86400.0
    config = {"scoring": cfg.__dict__, "n_hours": n_hours, "drift": drift, "vol": vol}
    result = to_backtest_result(run, k1h[0].open_time, k1h[-1].close_time, days, config)
    return result, days, config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Horizon-Incident 回测入口")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--hours", type=int, default=720)
    parser.add_argument("--output-dir", default="proofs/iter-1")
    parser.add_argument("--drift", type=float, default=0.0002)
    parser.add_argument("--vol", type=float, default=0.02)
    args = parser.parse_args(argv)

    cfg = ScoringConfig()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for symbol in args.symbols:
        result, days, config = run_one_symbol(symbol, cfg, n_hours=args.hours, drift=args.drift, vol=args.vol)
        all_results.append((symbol, result, days, config))
        print(
            f"[backtest] {symbol}: signals={result.total_signals} "
            f"win_rate={result.win_rate*100:.1f}% "
            f"payoff={result.avg_payoff_ratio:.2f} "
            f"ev={result.expected_value:+.3f}"
        )

    # 合并 BTC + ETH 写 REPORT
    if len(all_results) > 1:
        # 主报告用第一个 symbol（BTC）的结果，附带 ETH
        primary = all_results[0]
        report_path = write_report(primary[1], primary[2], primary[3], out_dir / "REPORT.md")
    else:
        primary = all_results[0]
        report_path = write_report(primary[1], primary[2], primary[3], out_dir / "REPORT.md")

    # JSON 落盘
    summary = {
        "results": [
            {
                "symbol": sym,
                "strategy": res.strategy,
                "win_rate": res.win_rate,
                "avg_payoff_ratio": res.avg_payoff_ratio,
                "expected_value": res.expected_value,
                "total_signals": res.total_signals,
                "wins": res.wins,
                "losses": res.losses,
                "signals_per_day": res.signals_per_day,
            }
            for sym, res, _, _ in all_results
        ]
    }
    (out_dir / "backtest_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[backtest] REPORT -> {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
真实数据回测入口：拉 BTC/ETH 真 K 线 → 构造事件 → 跑因子策略 → 写 REPORT。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.backtest.backtester import run_backtest, to_backtest_result
from src.backtest.real_data import (
    build_4h_aggregated_klines,
    build_events_from_klines,
    build_mixed_events,
    fetch_and_cache_klines,
)
from src.backtest.reporter import write_report
from src.signals.generator import SignalGenerator
from src.signals.scoring import ScoringConfig


def run_one_symbol(
    symbol: str,
    cfg: ScoringConfig,
    *,
    n_hours: int = 720,
    direction: str = "ABOVE",
    mixed: bool = False,
) -> tuple:
    k1h = fetch_and_cache_klines(symbol, interval="1h", limit=n_hours)
    k4h = build_4h_aggregated_klines(k1h)
    if mixed:
        events = build_mixed_events(k1h, symbol=symbol, strike_offset_pct=0.02)
        actual_dir = "MIXED"
    else:
        events = build_events_from_klines(k1h, symbol=symbol, direction=direction, strike_offset_pct=0.02)
        actual_dir = direction
    gen = SignalGenerator(scoring_config=cfg, strategy_name="factor_v1", ttl_seconds=60)
    run = run_backtest(
        symbol=symbol,
        klines_1m=None,  # 用 1h 替代（无 1m 数据时）
        klines_1h=k1h,
        klines_4h=k4h,
        events=events,
        generator=gen,
    )
    days = (k1h[-1].close_time - k1h[0].open_time).total_seconds() / 86400.0
    config = {
        "scoring": cfg.__dict__,
        "n_hours": n_hours,
        "data_source": "binance.com/api/v3/klines (real)",
        "direction": actual_dir,
        "strike_offset_pct": 0.02,
    }
    result = to_backtest_result(run, k1h[0].open_time, k1h[-1].close_time, days, config)
    return result, days, config, k1h


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Horizon-Incident 真实数据回测")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--hours", type=int, default=720)
    parser.add_argument("--direction", default="ABOVE", help="ABOVE / BELOW / MIXED")
    parser.add_argument("--output-dir", default="proofs/iter-2")
    args = parser.parse_args(argv)

    cfg = ScoringConfig()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mixed = args.direction.upper() == "MIXED"
    direction = "ABOVE" if mixed else args.direction

    all_results = []
    for symbol in args.symbols:
        result, days, config, k1h = run_one_symbol(symbol, cfg, n_hours=args.hours, direction=direction, mixed=mixed)
        all_results.append((symbol, result, days, config))
        print(
            f"[real-backtest] {symbol}: signals={result.total_signals} "
            f"win_rate={result.win_rate*100:.1f}% "
            f"payoff={result.avg_payoff_ratio:.2f} "
            f"ev={result.expected_value:+.3f} "
            f"pnl={result.total_pnl_pct:+.1f}%"
        )
        # 价格摘要
        first = float(k1h[0].close)
        last = float(k1h[-1].close)
        print(f"  区间：{k1h[0].open_time} → {k1h[-1].close_time}")
        print(f"  价格：{first:.2f} → {last:.2f} ({(last/first-1)*100:+.1f}%)")

    primary = all_results[0]
    report_path = write_report(primary[1], primary[2], primary[3], out_dir / "REPORT.md")

    summary = {
        "data_source": "币安现货 K 线（直连 api.binance.com）",
        "method": "K 线反推事件合约：每根 1h K 线 = 1 个 1h 事件，strike = open ± 2% 随机",
        "direction_strategy": args.direction,
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
                "total_pnl_pct": res.total_pnl_pct,
            }
            for sym, res, _, _ in all_results
        ],
    }
    (out_dir / "backtest_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[real-backtest] REPORT -> {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
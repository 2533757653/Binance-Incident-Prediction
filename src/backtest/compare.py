"""
Builder-F · iter-2 compare 工具：选最优 (strategy × symbol) 组合。

输入：multi_main.py 生成的 compare.json（包含 results 列表）。
输出：最优组合的摘要 + 落盘 proofs/iter-2/compare.json（与 multi_main 同格式，但只含最佳）。

用法：
    python src/backtest/compare.py proofs/iter-2/compare.json
    python src/backtest/compare.py --in proofs/iter-2/compare.json --out proofs/iter-2/best.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# 允许 python src/backtest/compare.py 直接跑
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _score(r: dict, *, threshold_win: float = 0.55) -> float:
    """
    选优打分：
    - 胜率 >= 0.55 的优先（在所有 candidate 中选 EV 最大的）
    - 否则选 EV 最大的
    评分 = win_rate + 0.5 * max(0, ev)  - 0.1 * brier_cal
    """
    return r["win_rate"] + 0.5 * max(0.0, r["expected_value"]) - 0.1 * r.get("brier_cal", 0.25)


def select_best(results: List[dict], *, threshold_win: float = 0.55) -> dict:
    if not results:
        raise ValueError("empty results list")
    qualified = [r for r in results if r["win_rate"] >= threshold_win]
    pool = qualified if qualified else results
    return max(pool, key=lambda r: (r["win_rate"], r["expected_value"]))


def build_compare_payload(loaded: dict) -> dict:
    """从 multi_main 的输出 JSON 构造一个"摘要 + 最优"格式的 compare.json。"""
    results = loaded["results"]
    best = select_best(results)
    return {
        "best_strategy": best["strategy"],
        "best_symbol": best["symbol"],
        "win_rate": best["win_rate"],
        "ev": best["expected_value"],
        "avg_payoff_ratio": best["avg_payoff_ratio"],
        "total_signals": best["total_signals"],
        "brier_raw": best["brier_raw"],
        "brier_cal": best["brier_cal"],
        "expected_value": best["expected_value"],
        "score": round(_score(best), 6),
        "n_above_55": sum(1 for r in results if r["win_rate"] >= 0.55),
        "n_rows": len(results),
        "calibration": loaded.get("calibration", {}),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Horizon-Incident iter-2 选最优 (strategy × symbol)")
    parser.add_argument("input", nargs="?", default="proofs/iter-2/compare.json",
                        help="multi_main 输出的 compare.json 路径")
    parser.add_argument("--in", dest="input_path", default=None, help="同 input（兼容）")
    parser.add_argument("--out", default="proofs/iter-2/best.json", help="输出文件")
    args = parser.parse_args(argv)

    in_path = Path(args.input_path or args.input).resolve()
    if not in_path.exists():
        print(f"[compare] input not found: {in_path}", file=sys.stderr)
        return 2
    loaded = json.loads(in_path.read_text(encoding="utf-8"))
    if "results" not in loaded:
        print(f"[compare] bad input format (no 'results')", file=sys.stderr)
        return 3

    payload = build_compare_payload(loaded)
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # 终端输出
    print(f"[compare] best: {payload['best_strategy']} × {payload['best_symbol']}  "
          f"win_rate={payload['win_rate']*100:.1f}%  ev={payload['ev']:+.3f}  "
          f"score={payload['score']:.3f}  (rows={payload['n_rows']}, "
          f"above_55={payload['n_above_55']})")
    print(f"[compare] -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

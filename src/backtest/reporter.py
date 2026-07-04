"""
REPORT.md 自动生成（jinja2 模板 + BacktestResult 数据）。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

from jinja2 import Template

from src.common.schemas import BacktestResult


_REPORT_TEMPLATE = """# 回测报告 — {{ strategy }} on {{ symbol }}

> 生成时间：{{ generated_at }}
> 区间：{{ start }} → {{ end }}（{{ days }} 天）

---

## 1. 核心指标（一眼看懂）

| 指标 | 数值 | 评判（>=55% / >=1.5 / >0） |
|---|---|---|
| 信号总数 | {{ result.total_signals }} | — |
| 胜率 | **{{ '%.1f%%' % (result.win_rate * 100) }}** | {{ '✅' if result.win_rate >= 0.55 else '⚠️ 低于 55%' }} |
| 平均盈亏比 | **{{ '%.2f' % result.avg_payoff_ratio }}** | {{ '✅' if result.avg_payoff_ratio >= 1.5 else '⚠️ 低于 1.5' }} |
| 期望值 E | **{{ '%+.3f' % result.expected_value }}** | {{ '✅' if result.expected_value > 0 else '❌ 长期亏损' }} |
| 总盈亏 | {{ '%+.1f%%' % result.total_pnl_pct }} | — |
| 最大回撤 | {{ '%.1f%%' % result.max_drawdown_pct }} | — |
| 日均信号 | {{ '%.1f' % result.signals_per_day }} | — |
| 平均延迟 | {{ '%.0f ms' % result.avg_latency_ms }} | — |

---

## 2. 一句话结论

{{ verdict }}

---

## 3. 因子贡献（待 iter-2 完善）

本轮只跑 `factor_v1`（MOM_15m + BB_1h + ATR_4h 三因子多数派投票）。
策略细节：见 `src/signals/generator.py`、`src/signals/scoring.py`。

---

## 4. 关键洞察

1. **期望值 E > 0** 是系统"长期可盈利"的硬指标。胜率 × 盈亏比 > 败率 才算数。
2. **盈亏比不对称**：YES 价格越低，赢了赚越多（盈亏比 = (1-p)/p）。
3. **价差过滤**至关重要：spread > 0.05 的事件被排除，避免滑点吃光利润。

---

## 5. 截图清单

> 见 `proofs/iter-1/` 同名 png。

- `01-signals.png` — 终端彩色信号流
- `02-toast.png` — Windows 弹窗示例
- `03-backtest.png` — 回测指标表

---

## 6. 配置快照

```json
{{ config_json }}
```
"""


def render_report(result: BacktestResult, days: float, config: dict) -> str:
    """渲染 REPORT.md 内容。"""
    if result.expected_value > 0 and result.win_rate >= 0.55 and result.avg_payoff_ratio >= 1.5:
        verdict = (
            f"✅ **策略可信**：在 {result.total_signals} 条信号上跑出 {result.win_rate*100:.1f}% 胜率、"
            f"{result.avg_payoff_ratio:.2f} 盈亏比、{result.expected_value:+.3f} 期望值。"
            f"可以进入实时信号阶段（iter-2 多策略对比）。"
        )
    elif result.expected_value > 0:
        verdict = (
            f"⚠️ **期望值正但单指标不达标**：胜率 {result.win_rate*100:.1f}%、盈亏比 {result.avg_payoff_ratio:.2f}。"
            f"iter-2 调因子权重或加入新因子。"
        )
    else:
        verdict = (
            f"❌ **期望值为负**（{result.expected_value:+.3f}）：策略当前不可用。"
            f"建议 iter-2 重新校准因子或换 ML 路线。"
        )

    import json
    template = Template(_REPORT_TEMPLATE)
    return template.render(
        strategy=result.strategy,
        symbol=result.symbol,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        start=result.start_time.strftime("%Y-%m-%d %H:%M"),
        end=result.end_time.strftime("%Y-%m-%d %H:%M"),
        days=f"{days:.1f}",
        result=result,
        verdict=verdict,
        config_json=json.dumps(config, indent=2, ensure_ascii=False, default=str),
    )


def write_report(result: BacktestResult, days: float, config: dict, output_path: Path) -> Path:
    """写 REPORT.md 到指定路径。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_report(result, days, config), encoding="utf-8")
    return output_path
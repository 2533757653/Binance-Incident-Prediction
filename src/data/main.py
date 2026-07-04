"""Builder-D 数据接入层主入口（iter-2 真数据版）。

行为：
1. 拉取 eapi /exerciseHistory → 转 EventContract 流
2. 拉取 /api/v3/klines 喂因子（4 个标的 × 1h × 720）
3. 终端打印：按 underlying 分组的最近 100 条事件

运行：
    python D:/Horizon-Incident/src/data/main.py [--duration-sec 30] [--force]

退出码：
    0   正常
    10  数据源失败
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# 让脚本可独立跑
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rich.console import Console
from rich.table import Table

from src.common.errors import FeedError
from src.common.schemas import EventContract, KlineBar
from src.data.exercise_history import ExerciseHistoryFetcher
from src.data.historical_events import HistoricalEventBuilder
from src.data.klines_fetcher import KlinesFetcher

# 日志
log = logging.getLogger("horizon.data")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

console = Console()

_shutdown = False


def _on_sigint(_sig, _frm) -> None:
    global _shutdown
    _shutdown = True
    console.print("\n[yellow]收到 Ctrl-C，准备退出 …[/yellow]")


signal.signal(signal.SIGINT, _on_sigint)


# ============================================================
# 渲染
# ============================================================
def _render_events_by_underlying(events: list[EventContract], top_n: int = 100) -> Table:
    """按 underlying 分组渲染最近 N 条事件。"""
    t = Table(title=f"Horizon-Incident · Exercise History (top {top_n})", show_lines=True)
    t.add_column("Underlying", style="bold cyan", width=10)
    t.add_column("Direction", style="bold", width=8)
    t.add_column("Strike", style="white", justify="right")
    t.add_column("YES prob", style="green", justify="right")
    t.add_column("Settle Time", style="dim")
    t.add_column("Event ID", style="dim")

    recent = events[:top_n]
    for ev in recent:
        t.add_row(
            ev.underlying,
            ev.direction,
            f"{ev.strike_price:g}",
            f"{ev.current_yes_price:.4f}",
            ev.settle_time.strftime("%Y-%m-%d %H:%M"),
            ev.event_id,
        )
    return t


def _render_summary(groups: dict[str, list[EventContract]]) -> Table:
    """按 underlying 汇总统计。"""
    t = Table(title="Summary by Underlying", show_lines=True)
    t.add_column("Underlying", style="bold cyan", width=10)
    t.add_column("Total", justify="right")
    t.add_column("ABOVE", justify="right", style="green")
    t.add_column("BELOW", justify="right", style="red")
    t.add_column("Avg YES prob", justify="right")
    t.add_column("Date Range", style="dim")

    for u, evs in sorted(groups.items()):
        above = sum(1 for e in evs if e.direction == "ABOVE")
        below = sum(1 for e in evs if e.direction == "BELOW")
        avg_prob = sum(e.current_yes_price for e in evs) / len(evs) if evs else 0
        if evs:
            times = sorted([e.settle_time for e in evs])
            dr = f"{times[0].strftime('%m-%d')} ~ {times[-1].strftime('%m-%d')}"
        else:
            dr = "-"
        t.add_row(u, str(len(evs)), str(above), str(below), f"{avg_prob:.4f}", dr)
    return t


# ============================================================
# main
# ============================================================
def main(duration_sec: int = 30, force: bool = False) -> int:
    started = time.time()

    # ---- 1. 拉取 + 转换 exerciseHistory → EventContract ----
    console.print("[bold cyan]Step 1[/bold cyan] · 拉取 eapi/exerciseHistory …")
    builder = HistoricalEventBuilder()
    try:
        events = builder.build(top_n=100, force=force)
    except FeedError as e:
        log.error("exerciseHistory 拉取失败: %s", e)
        console.print(f"[red]FAIL[/red] exerciseHistory 拉取失败: {e}")
        return 10
    finally:
        builder.close()

    console.print(f"[green]事件转换完成[/green]：{len(events)} 条 EventContract")

    # ---- 2. 按 underlying 分组 ----
    groups: dict[str, list[EventContract]] = defaultdict(list)
    for ev in events:
        groups[ev.underlying].append(ev)

    console.print(_render_events_by_underlying(events, top_n=100))
    console.print(_render_summary(groups))

    # ---- 3. 拉 K 线（4 标的 × 1h） ----
    console.print("\n[bold cyan]Step 2[/bold cyan] · 拉取 K 线（4 标的 × 1h × 720） …")
    kf = KlinesFetcher()
    kline_counts: dict[str, int] = {}
    try:
        symbols = sorted({ev.underlying + "USDT" for ev in events})
        if not symbols:
            symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]
        for sym in symbols:
            try:
                bars: list[KlineBar] = kf.fetch(sym, "1h", limit=720, force=force)
                kline_counts[sym] = len(bars)
                console.print(f"  [green]OK[/green] {sym}: {len(bars)} 根 K 线")
            except FeedError as e:
                console.print(f"  [yellow]SKIP[/yellow] {sym}: {e}")
    finally:
        kf.close()

    # ---- 4. 简短运行窗口（让 Ctrl-C 友好） ----
    if duration_sec and duration_sec > 0:
        remaining = max(0, duration_sec - (time.time() - started))
        if remaining > 0 and not _shutdown:
            console.print(f"\n[dim]运行窗口 {remaining:.1f}s（Ctrl-C 退出）…[/dim]")
            end = time.time() + remaining
            while not _shutdown and time.time() < end:
                time.sleep(0.2)

    console.print("\n[bold green]完成[/bold green]")
    console.print(
        f"  事件: {len(events)} 条（{len(groups)} 标的）"
    )
    console.print(
        f"  K 线: {sum(kline_counts.values())} 根（{len(kline_counts)} 标的）"
    )
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--duration-sec", type=int, default=0, help="运行窗口（秒），0=跑完即退")
    p.add_argument("--force", action="store_true", help="强制重拉网络（忽略缓存）")
    args = p.parse_args()
    try:
        sys.exit(main(duration_sec=args.duration_sec, force=args.force))
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("main 异常: %s", e)
        sys.exit(10)

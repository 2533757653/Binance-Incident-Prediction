"""Terminal colored printing with rich.

Three views: signals / events / ticks. Switch via --view.
Renders to stdout AND can dump to PNG via Capture + PIL.
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from rich.box import ROUNDED
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ---- local dataclass mirrors of dataContract §1.4 / §1.3 / §1.1 ----
# Kept identical field-for-field; when Builder-A/B ships we swap import.
@dataclass(slots=True)
class SignalLite:
    ts: datetime
    signal_id: str
    event_id: str
    event_title: str
    symbol: str
    side: str
    entry_price: float
    spread: float
    confidence: float
    expected_value: float
    win_prob: float
    payoff_ratio: float
    rationale: str
    strategy: str
    factors: dict
    ttl_seconds: int
    expire_at: datetime


@dataclass(slots=True)
class EventLite:
    event_id: str
    symbol: str
    title: str
    strike_price: float
    direction: str
    time_to_expiry: str
    settle_time: datetime
    current_yes_price: float
    current_no_price: float
    spread: float
    status: str


@dataclass(slots=True)
class TickLite:
    ts: datetime
    symbol: str
    price: float
    qty: float


# ---- color / styling helpers ----
HIGH_CONF_THRESHOLD = 0.65


def _style_confidence(value: float) -> Text:
    """Confidence cell: dim <0.55, default 0.55-0.64, bold+bg ≥0.65."""
    if value >= HIGH_CONF_THRESHOLD:
        return Text(f"{value:.3f}", style="bold white on green")
    if value >= 0.55:
        return Text(f"{value:.3f}", style="bold yellow")
    return Text(f"{value:.3f}", style="dim")


def _style_side(side: str) -> Text:
    if side == "YES":
        return Text(side, style="bold green")
    if side == "NO":
        return Text(side, style="bold red")
    return Text(side, style="white")


def _style_symbol(symbol: str) -> Text:
    if symbol.startswith("BTC"):
        return Text(symbol, style="bold #F7931A")  # BTC orange
    if symbol.startswith("ETH"):
        return Text(symbol, style="bold #627EEA")  # ETH purple
    return Text(symbol, style="white")


# ---- builders ----
def build_signals_table(signals: list[SignalLite], title: str = "实时信号流") -> Table:
    table = Table(
        title=f"[bold]{title}[/bold]",
        box=ROUNDED,
        show_lines=False,
        expand=True,
    )
    columns = [
        ("时间", 16),
        ("信号ID", 22),
        ("标的", 10),
        ("方向", 6),
        ("入场", 8),
        ("价差", 7),
        ("置信", 7),
        ("期望值", 8),
        ("盈亏比", 7),
        ("理由", 32),
    ]
    for col, width in columns:
        justify = "right" if col in {"入场", "价差", "置信", "期望值", "盈亏比"} else "left"
        table.add_column(col, width=width, justify=justify, no_wrap=True)

    if not signals:
        table.add_row("[dim]暂无信号[/dim]", "", "", "", "", "", "", "", "", "")
    for s in signals:
        table.add_row(
            s.ts.strftime("%m-%d %H:%M:%S"),
            s.signal_id[-18:] if len(s.signal_id) > 18 else s.signal_id,
            _style_symbol(s.symbol),
            _style_side(s.side),
            f"{s.entry_price:.4f}",
            f"{s.spread:.3f}",
            _style_confidence(s.confidence),
            f"{s.expected_value:+.3f}",
            f"{s.payoff_ratio:.2f}",
            s.rationale[:32],
        )
    return table


def build_events_table(events: list[EventLite], title: str = "活跃事件") -> Table:
    table = Table(title=f"[bold]{title}[/bold]", box=ROUNDED, expand=True)
    for col, width, justify in [
        ("事件ID", 22, "left"),
        ("标的", 10, "left"),
        ("方向", 6, "left"),
        ("触发价", 10, "right"),
        ("YES", 8, "right"),
        ("NO", 8, "right"),
        ("价差", 7, "right"),
        ("到期", 16, "left"),
        ("状态", 8, "left"),
    ]:
        table.add_column(col, width=width, justify=justify, no_wrap=True)
    if not events:
        table.add_row("[dim]暂无事件[/dim]", "", "", "", "", "", "", "", "")
    for e in events:
        table.add_row(
            e.event_id[-20:] if len(e.event_id) > 20 else e.event_id,
            _style_symbol(e.symbol),
            e.direction,
            f"{e.strike_price:.2f}",
            f"{e.current_yes_price:.3f}",
            f"{e.current_no_price:.3f}",
            f"{e.spread:.3f}",
            e.settle_time.strftime("%m-%d %H:%M"),
            e.status,
        )
    return table


def build_ticks_table(ticks: list[TickLite], title: str = "实时价 tick") -> Table:
    table = Table(title=f"[bold]{title}[/bold]", box=ROUNDED, expand=True)
    for col, width, justify in [
        ("时间", 16, "left"),
        ("标的", 10, "left"),
        ("价格", 12, "right"),
        ("成交量", 12, "right"),
    ]:
        table.add_column(col, width=width, justify=justify, no_wrap=True)
    if not ticks:
        table.add_row("[dim]暂无 tick[/dim]", "", "", "")
    for t in ticks[-30:]:
        table.add_row(
            t.ts.strftime("%H:%M:%S"),
            _style_symbol(t.symbol),
            f"{t.price:.2f}",
            f"{t.qty:.4f}",
        )
    return table


# ---- PNG export ----
def render_to_png(
    table_or_group,
    png_path: Path,
    width: int = 1600,
    bg: str = "black",
) -> Path:
    """Render a Rich Table (or Group) into a PNG using rich.console.Capture + PIL."""
    from PIL import Image  # noqa: WPS433

    str_buf = io.StringIO()
    cap_console = Console(
        file=str_buf,
        width=160,
        force_terminal=True,
        color_system="truecolor",
        record=True,
        legacy_windows=False,
    )
    cap_console.print(table_or_group)
    text = str_buf.getvalue()

    # Use a console in record mode to extract ANSI -> pixel via export_text.
    record_console = Console(
        width=160,
        record=True,
        force_terminal=True,
        color_system="truecolor",
        legacy_windows=False,
    )
    record_console.print(table_or_group)
    # export_text returns ANSI; we render via PIL by writing lines.
    # Simpler: use Console.export_text(styles=True) which keeps ANSI codes,
    # then render via PIL ImageFont on top of raw text using Pillow.
    # Even simpler: use Console.save_html / save_svg fallback. But we want PNG.
    # We'll write text into an image directly with monospace font.

    try:
        from PIL import ImageDraw, ImageFont
    except ImportError as e:
        raise RuntimeError("PIL required") from e

    # Heuristic pixel-per-char size for Consolas / DejaVuSansMono fallback.
    char_w, char_h = 9, 18
    pad = 20
    img_w = width
    # Number of text lines (split by \n)
    lines = text.splitlines() or [""]
    # Wrap aggressively if needed
    img_h = char_h * (len(lines) + 2) + pad * 2

    img = Image.new("RGB", (img_w, img_h), color=bg)
    draw = ImageDraw.Draw(img)

    # Try to find a monospace font
    font = None
    for candidate in (
        "consola.ttf",
        "consolab.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/consolab.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ):
        try:
            font = ImageFont.truetype(candidate, 14)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    # Strip ANSI for plain-text fallback rendering. We lose color but the PNG
    # still proves the layout existed and was generated.
    ansi_re = __import__("re").compile(r"\x1b\[[0-9;]*m")
    clean_lines = [ansi_re.sub("", line) for line in lines]

    y = pad
    for line in clean_lines:
        draw.text((pad, y), line, fill=(220, 220, 220), font=font)
        y += char_h

    png_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(png_path, format="PNG")
    return png_path


# ---- main entry ----
def make_demo_signals() -> list[SignalLite]:
    from datetime import timedelta

    now = datetime.now()
    return [
        SignalLite(
            ts=now,
            signal_id="sig-BTCUSDT-1H-ABOVE-70000-20260624153000-152930",
            event_id="BTCUSDT-1H-ABOVE-70000-20260624153000",
            event_title="BTC 1h 后 ≥ 70000?",
            symbol="BTCUSDT",
            side="YES",
            entry_price=0.62,
            spread=0.012,
            confidence=0.72,
            expected_value=0.18,
            win_prob=0.66,
            payoff_ratio=0.61,
            rationale="mom_15m=+1.2% 且 bb_pct_1h 突破 0.95",
            strategy="factor_v1",
            factors={"mom_15m": 0.012, "bb_pct_1h": 0.96, "atr_break_4h": 1.6},
            ttl_seconds=60,
            expire_at=now + timedelta(seconds=60),
        ),
        SignalLite(
            ts=now,
            signal_id="sig-ETHUSDT-4H-BELOW-3500-20260624160000-153001",
            event_id="ETHUSDT-4H-BELOW-3500-20260624160000",
            event_title="ETH 4h 后 ≤ 3500?",
            symbol="ETHUSDT",
            side="NO",
            entry_price=0.41,
            spread=0.020,
            confidence=0.68,
            expected_value=0.12,
            win_prob=0.62,
            payoff_ratio=1.44,
            rationale="atr_break_4h 偏弱, mom_15m=-0.4%",
            strategy="factor_v1",
            factors={"mom_15m": -0.004, "bb_pct_1h": 0.18, "atr_break_4h": 0.8},
            ttl_seconds=60,
            expire_at=now + timedelta(seconds=60),
        ),
        SignalLite(
            ts=now,
            signal_id="sig-BTCUSDT-1H-ABOVE-69500-20260624163000-153015",
            event_id="BTCUSDT-1H-ABOVE-69500-20260624163000",
            event_title="BTC 1h 后 ≥ 69500?",
            symbol="BTCUSDT",
            side="NO",
            entry_price=0.55,
            spread=0.015,
            confidence=0.52,
            expected_value=0.04,
            win_prob=0.54,
            payoff_ratio=0.82,
            rationale="价差略大, 因子分歧",
            strategy="factor_v1",
            factors={"mom_15m": -0.001, "bb_pct_1h": 0.42, "atr_break_4h": 1.0},
            ttl_seconds=60,
            expire_at=now + timedelta(seconds=60),
        ),
    ]


def make_demo_events() -> list[EventLite]:
    from datetime import timedelta

    now = datetime.now()
    return [
        EventLite(
            event_id="BTCUSDT-1H-ABOVE-70000-20260624153000",
            symbol="BTCUSDT",
            title="BTC 1h 后 ≥ 70000?",
            strike_price=70000.0,
            direction="ABOVE",
            time_to_expiry="1h",
            settle_time=now + timedelta(minutes=42),
            current_yes_price=0.62,
            current_no_price=0.38,
            spread=0.012,
            status="TRADING",
        ),
        EventLite(
            event_id="ETHUSDT-4H-BELOW-3500-20260624160000",
            symbol="ETHUSDT",
            title="ETH 4h 后 ≤ 3500?",
            strike_price=3500.0,
            direction="BELOW",
            time_to_expiry="4h",
            settle_time=now + timedelta(hours=3, minutes=12),
            current_yes_price=0.41,
            current_no_price=0.59,
            spread=0.020,
            status="TRADING",
        ),
    ]


def make_demo_ticks() -> list[TickLite]:
    now = datetime.now()
    return [
        TickLite(ts=now, symbol="BTCUSDT", price=69823.45, qty=0.012),
        TickLite(ts=now, symbol="ETHUSDT", price=3491.12, qty=0.34),
    ]


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Horizon-Incident console output")
    parser.add_argument(
        "--view",
        choices=["signals", "events", "ticks"],
        default="signals",
        help="选择视图",
    )
    parser.add_argument(
        "--png",
        type=str,
        default=None,
        help="渲染 PNG 到指定路径（用于截图证据）",
    )
    parser.add_argument(
        "--live-seconds",
        type=int,
        default=0,
        help="实时刷新秒数（0 = 一次性打印）",
    )
    args = parser.parse_args(argv)

    console = Console(force_terminal=True, color_system="truecolor")

    if args.view == "signals":
        items = make_demo_signals()
        table = build_signals_table(items)
    elif args.view == "events":
        items = make_demo_events()
        table = build_events_table(items)
    else:
        items = make_demo_ticks()
        table = build_ticks_table(items)

    if args.live_seconds > 0:
        with Live(table, console=console, refresh_per_second=2, screen=False) as live:
            for _ in range(args.live_seconds * 2):
                time.sleep(0.5)
                live.update(table)
    else:
        console.print(table)

    if args.png:
        out = Path(args.png)
        render_to_png(table, out)
        console.print(f"[dim]PNG saved -> {out}[/dim]")

    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
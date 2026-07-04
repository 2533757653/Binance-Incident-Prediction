"""Entry point: run rich table + (optionally) toast listening for synthetic signals.

Usage:
    python D:/Horizon-Incident/src/output/main.py --view signals --png .../01-signals.png
    python D:/Horizon-Incident/src/output/main.py --toast --png .../02-toast.png
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root on sys.path so `src.*` imports work when run as script
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.output.console import (  # noqa: E402
    build_signals_table,
    make_demo_signals,
    make_demo_events,
    make_demo_ticks,
    render_to_png,
)
from src.output.toast import default_service, should_trigger_toast  # noqa: E402

LOG = logging.getLogger("horizon.output.main")


def _demo_high_conf_signal():
    """Return the highest-confidence signal from demo set (for toast demo)."""
    signals = make_demo_signals()
    return max(signals, key=lambda s: s.confidence)


def cmd_render(args: argparse.Namespace) -> int:
    """Render a view + optional PNG."""
    if args.view == "signals":
        items = make_demo_signals()
        table = build_signals_table(items)
    elif args.view == "events":
        items = make_demo_events()
        table = build_signals_table(make_demo_signals())  # also include signals header
        from rich.console import Group
        from src.output.console import build_events_table
        table = Group(build_signals_table(make_demo_signals()), build_events_table(items))
    else:
        from src.output.console import build_ticks_table
        table = build_ticks_table(make_demo_ticks())

    from rich.console import Console
    console = Console(force_terminal=True, color_system="truecolor")
    console.print(table)

    if args.png:
        render_to_png(table, Path(args.png))
        console.print(f"[dim]PNG saved -> {args.png}[/dim]")
    return 0


def cmd_toast(args: argparse.Namespace) -> int:
    """Trigger a toast for the demo high-confidence signal; render PNG proof."""
    sig = _demo_high_conf_signal()
    LOG.info(
        "demo signal: signal_id=%s conf=%.2f side=%s entry=%.4f",
        sig.signal_id,
        sig.confidence,
        sig.side,
        sig.entry_price,
    )

    triggered = should_trigger_toast(sig.confidence)
    record = None
    if triggered:
        record = default_service.notify_signal(
            signal_id=sig.signal_id,
            event_title=sig.event_title,
            side=sig.side,
            entry_price=sig.entry_price,
            confidence=sig.confidence,
            triggered_at=datetime.now(),
        )
        LOG.info("toast record: via=%s success=%s err=%s",
                 record.delivered_via, record.success, record.error)
        # Give the toast UI a moment to render before we screenshot
        time.sleep(2.0)
    else:
        LOG.warning("demo signal conf=%.2f below toast threshold; not firing",
                    sig.confidence)

    # Always also render the rich table as a sanity proof
    table = build_signals_table(make_demo_signals())
    from rich.console import Console
    console = Console(force_terminal=True, color_system="truecolor")
    console.print(table)

    if args.png:
        render_to_png(table, Path(args.png))
        console.print(f"[dim]PNG saved -> {args.png}[/dim]")

    return 0 if (record is None or record.success or record.delivered_via == "console") else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Horizon output entry")
    sub = parser.add_subparsers(dest="cmd")

    p_render = sub.add_parser("render", help="render a view + optional PNG")
    p_render.add_argument("--view", choices=["signals", "events", "ticks"],
                          default="signals")
    p_render.add_argument("--png", type=str, default=None)

    p_toast = sub.add_parser("toast", help="fire a demo toast + render PNG proof")
    p_toast.add_argument("--png", type=str, default=None)

    args = parser.parse_args(argv)
    if args.cmd == "toast":
        return cmd_toast(args)
    # default: render
    if args.cmd is None:
        # Back-compat: `python main.py --view signals --png ...`
        parser2 = argparse.ArgumentParser()
        parser2.add_argument("--view", choices=["signals", "events", "ticks"],
                             default="signals")
        parser2.add_argument("--png", type=str, default=None)
        a2 = parser2.parse_args(argv)
        return cmd_render(a2)
    return cmd_render(args)


if __name__ == "__main__":
    sys.exit(main())
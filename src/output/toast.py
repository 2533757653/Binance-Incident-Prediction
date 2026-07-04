"""Windows toast notifications for high-confidence signals.

Primary: win10toast-click. Fallback: plyer. Silent degrade on failure.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

LOG = logging.getLogger("horizon.output.toast")

HIGH_CONF_THRESHOLD = 0.65


@dataclass(slots=True)
class ToastRecord:
    triggered_at: datetime
    event_title: str
    side: str
    entry_price: float
    confidence: float
    signal_id: str
    delivered_via: str  # "win10toast" | "plyer" | "console"
    success: bool
    error: Optional[str] = None


class ToastService:
    """Thread-safe singleton-ish wrapper. Safe to use as a global."""

    def __init__(self, app_id: str = "Horizon-Incident") -> None:
        self.app_id = app_id
        self._wtoaster = None
        self._plyer = None
        self._wtoaster_err: Optional[str] = None
        self._init_lock = threading.Lock()
        self._init_done = False

    def _ensure_init(self) -> None:
        if self._init_done:
            return
        with self._init_lock:
            if self._init_done:
                return
            try:
                from win10toast_click import ToastNotifier  # type: ignore

                self._wtoaster = ToastNotifier()
                LOG.info("toast backend: win10toast-click ready")
            except Exception as e:  # noqa: BLE001
                self._wtoaster_err = repr(e)
                LOG.warning("win10toast-click init failed: %s; will fall back to plyer", e)
                self._wtoaster = None
            try:
                from plyer import notification  # type: ignore

                self._plyer = notification
            except Exception as e:  # noqa: BLE001
                LOG.warning("plyer init failed: %s", e)
                self._plyer = None
            self._init_done = True

    def notify_signal(
        self,
        signal_id: str,
        event_title: str,
        side: str,
        entry_price: float,
        confidence: float,
        triggered_at: datetime | None = None,
        duration_sec: int = 8,
    ) -> ToastRecord:
        """Fire a toast for a high-confidence signal.

        Returns a ToastRecord describing what happened (delivered_via + success).
        Silent fallback: terminal print still happens via the caller.
        """
        self._ensure_init()
        triggered_at = triggered_at or datetime.now()

        title = f"[Horizon] {side} {event_title}"
        msg = (
            f"side={side} entry={entry_price:.4f} "
            f"conf={confidence:.2f} @ {triggered_at.strftime('%H:%M:%S')}"
        )

        # Try win10toast-click first
        if self._wtoaster is not None:
            try:
                # win10toast-click runs threaded on its own; non-blocking
                self._wtoaster.show_toast(
                    title=title,
                    msg=msg,
                    duration=duration_sec,
                    threaded=True,
                )
                return ToastRecord(
                    triggered_at=triggered_at,
                    event_title=event_title,
                    side=side,
                    entry_price=entry_price,
                    confidence=confidence,
                    signal_id=signal_id,
                    delivered_via="win10toast",
                    success=True,
                )
            except Exception as e:  # noqa: BLE001
                LOG.warning("win10toast fire failed: %s; trying plyer", e)

        # Fall back to plyer
        if self._plyer is not None:
            try:
                self._plyer.notify(
                    title=title,
                    message=msg,
                    app_name=self.app_id,
                    timeout=duration_sec,
                )
                return ToastRecord(
                    triggered_at=triggered_at,
                    event_title=event_title,
                    side=side,
                    entry_price=entry_price,
                    confidence=confidence,
                    signal_id=signal_id,
                    delivered_via="plyer",
                    success=True,
                )
            except Exception as e:  # noqa: BLE001
                LOG.warning("plyer fire failed: %s; degrade to console", e)

        # Last resort: silent, but still record the attempt.
        return ToastRecord(
            triggered_at=triggered_at,
            event_title=event_title,
            side=side,
            entry_price=entry_price,
            confidence=confidence,
            signal_id=signal_id,
            delivered_via="console",
            success=False,
            error=self._wtoaster_err or "no toast backend available",
        )


# Module-level singleton for convenience
default_service = ToastService()


def should_trigger_toast(confidence: float) -> bool:
    """Predicate matching spec: confidence >= 0.65 triggers."""
    return confidence >= HIGH_CONF_THRESHOLD
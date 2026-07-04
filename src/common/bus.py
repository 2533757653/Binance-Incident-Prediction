"""
Horizon-Incident 进程内消息总线。
Pydantic 序列化 + queue.Queue 跨 Builder 通信。
"""
from __future__ import annotations

from queue import Queue

from .schemas import EventContract, PriceTick, Signal

price_queue: "Queue[PriceTick]" = Queue(maxsize=10000)
event_queue: "Queue[EventContract]" = Queue(maxsize=1000)
signal_queue: "Queue[Signal]" = Queue(maxsize=1000)

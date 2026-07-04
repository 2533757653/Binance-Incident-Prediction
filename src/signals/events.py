"""
Builder-F · 多事件类型支持（多标的并行信号生成）。

iter-1 只支持 BTC/ETH 双标的；
iter-2 扩展到 4 个标的：BTC / ETH / SOL / DOGE。

设计要点：
1. 每个标的独立一个 SignalGenerator（或 MLStrategy）+ 自己的 K 线数据源
2. 用 concurrent.futures.ThreadPoolExecutor 并发跑 4 个标的
3. MultiSymbolSignalGenerator 接收 (event, klines) 流，对每个 event 按 symbol 分发
4. 输出统一的 Signal 列表（带 underlying 字段透传）
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from src.common.schemas import EventContract, Signal, make_signal_id
from src.common.errors import SignalError

from .generator import SignalGenerator
from .calibration import ConfidenceCalibrator
from .scoring import ScoringConfig

logger = logging.getLogger(__name__)

_TZ_CN = timezone(timedelta(hours=8))


# ============================================================
# 常量：4 个支持的标的
# ============================================================
DEFAULT_SYMBOLS: Tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT")

# 标的 → 起始价（用于合成 K 线 / 数据缺失兜底）
SYMBOL_START_PRICE: Dict[str, float] = {
    "BTCUSDT": 65000.0,
    "ETHUSDT": 3500.0,
    "SOLUSDT": 150.0,
    "DOGEUSDT": 0.15,
}

# 标的 → 底层币种（EventContract.underlying 用）
SYMBOL_UNDERLYING: Dict[str, str] = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
    "SOLUSDT": "SOL",
    "DOGEUSDT": "DOGE",
}


# ============================================================
# 工厂：单标的的 SignalGenerator
# ============================================================
def build_generator_for_symbol(
    symbol: str,
    *,
    scoring_config: Optional[ScoringConfig] = None,
    strategy_name: str = "factor_v1",
    calibrator: Optional[ConfidenceCalibrator] = None,
) -> SignalGenerator:
    """
    给定标的，构造一个对应的 SignalGenerator。
    同一标的的所有事件共享同一个 generator 实例（共享 K 线窗口状态）。
    """
    cfg = scoring_config or ScoringConfig()
    return SignalGenerator(
        scoring_config=cfg,
        strategy_name=strategy_name,
        ttl_seconds=60,
    )


# ============================================================
# MultiSymbolSignalGenerator
# ============================================================
class MultiSymbolSignalGenerator:
    """
    多标的并行信号生成器。

    用法：
        gen = MultiSymbolSignalGenerator(symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"))
        gen.set_klines("BTCUSDT", k1m=..., k1h=..., k4h=...)
        signals = gen.run_batch(events)

    特点：
        - 每个标的独立一个 SignalGenerator（避免 K 线状态互相污染）
        - 4 个标的用 ThreadPoolExecutor 并发
        - 可选 ConfidenceCalibrator 注入到每个 generator（暂未替换 raw confidence）
    """

    def __init__(
        self,
        symbols: Sequence[str] = DEFAULT_SYMBOLS,
        scoring_config: Optional[ScoringConfig] = None,
        strategy_name: str = "factor_v1",
        max_workers: int = 4,
        calibrator: Optional[ConfidenceCalibrator] = None,
    ):
        if not symbols:
            raise ValueError("symbols must not be empty")
        self.symbols: Tuple[str, ...] = tuple(symbols)
        self.scoring_config = scoring_config or ScoringConfig()
        self.strategy_name = strategy_name
        self.max_workers = min(max_workers, len(symbols))
        self.calibrator = calibrator

        # 每个 symbol 一个 SignalGenerator
        self._gens: Dict[str, SignalGenerator] = {
            sym: build_generator_for_symbol(
                sym,
                scoring_config=self.scoring_config,
                strategy_name=strategy_name,
                calibrator=calibrator,
            )
            for sym in self.symbols
        }
        # K 线缓冲：{symbol: {"1m": [...], "1h": [...], "4h": [...]}}
        self._klines: Dict[str, Dict[str, list]] = {sym: {"1m": [], "1h": [], "4h": []} for sym in self.symbols}

    # --------------------------------------------------------
    # K 线管理
    # --------------------------------------------------------
    def set_klines(
        self,
        symbol: str,
        *,
        k1m: Optional[Sequence] = None,
        k1h: Optional[Sequence] = None,
        k4h: Optional[Sequence] = None,
    ) -> None:
        if symbol not in self.symbols:
            raise ValueError(f"symbol {symbol} not in self.symbols={self.symbols}")
        if k1m is not None:
            self._klines[symbol]["1m"] = list(k1m)
        if k1h is not None:
            self._klines[symbol]["1h"] = list(k1h)
        if k4h is not None:
            self._klines[symbol]["4h"] = list(k4h)

    def set_klines_bulk(self, klines_by_symbol: Dict[str, Dict[str, Sequence]]) -> None:
        for sym, ks in klines_by_symbol.items():
            self.set_klines(
                sym,
                k1m=ks.get("1m"),
                k1h=ks.get("1h"),
                k4h=ks.get("4h"),
            )

    # --------------------------------------------------------
    # 分发：把事件按 symbol 分桶
    # --------------------------------------------------------
    def _bucket_by_symbol(self, events: Iterable[EventContract]) -> Dict[str, List[EventContract]]:
        buckets: Dict[str, List[EventContract]] = {sym: [] for sym in self.symbols}
        for ev in events:
            if ev.symbol in buckets:
                buckets[ev.symbol].append(ev)
            else:
                logger.debug("drop event with unsupported symbol %s", ev.symbol)
        return buckets

    # --------------------------------------------------------
    # 单标的：跑一组 events
    # --------------------------------------------------------
    def _run_symbol(
        self,
        symbol: str,
        events: List[EventContract],
        now: Optional[datetime] = None,
    ) -> List[Signal]:
        gen = self._gens[symbol]
        ks = self._klines.get(symbol, {})
        out: List[Signal] = []
        for ev in events:
            try:
                sig = gen.consume_event(
                    event=ev,
                    klines_1m=ks.get("1m", []),
                    klines_1h=ks.get("1h", []),
                    klines_4h=ks.get("4h", []),
                    now=now,
                )
            except Exception as e:
                logger.exception("consume_event failed for %s: %s", ev.event_id, e)
                continue
            if sig is not None:
                # 校准 confidence（如果 calibrator 已 fit）
                if self.calibrator is not None and getattr(self.calibrator, "_fitted", False):
                    raw_c = float(sig.confidence)
                    new_c = self.calibrator.calibrate(raw_c)
                    sig = sig.model_copy(update={"confidence": round(new_c, 6)})
                out.append(sig)
        return out

    # --------------------------------------------------------
    # 并发跑 4 个标的
    # --------------------------------------------------------
    def run_batch(
        self,
        events: Iterable[EventContract],
        now: Optional[datetime] = None,
    ) -> List[Signal]:
        """
        并发跑多标的，返回所有产出的 Signal。
        """
        buckets = self._bucket_by_symbol(events)
        # 跳过空桶
        active = {sym: evs for sym, evs in buckets.items() if evs}
        if not active:
            return []

        results: List[Signal] = []
        if len(active) == 1 or self.max_workers <= 1:
            # 单线程退化
            for sym, evs in active.items():
                results.extend(self._run_symbol(sym, evs, now=now))
            return results

        # 并发
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            future_to_sym = {
                ex.submit(self._run_symbol, sym, evs, now): sym
                for sym, evs in active.items()
            }
            for fut in as_completed(future_to_sym):
                sym = future_to_sym[fut]
                try:
                    out = fut.result()
                except Exception as e:
                    logger.exception("run_symbol(%s) failed: %s", sym, e)
                    continue
                results.extend(out)

        # 按 ts 排序，方便下游消费
        results.sort(key=lambda s: s.ts)
        return results

    # --------------------------------------------------------
    # 异步：从 event_queue 持续消费
    # --------------------------------------------------------
    def run_forever(self, poll_timeout: float = 1.0) -> None:
        """
        阻塞循环：从 src.common.bus.event_queue 读，按 symbol 分桶 → run_batch。
        """
        from src.common.bus import event_queue, signal_queue
        logger.info("MultiSymbolSignalGenerator run_forever started (symbols=%s)", self.symbols)
        while True:
            try:
                # 非阻塞拉一批
                events: List[EventContract] = []
                while True:
                    try:
                        ev = event_queue.get_nowait()
                    except Exception:
                        break
                    events.append(ev)
                    if len(events) >= 64:
                        break
                if not events:
                    # 短暂 idle
                    import time
                    time.sleep(poll_timeout)
                    continue
                sigs = self.run_batch(events)
                for sig in sigs:
                    signal_queue.put(sig)
                    logger.info("published %s %s conf=%.3f ev=%.3f", sig.signal_id, sig.side, sig.confidence, sig.expected_value)
            except SignalError as e:
                logger.exception("signal bus error: %s", e)
                continue
            except KeyboardInterrupt:
                logger.info("run_forever interrupted")
                return

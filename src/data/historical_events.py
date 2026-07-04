"""历史结算 → 事件合约转换（Builder-D 产出 · iter-2）。

把 ``exerciseHistory`` 的每条记录（已结算的期权）转成 ``EventContract``：

- 解析 symbol 格式 ``BTC-260624-66500-C`` → underlying / expiry / strike / side
- C（Call）→ direction="ABOVE"，P（Put）→ direction="BELOW"
- 隐含 yes 概率：基于 ``realStrikePrice`` vs ``strikePrice`` 的差距（logit 缩放）
- snapshot 时间 = expiryDate（事件已到期，作为历史快照）

限制（task spec）：
- 只保留 4 个标的的记录：BTC / ETH / SOL / DOGE

**派生字段约定**（沿用 dataContract §1.3）：
- ``strikeResult`` 不在 EventContract schema 里 → 留在 raw dict（不进 bus）
- ``time_to_expiry`` 从 expiryDate 距今的秒数粗略映射到 5m/15m/1h/4h/1d
- ``current_yes_price`` = 隐含 YES 概率（视作"snapshot 时的中间价"）
- ``current_no_price`` = 1 - current_yes_price
- 价差 = 0.05（用固定 tick 模拟历史价差，因 eapi exerciseHistory 不返回 bid/ask）
- status = "EXPIRED"（历史数据不进入实时交易流）
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

from pydantic import ValidationError

from src.common.errors import FeedError
from src.common.schemas import EventContract
from src.data.exercise_history import (
    ExerciseHistoryFetcher,
    _now_sh,  # noqa: PLC2701 -- 复用时区工具
    fetch_exercise_history,
)

log = logging.getLogger(__name__)

# iter-2 焦点标的
_ALLOWED_UNDERLYINGS = {"BTC", "ETH", "SOL", "DOGE"}

# symbol 解析正则：{UNDERLYING}-{YYMMDD}-{STRIKE}-{C|P}
# 例：BTC-260624-66500-C
_SYMBOL_RE = re.compile(
    r"^(?P<under>[A-Z]{2,5})-(?P<date>\d{6})-(?P<strike>\d+(?:\.\d+)?)-(?P<side>[CP])$"
)

# 行权结果枚举（实测 2026-06-24）
_RESULT_EXPIRED = "EXTRINSIC_VALUE_EXPIRED"   # Call 行权成功（real > strike）
_RESULT_STRICKEN = "REALISTIC_VALUE_STRICKEN"  # Put 行权成功（real < strike）

# 隐含概率计算参数（logit 缩放）
_PROB_SCALE = 5.0      # edge 缩放系数
_PROB_CLIP_LO = 0.05
_PROB_CLIP_HI = 0.95
_FIXED_SPREAD = 0.05   # 历史快照统一价差（eapi 不返回 bid/ask）

# 时区
_TZ_SH = timezone(timedelta(hours=8))


# ============================================================
# symbol 解析
# ============================================================
def _parse_symbol(symbol: str) -> Optional[dict]:
    """解析 ``BTC-260624-66500-C`` 格式。

    返回 ``{underlying, expiry_date, strike, side}``，解析失败返回 None。
    """
    m = _SYMBOL_RE.match(symbol.strip().upper())
    if not m:
        return None
    underlying = m.group("under")
    if underlying not in _ALLOWED_UNDERLYINGS:
        return None
    date_str = m.group("date")  # YYMMDD
    try:
        # YYMMDD → 20YY-MM-DD
        year = 2000 + int(date_str[0:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        expiry = datetime(year, month, day, tzinfo=_TZ_SH).replace(hour=8)  # 默认 08:00 UTC+8 结算
    except (ValueError, IndexError):
        return None
    try:
        strike = float(m.group("strike"))
    except ValueError:
        return None
    side = m.group("side")  # C / P
    return {
        "underlying": underlying,
        "expiry": expiry,
        "strike": strike,
        "side": side,
    }


# ============================================================
# 字段映射
# ============================================================
def _infer_time_to_expiry(expiry: datetime) -> str:
    """根据 expiry 距 today 的天数粗略推断 time_to_expiry。"""
    days = (expiry - _now_sh()).days
    # 历史事件都是过去到期 → "1d" 占位
    if days < 0:
        # 历史事件，统一标记为 1d（已结算）
        return "1d"
    # 未来事件：按剩余天数映射
    if days < 1:
        return "4h"
    if days < 7:
        return "1d"
    return "1d"


def _implied_yes_prob(direction: str, strike: float, real_price: float) -> float:
    """根据方向 + edge 算隐含 YES 概率。

    - ABOVE (Call): edge = (real - strike) / strike
    - BELOW (Put):  edge = (strike - real) / strike
    - yes_prob = clip(0.5 + edge * scale, lo, hi)
    """
    if strike <= 0:
        return 0.5
    if direction == "ABOVE":
        edge = (real_price - strike) / strike
    else:  # BELOW
        edge = (strike - real_price) / strike
    prob = 0.5 + edge * _PROB_SCALE
    return max(_PROB_CLIP_LO, min(_PROB_CLIP_HI, prob))


def _build_event_id(underlying: str, expiry: datetime, strike: float, direction: str) -> str:
    """dataContract §0.7：{SYMBOL}-{TF}-{DIRECTION}-{STRIKE}-{TS}。

    strike 格式：>= 1 用整数；< 1 用原值（保留小数），如 DOGE 0.15。
    """
    symbol = f"{underlying}USDT"
    tf = _infer_time_to_expiry(expiry)
    ts = expiry.strftime("%Y%m%d%H%M%S")
    if strike >= 1:
        strike_str = f"{int(strike)}"
    else:
        strike_str = f"{strike:g}"  # 保留有效位小数
    return f"{symbol}-{tf}-{direction}-{strike_str}-{ts}"


def _row_to_event(row: dict) -> Optional[EventContract]:
    """把一条 eapi exerciseHistory 记录转成 EventContract。"""
    try:
        symbol = (row.get("symbol") or "").strip().upper()
        parsed = _parse_symbol(symbol)
        if parsed is None:
            return None

        # 必需字段
        try:
            strike = float(row["strikePrice"])
            real_price = float(row["realStrikePrice"])
            expiry_ms = int(row["expiryDate"])
        except (KeyError, ValueError, TypeError) as e:
            log.debug("[historical_events] 缺字段: %s | %s", e, row)
            return None

        # 用毫秒戳为准；如缺则用 symbol 解析的 expiry
        if expiry_ms > 0:
            settle_time = datetime.fromtimestamp(expiry_ms / 1000, tz=_TZ_SH)
        else:
            settle_time = parsed["expiry"]

        side = parsed["side"]
        direction = "ABOVE" if side == "C" else "BELOW"
        tf = _infer_time_explicit(settle_time)
        yes_prob = _implied_yes_prob(direction, strike, real_price)
        no_prob = round(1.0 - yes_prob, 4)

        # 价差用固定 tick（eapi 不返回 bid/ask）
        yes_bid = max(0.01, round(yes_prob - _FIXED_SPREAD / 2, 4))
        yes_ask = min(0.99, round(yes_prob + _FIXED_SPREAD / 2, 4))
        no_bid = max(0.01, round(no_prob - _FIXED_SPREAD / 2, 4))
        no_ask = min(0.99, round(no_prob + _FIXED_SPREAD / 2, 4))

        # 数据校验：bid < ask（否则 schema 抛错）
        if yes_ask <= yes_bid:
            yes_ask = round(yes_bid + 0.01, 4)
        if no_ask <= no_bid:
            no_ask = round(no_bid + 0.01, 4)

        return EventContract(
            event_id=_build_event_id(parsed["underlying"], settle_time, strike, direction),
            symbol=f"{parsed['underlying']}USDT",
            title=(
                f"{parsed['underlying']} 到期时 {'≥' if direction == 'ABOVE' else '<'} "
                f"{int(strike)}? (real={real_price:.2f})"
            ),
            underlying=parsed["underlying"],
            strike_price=strike,
            direction=direction,
            time_to_expiry=tf,
            settle_time=settle_time,
            current_yes_price=round(yes_prob, 4),
            current_no_price=no_prob,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            spread=round(yes_ask - yes_bid, 4),
            volume_24h=0.0,
            open_interest=0.0,
            status="EXPIRED",
        )
    except ValidationError as e:
        log.debug("[historical_events] 校验失败: %s | row=%s", e, row)
        return None
    except Exception as e:  # noqa: BLE001
        log.warning("[historical_events] 转换异常: %s | row=%s", e, row)
        return None


def _infer_time_explicit(settle_time: datetime) -> str:
    """显式映射：根据 settle_time 距 now 的差值，映射到 5m/15m/1h/4h/1d。

    历史数据（settle_time < now）→ 全部映射为 1d。
    """
    now = _now_sh()
    delta_sec = (settle_time - now).total_seconds()
    if delta_sec < 0:
        return "1d"
    if delta_sec < 5 * 60:
        return "5m"
    if delta_sec < 15 * 60:
        return "15m"
    if delta_sec < 30 * 60:
        return "30m"
    if delta_sec < 60 * 60:
        return "1h"
    if delta_sec < 4 * 60 * 60:
        return "4h"
    return "1d"


# ============================================================
# Builder
# ============================================================
class HistoricalEventBuilder:
    """把 exerciseHistory 原始记录转成 EventContract 流。

    用法::

        b = HistoricalEventBuilder()
        events = b.build()          # list[EventContract]，按 settle_time 倒序
        top = b.build(top_n=50)
    """

    def __init__(self, fetcher: Optional[ExerciseHistoryFetcher] = None) -> None:
        self.fetcher = fetcher or ExerciseHistoryFetcher()

    def close(self) -> None:
        try:
            self.fetcher.close()
        except Exception:  # noqa: BLE001
            pass

    # ---------------------------------------------------------- 公开
    def build(self, *, top_n: Optional[int] = None, force: bool = False) -> list[EventContract]:
        """返回 ``list[EventContract]``，按 settle_time 倒序。

        :param top_n: 限制返回条数（None=全部）
        :param force: 强制重拉网络（忽略缓存）
        """
        try:
            rows = self.fetcher.fetch(force=force)
        except FeedError as e:
            log.error("[historical_events] 拉取失败: %s", e)
            raise

        out: list[EventContract] = []
        for row in rows:
            ev = _row_to_event(row)
            if ev is not None:
                out.append(ev)

        # 按 settle_time 倒序
        out.sort(key=lambda e: e.settle_time, reverse=True)
        if top_n is not None and top_n > 0:
            out = out[:top_n]
        log.info("[historical_events] 转换完成：%d 条 → EventContract", len(out))
        return out


# ============================================================
# 便捷函数
# ============================================================
def build_historical_events(
    *, top_n: Optional[int] = None, force: bool = False
) -> list[EventContract]:
    """快捷函数：拉 + 转换 + 关闭。"""
    builder = HistoricalEventBuilder()
    try:
        return builder.build(top_n=top_n, force=force)
    finally:
        builder.close()


__all__ = [
    "HistoricalEventBuilder",
    "build_historical_events",
    "fetch_exercise_history",
]

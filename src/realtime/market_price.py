"""
真实盘口价：从币安 eapi 期权 ticker + mark 接口，推算"未来 1h 事件"的隐含概率。

由于事件合约（1h 二元）无公开 API，用 eapi 期权（同 strike 同方向）作为参考：
- 找 ATM（at-the-money）期权：strike 接近当前价
- 用 BS 公式 + markIV 反推 prob(S > K at T)
- 这个 prob 就是市场对该事件的"隐含 YES 概率"

数据源：https://eapi.binance.com/eapi/v1/ticker + /mark
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import httpx

log = logging.getLogger(__name__)

_TZ_CN = timezone(timedelta(hours=8))


@dataclass
class OptionTicker:
    symbol: str               # e.g. "BTC-260626-70000-C"
    underlying: str           # "BTCUSDT"
    side: str                 # "CALL" / "PUT"
    strike: float
    expiry_ms: int
    bid_price: float          # USDT
    ask_price: float          # USDT
    last_price: float
    mark_iv: float            # 隐含波动率（来自 /mark）


def _norm_cdf(x: float) -> float:
    """标准正态 CDF。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_prob_itm(S: float, K: float, T_years: float, sigma: float, r: float = 0.0) -> float:
    """
    Black-Scholes CALL 价格 → prob(S > K at expiry) = N(d2)

    S: 现价
    K: 行权价
    T_years: 到期时间（年化）
    sigma: 隐含波动率
    r: 无风险利率（默认 0，crypto 短期事件近似）

    返回：YES 概率（ITM 概率）= N(d2)
    """
    if T_years <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d2 = (math.log(S / K) - 0.5 * sigma * sigma * T_years) / (sigma * math.sqrt(T_years))
    return _norm_cdf(d2)


def bs_put_prob_itm(S: float, K: float, T_years: float, sigma: float, r: float = 0.0) -> float:
    """PUT ITM 概率 = 1 - N(-d2) = N(d2)。"""
    if T_years <= 0 or sigma <= 0:
        return 1.0 if S < K else 0.0
    d2 = (math.log(S / K) - 0.5 * sigma * sigma * T_years) / (sigma * math.sqrt(T_years))
    return _norm_cdf(-d2)


def fetch_option_tickers(timeout: float = 8.0, max_retries: int = 3) -> List[dict]:
    """拉所有期权 ticker（带 429 重试）。"""
    import time
    last_err = None
    for attempt in range(max_retries):
        try:
            r = httpx.get("https://eapi.binance.com/eapi/v1/ticker", timeout=timeout)
            if r.status_code == 429:
                wait = 2 ** attempt
                log.warning("[market_price] ticker 429, %ds 后重试", wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise last_err if last_err else RuntimeError("ticker fetch failed")


def parse_underlying(symbol: str) -> str:
    """从 eapi symbol 解析 underlying: 'ETH-260626-3000-C' → 'ETHUSDT'."""
    base = symbol.split("-")[0]
    return base + "USDT"


def parse_side(symbol: str) -> str:
    """从 eapi symbol 解析 side: 'ETH-260626-3000-C' → 'CALL' / 'PUT'."""
    suffix = symbol.split("-")[-1]
    return "CALL" if suffix == "C" else ("PUT" if suffix == "P" else "?")


def fetch_option_marks(timeout: float = 8.0) -> dict:
    """拉 mark 数据（包含 markIV）。"""
    r = httpx.get("https://eapi.binance.com/eapi/v1/mark", timeout=timeout)
    r.raise_for_status()
    rows = r.json()
    return {m["symbol"]: m for m in rows}


def fetch_exchange_info(timeout: float = 8.0) -> dict:
    """拉 exchangeInfo（包含 optionSymbols 含 expireDate）。"""
    r = httpx.get("https://eapi.binance.com/eapi/v1/exchangeInfo", timeout=timeout)
    r.raise_for_status()
    body = r.json()
    return {s["symbol"]: s for s in body.get("optionSymbols", [])}


def find_atm_call(
    current_price: float,
    tickers: List[dict],
    marks: dict,
    exchange_info: dict,
    underlying: str,
    *,
    target_minutes: int = 60,
    skew_pct: float = 0.05,
) -> Optional[OptionTicker]:
    """
    找 ATM CALL：strike 接近 current_price × (1+skew) 且 expiry 接近 target_minutes。
    skew_pct: 略偏 ITM，让期权有正向价值
    """
    target_strike = current_price * (1.0 + skew_pct)
    now_ms = int(datetime.now(tz=_TZ_CN).timestamp() * 1000)
    target_expiry_ms = now_ms + target_minutes * 60 * 1000
    best = None
    best_dist = float("inf")
    for t in tickers:
        sym = t.get("symbol", "")
        if parse_side(sym) != "CALL":
            continue
        if parse_underlying(sym) != underlying:
            continue
        # 从 exchange_info 拿 expireDate
        info = exchange_info.get(sym, {})
        try:
            strike = float(t.get("strikePrice", 0))
            expiry = int(info.get("expiryDate", 0))
        except (TypeError, ValueError):
            continue
        if strike <= 0 or expiry <= now_ms:
            continue
        strike_dist = abs(strike - target_strike) / max(target_strike, 1.0)
        expiry_dist = abs(expiry - target_expiry_ms) / max(target_expiry_ms - now_ms, 1.0)
        total_dist = strike_dist + expiry_dist * 2
        if total_dist < best_dist:
            mark = marks.get(sym, {})
            best = OptionTicker(
                symbol=sym,
                underlying=underlying,
                side="CALL",
                strike=strike,
                expiry_ms=expiry,
                bid_price=float(t.get("bidPrice", 0) or 0),
                ask_price=float(t.get("askPrice", 0) or 0),
                last_price=float(t.get("lastPrice", 0) or 0),
                mark_iv=float(mark.get("markIV", 0) or 0) if mark else 0.0,
            )
            best_dist = total_dist
    return best


def implied_prob_for_event(
    current_price: float,
    event_strike: float,
    event_minutes: int,
    *,
    underlying: str = "BTCUSDT",
) -> Optional[dict]:
    """
    用 eapi 期权数据反推事件合约的"市场隐含概率"。

    返回 dict:
      implied_prob_yes: 0~1
      source: "eapi option {symbol}"
      atm_option: OptionTicker（如果找到）
    """
    try:
        tickers = fetch_option_tickers()
        log.debug("[market_price] tickers: %d", len(tickers))
        marks = fetch_option_marks()
        log.debug("[market_price] marks: %d", len(marks))
        exchange_info = fetch_exchange_info()
        log.debug("[market_price] info: %d", len(exchange_info))
    except Exception as e:
        import traceback
        log.warning("[market_price] 拉期权数据失败: %s", e)
        log.warning("[market_price] trace: %s", traceback.format_exc())
        return None

    atm = find_atm_call(current_price, tickers, marks, exchange_info, underlying,
                        target_minutes=event_minutes, skew_pct=0.02)
    if atm is None or atm.mark_iv <= 0:
        log.debug("[market_price] %s 没找到合适的 ATM 期权", underlying)
        return None

    # 计算到期时间（年化）
    T_seconds = (atm.expiry_ms - int(datetime.now(tz=_TZ_CN).timestamp() * 1000)) / 1000.0
    T_years = max(T_seconds / (365.0 * 24 * 3600), 1e-9)
    prob = bs_call_prob_itm(current_price, event_strike, T_years, atm.mark_iv)
    return {
        "implied_prob_yes": float(prob),
        "atm_option": atm.symbol,
        "atm_strike": atm.strike,
        "atm_iv": atm.mark_iv,
        "atm_expiry_hours": T_seconds / 3600.0,
        "source": "eapi_option_BS",
    }
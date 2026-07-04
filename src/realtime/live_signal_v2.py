"""
实时信号推送 v2（生产级）：
- 30m + 1h 双窗口
- 90 天训练的 XGBoost 模型
- 生产级阈值 0.65/0.35
- 信号去重（TTL=30 分钟，与事件窗口一致）
- 真实盘口价（eapi 期权 BS 反推）
- plyer Windows 弹窗
- 错误处理：429 重试 / 断网降级 / 信号特征缺失跳过
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.backtest.features_v4 import FEATURE_NAMES_V4 as FEATURE_NAMES, compute_features_v4 as compute_features
from src.backtest.real_data import build_aggregated_klines, fetch_klines_paginated
from src.common.schemas import EventContract, Signal, make_signal_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("horizon.live.v2")

_TZ_CN = timezone(timedelta(hours=8))


def now_sh() -> datetime:
    return datetime.now(tz=_TZ_CN)


def load_model(path: Path):
    data = joblib.load(path)
    return data["model"], data["feature_names"]


# ============================================================
# 数据接入（30m + 1h 双窗口）
# ============================================================
# ============================================================
# 数据接入（双轨：直连 + 代理 fallback）
# ============================================================
_PROXY_URL = os.environ.get("HORIZON_PROXY", "")  # 启动时由 main() 设置
_PROXY_FAIL_COUNT: dict[str, int] = {}  # host -> 连续失败次数

# 退出码
EXIT_PROXY_UNREACHABLE = 12  # 代理配置了但端口不通


def _detect_system_proxy_windows() -> str:
    """读 Windows 注册表里的系统代理设置（Clash 默认走这里）。"""
    if sys.platform != "win32":
        return ""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings") as key:
            try:
                enable, _ = winreg.QueryValueEx(key, "ProxyEnable")
            except FileNotFoundError:
                return ""
            if enable != 1:
                return ""
            try:
                server, _ = winreg.QueryValueEx(key, "ProxyServer")
                return f"http://{server}" if not server.startswith("http") else server
            except FileNotFoundError:
                return ""
    except Exception as e:
        log.debug("[live] 读 Windows 系统代理失败: %s", e)
        return ""


def _make_client(timeout: float = 10.0, use_proxy: bool = False) -> httpx.Client:
    """构造 httpx 客户端（可选代理）。"""
    if use_proxy and _PROXY_URL:
        try:
            transport = httpx.HTTPTransport(proxy=httpx.Proxy(url=_PROXY_URL))
            return httpx.Client(timeout=timeout, transport=transport, follow_redirects=True)
        except Exception as e:
            log.warning("[live] 构造代理 client 失败: %s，回退到直连", e)
    return httpx.Client(timeout=timeout, follow_redirects=True)


def _proxy_healthcheck(proxy_url: str, timeout: float = 5.0,
                       retries: int = 2) -> tuple[bool, str]:
    """通过代理发真实 HTTPS GET 验证代理可用。

    只 ping 端口（TCP）能查出 Clash 没开，但查不出"端口开了但规则没匹配"
    或者"代理服务器抽风 HTTPS 转发失败"。所以发个真实请求过去。

    自身重试 ``retries`` 次（间隔 1 秒），容忍代理瞬时抖动，
    真死透了才报错退出。最坏耗时 ~ (timeout+1) * (retries+1)。
    """
    last_err = ""
    for i in range(retries + 1):
        try:
            transport = httpx.HTTPTransport(proxy=httpx.Proxy(url=proxy_url))
            with httpx.Client(timeout=timeout, transport=transport, follow_redirects=True) as c:
                r = c.get("https://api.binance.com/api/v3/ping")
                if r.status_code == 200:
                    return True, "binance ping 200 OK（通过代理）"
                last_err = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
        if i < retries:
            time.sleep(1)
    return False, last_err or "all retries failed"


def fetch_klines(symbol: str, interval: str, limit: int = 200, max_retries: int = 8) -> list[dict]:
    """
    iter-9j: 增量 K 线 + 长 retry 容忍（8 次 + backoff 60s，最坏 ~9 分钟）。

    第一次跑：全量拉 200 根，写缓存
    之后：startTime 只拉缓存之后的 1~3 根
    """
    host = "api.binance.com"
    from src.realtime.incremental_klines import fetch_klines_incremental

    last_err = None
    for attempt in range(max_retries):
        use_proxy = bool(_PROXY_URL)
        try:
            data = fetch_klines_incremental(
                symbol, interval,
                limit=limit,
                proxy_url=_PROXY_URL if use_proxy else "",
                timeout=20.0,
            )
            _PROXY_FAIL_COUNT[host] = 0
            return data
        except Exception as e:
            last_err = e
            _PROXY_FAIL_COUNT[host] = _PROXY_FAIL_COUNT.get(host, 0) + 1
            # iter-9j: 长 backoff（2s/4s/8s/16s/30s/45s/60s/60s）
            wait = min(60, 2 ** (attempt + 1))
            log.warning("[live] %s %s 拉取失败（attempt %d/%d）: %s，%ds 后重试",
                        symbol, interval, attempt + 1, max_retries, e, wait)
            time.sleep(wait)
    raise last_err if last_err else RuntimeError("fetch failed")


def fetch_option_data(endpoint: str, *, cache_ttl_seconds: int = 60) -> list:
    """iter-9i: eapi 端点带短 TTL 缓存（默认 60s）。"""
    from src.realtime.incremental_klines import fetch_eapi_incremental
    return fetch_eapi_incremental(
        endpoint,
        proxy_url=_PROXY_URL,
        timeout=20.0,
        cache_ttl_seconds=cache_ttl_seconds,
    )


def get_features_now(symbol: str, model, feature_names: list,
                     enable_toast: bool = False) -> Optional[tuple[list[float], dict]]:
    """
    拉实时 K 线，计算当前 30m 事件的特征。

    iter-9i: 用增量 K 线缓存 + eapi 短 TTL 缓存，单次 Clash 请求 1~3 根（不再 200 根）。
    """
    try:
        k30_raw = fetch_klines(symbol, "30m", limit=200)
        time.sleep(0.5)
        k1h_raw = fetch_klines(symbol, "1h", limit=200)
    except Exception as e:
        log.error("[live] " + "#" * 64)
        log.error("[live] ### %s 数据拉取彻底失败（已重试）：%s", symbol, e)
        log.error("[live] ### 本轮跳过该符号，下一轮再试。")
        log.error("[live] ### 可能原因：Clash 端口饱和 / GFW 抖动 / 币安限流")
        log.error("[live] " + "#" * 64)
        _fire_data_failure_toast(symbol, str(e), enable_toast=enable_toast)
        return None

    if len(k30_raw) < 30 or len(k1h_raw) < 24:
        return None

    closes_30m = np.array([float(k[4]) for k in k30_raw], dtype=float)
    highs_30m = np.array([float(k[2]) for k in k30_raw], dtype=float)
    lows_30m = np.array([float(k[3]) for k in k30_raw], dtype=float)
    volumes_30m = np.array([float(k[5]) for k in k30_raw], dtype=float)
    closes_1h = np.array([float(k[4]) for k in k1h_raw], dtype=float)
    highs_1h = np.array([float(k[2]) for k in k1h_raw], dtype=float)
    lows_1h = np.array([float(k[3]) for k in k1h_raw], dtype=float)
    volumes_1h = np.array([float(k[5]) for k in k1h_raw], dtype=float)

    i = len(closes_30m) - 1
    # iter-14: 用 v4 15 特征（事件合约二元方向：涨/跌）
    # 4h 聚合（v4 接口需要 closes_4h，但 lite 版不用——传个空数组）
    closes_4h = np.array([closes_30m[0]])  # dummy, v4 lite 不使用
    features = compute_features(
        closes_30m, highs_30m, lows_30m, volumes_30m,
        closes_1h, highs_1h, lows_1h, volumes_1h,
        closes_4h,
        i_30m=i,
    )
    current_price = float(closes_30m[i])
    return features, {
        "k30_count": len(k30_raw),
        "k1h_count": len(k1h_raw),
        "current_price": current_price,
    }


def predict_prob(model, features: list[float]) -> float:
    X = np.array([features])
    return float(model.predict_proba(X)[0, 1])


# ============================================================
# 信号构造
# ============================================================
def construct_event(
    symbol: str,
    current_price: float,
    strike_offset_pct: float = 0.02,
    event_minutes: int = 30,
) -> EventContract:
    now = now_sh()
    strike = round(current_price * (1.0 + strike_offset_pct), 2)
    settle_time = now + timedelta(minutes=event_minutes)
    yes_mid = 0.5
    no_mid = 0.5
    spread = 0.02
    tf_field = "1h" if event_minutes == 30 else f"{event_minutes // 60}h"
    return EventContract(
        event_id=f"{symbol}-{tf_field}-LIVE-{int(strike)}-{now.strftime('%Y%m%d%H%M%S')}",
        symbol=symbol,
        title=f"{symbol[:3]} {event_minutes}m 后 {'≥' if strike>current_price else '<'} {int(strike)}?",
        underlying=symbol[:3],
        strike_price=strike,
        direction="ABOVE",
        time_to_expiry=tf_field,
        settle_time=settle_time,
        current_yes_price=yes_mid,
        current_no_price=no_mid,
        yes_bid=yes_mid - spread / 2,
        yes_ask=yes_mid + spread / 2,
        no_bid=no_mid - spread / 2,
        no_ask=no_mid + spread / 2,
        spread=spread,
        volume_24h=0.0,
        open_interest=0.0,
        status="TRADING",
    )


def build_signal(
    ev: EventContract,
    side: str,
    prob: float,
    features: list[float],
    feature_names: list[str],
    *,
    strategy: str = "ml_xgb_v4",
    ttl_seconds: int = 1800,  # 30 分钟（事件窗口）
) -> Signal:
    now = now_sh()
    factors = {name: round(float(v), 6) for name, v in zip(feature_names, features)}
    factors["time_to_settle_min"] = round((ev.settle_time - now).total_seconds() / 60.0, 2)
    confidence = max(prob, 1 - prob)
    entry = ev.current_yes_price if side == "YES" else ev.current_no_price
    payoff_ratio = (1.0 - entry) / max(entry, 0.01)
    win_prob = prob if side == "YES" else (1 - prob)
    ev_val = win_prob * payoff_ratio - (1 - win_prob)
    rationale = (
        f"v2 ML prob_yes={prob:.2f} → "
        f"ret_30m={factors.get('ret_30m',0)*100:+.2f}% "
        f"trend_24h={factors.get('trend_24h',0)*100:+.2f}%"
    )
    return Signal(
        signal_id=make_signal_id(ev.event_id, now),
        ts=now,
        event_id=ev.event_id,
        event_title=ev.title,
        symbol=ev.symbol,
        side=side,
        entry_price=round(float(entry), 4),
        spread=round(float(ev.spread), 4),
        confidence=round(float(confidence), 4),
        expected_value=round(float(ev_val), 4),
        win_prob=round(float(win_prob), 4),
        payoff_ratio=round(float(payoff_ratio), 4),
        rationale=rationale[:80],
        strategy=strategy,
        factors=factors,
        ttl_seconds=ttl_seconds,
        expire_at=now + timedelta(seconds=ttl_seconds),
    )


# ============================================================
# 信号去重
# ============================================================
class SignalDedup:
    def __init__(self, ttl_seconds: int = 1800):
        self.ttl_seconds = ttl_seconds
        self._last: dict[tuple[str, str], tuple[float, float]] = {}

    def should_emit(self, symbol: str, side: str, confidence: float, now: float = None) -> tuple[bool, str]:
        now = now if now is not None else time.time()
        key = (symbol, side)
        last = self._last.get(key)
        if last is None:
            self._last[key] = (now, confidence)
            return True, "fresh"
        last_ts, last_conf = last
        if now - last_ts > self.ttl_seconds:
            self._last[key] = (now, confidence)
            return True, "ttl_expired"
        if confidence > last_conf:
            self._last[key] = (now, confidence)
            return True, "higher_conf"
        return False, "duplicate_lower"


# ============================================================
# 真实盘口价
# ============================================================
def get_market_implied(symbol: str, event_strike: float, event_minutes: int = 30) -> Optional[dict]:
    """eapi ATM 期权 BS 反推 implied_prob（iter-9i: 带 60s 短 TTL 缓存）。"""
    try:
        from src.realtime.market_price import (
            fetch_option_tickers, fetch_option_marks, fetch_exchange_info,
            find_atm_call, bs_call_prob_itm,
        )
        # iter-9i: 用短 TTL 缓存避免重复拉（60s 内复用）
        from src.realtime.incremental_klines import fetch_eapi_incremental
        tickers = fetch_eapi_incremental("/ticker", proxy_url=_PROXY_URL, cache_ttl_seconds=60)
        marks = fetch_eapi_incremental("/mark", proxy_url=_PROXY_URL, cache_ttl_seconds=60)
        info = fetch_eapi_incremental("/exchangeInfo", proxy_url=_PROXY_URL, cache_ttl_seconds=300)
        atm = find_atm_call(event_strike * 0.98, tickers, marks, info, symbol,
                            target_minutes=event_minutes, skew_pct=0.0)
        if atm is None or atm.mark_iv <= 0:
            return None
        T_seconds = (atm.expiry_ms - int(now_sh().timestamp() * 1000)) / 1000.0
        T_years = max(T_seconds / (365.0 * 24 * 3600), 1e-9)
        prob = bs_call_prob_itm(event_strike * 0.98, event_strike, T_years, atm.mark_iv)
        return {
            "implied_prob_yes": float(prob),
            "atm_option": atm.symbol,
            "atm_iv": atm.mark_iv,
        }
    except Exception as e:
        log.debug("[market] 拉期权数据失败: %s", e)
        return None


# ============================================================
# 弹窗
# ============================================================
def fire_toast(sig: Signal) -> bool:
    """plyer 弹窗。返回是否成功。"""
    try:
        from plyer import notification
        notification.notify(
            title=f"[{sig.side}] {sig.symbol} @ conf={sig.confidence:.2f}",
            message=f"{sig.event_title}\nentry={sig.entry_price} ev={sig.expected_value:+.2f}",
            app_name="Horizon-Incident",
            timeout=8,
        )
        return True
    except Exception as e:
        log.warning("[toast] plyer 失败: %s", e)
        return False


# 数据失败 toast 防刷：每个 symbol 5 分钟最多弹一次
_DATA_FAIL_TOAST_COOLDOWN = 300  # 秒
_data_fail_last_ts: dict[str, float] = {}


def _fire_data_failure_toast(symbol: str, err: str, enable_toast: bool = False) -> None:
    """数据拉取彻底失败时弹一个红色 toast（5 分钟内同符号不重复）。"""
    if not enable_toast:
        return
    now = time.time()
    last = _data_fail_last_ts.get(symbol, 0)
    if now - last < _DATA_FAIL_TOAST_COOLDOWN:
        log.info("[toast] %s 数据失败 toast 冷却中（%.0fs 内不重复）", symbol, _DATA_FAIL_TOAST_COOLDOWN - (now - last))
        return
    try:
        from plyer import notification
        short_err = err.replace("\n", " ")[:140]
        notification.notify(
            title=f"[Horizon] {symbol} 数据拿不到",
            message=f"已重试 3 次仍失败\n{short_err}\n本轮跳过，下轮再试",
            app_name="Horizon-Incident",
            timeout=10,
        )
        _data_fail_last_ts[symbol] = now
        log.info("[toast] 已弹数据失败窗: %s", symbol)
    except Exception as e:
        log.warning("[toast] 数据失败 toast 失败: %s", e)


def fire_notify(sig: Signal, sig_record: dict) -> None:
    """弹窗通知（仅本地，iter-9f 后无邮件）。"""
    # iter-13: 弹窗阈值提到 0.75（与决策阈值一致）
    if sig.confidence >= 0.75:
        fire_toast(sig)


# ============================================================
# 主循环
# ============================================================
_stop_event = threading.Event()


def run_live(
    symbols: list[str],
    model_paths: dict[str, Path],
    *,
    poll_seconds: int = 0,  # 0 = 自动按 event_minutes 取
    strike_offset_pct: float = 0.02,
    event_minutes: int = 30,
    min_prob_yes: float = 0.65,
    max_prob_yes: float = 0.35,
    log_dir: str = "logs",
    use_real_market: bool = True,
    enable_toast: bool = True,
    proxy_url: str = "",
):
    global _PROXY_URL
    _PROXY_URL = proxy_url or _PROXY_URL
    # 自动检测 Windows 系统代理
    if not _PROXY_URL:
        auto = _detect_system_proxy_windows()
        if auto:
            _PROXY_URL = auto
            log.info("[live] 自动检测到 Windows 系统代理: %s", auto)
    if _PROXY_URL:
        log.info("[live] 代理模式: %s", _PROXY_URL)
        log.info("[live] 注：未做启动健康检查（首次拉数据失败会自动 fallback 重试）")
    else:
        log.warning("[live] 直连模式（无代理）—— 币安 eapi 在国内大概率连不上，建议开代理")
    # iter-7: 扫描节奏对齐事件窗口
    if poll_seconds <= 0:
        poll_seconds = 120
        log.info("[live] 自动选择扫描间隔: %d 秒（2 分钟）", poll_seconds)
    dedup = SignalDedup(ttl_seconds=event_minutes * 60)
    log.info("[live] 去重 TTL = %d 秒 (= 事件窗口 %dm)", event_minutes * 60, event_minutes)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "signals_v2.jsonl"

    models = {}
    for sym in symbols:
        if sym not in model_paths:
            log.error("[live] %s 无模型路径", sym)
            continue
        m, fn = load_model(model_paths[sym])
        models[sym] = (m, fn)
        log.info("[live] 加载模型 %s (%d 特征)", sym, len(fn))

    print(f"\n{'='*70}")
    print(f"  Horizon-Incident Live Signal v2")
    print(f"  ─────────────────────────────────────────")
    print(f"  Symbols:    {', '.join(symbols)}")
    print(f"  Window:     {event_minutes}m  |  Scan: {poll_seconds}s  |  TTL: {event_minutes}m")
    print(f"  Thresholds: YES >= {min_prob_yes}  |  NO <= {max_prob_yes}")
    print(f"  Features:   market_price={use_real_market}  toast={enable_toast}")
    print(f"  Log:        {log_path}")
    print(f"  Heartbeat:  every 30s (so you know it's alive)")
    print(f"  Exit:       Ctrl+C or close this window")
    print(f"{'='*70}\n")

    cycle = 0
    while not _stop_event.is_set():
        cycle += 1
        cycle_start = now_sh()
        print(f"\n[{cycle_start.strftime('%H:%M:%S')}] === 第 {cycle} 轮扫描 ===")
        for sym in symbols:
            if sym not in models:
                continue
            try:
                result = get_features_now(sym, *models[sym], enable_toast=enable_toast)
            except Exception as e:
                log.error("[live] %s get_features 失败: %s", sym, e)
                continue
            if result is None:
                continue
            features, meta = result
            model, fn = models[sym]
            prob = predict_prob(model, features)
            current_price = meta["current_price"]

            # 真实盘口
            strike = current_price * (1.0 + strike_offset_pct)
            market = None
            if use_real_market:
                market = get_market_implied(sym, strike, event_minutes)

            # 决策
            if prob >= min_prob_yes:
                side = "YES"
            elif prob <= max_prob_yes:
                side = "NO"
            else:
                mkt_str = f" mkt_implied={market['implied_prob_yes']:.2f}" if market else ""
                print(f"  [{sym}] HOLD price={current_price:.2f} ml_prob={prob:.2f}{mkt_str}")
                continue

            # iter-13: ADX 过滤（震荡市不打）
            adx = features_dict.get("adx_1h", 25.0) if 'features_dict' in dir() else 25.0
            try:
                adx = float(features[fn.index("adx_1h")])
            except Exception:
                adx = 25.0
            if adx < 20:
                print(f"  [{sym}] {side} ml_prob={prob:.2f} ADX={adx:.1f} → SKIP（震荡市 ADX<20）")
                continue

            # 构造信号 + 去重
            ev = construct_event(sym, current_price, strike_offset_pct=strike_offset_pct, event_minutes=event_minutes)
            sig = build_signal(ev, side, prob, features, fn, ttl_seconds=event_minutes * 60)
            emit, reason = dedup.should_emit(sym, side, sig.confidence)
            if not emit:
                print(f"  [{sym}] {side} conf={sig.confidence:.2f} → 去重 ({reason})")
                continue

            side_mark = "[+YES]" if side == "YES" else "[-NO ]"
            conf_mark = " HIGH" if sig.confidence >= 0.7 else ""
            mkt_str = ""
            if market:
                mkt_str = f" mkt={market['implied_prob_yes']:.2f} IV={market['atm_iv']:.2f}"
            print(f"  {side_mark} [{sym}] {side} entry={sig.entry_price:.2f} "
                  f"ml_prob={prob:.2f} conf={sig.confidence:.2f} ev={sig.expected_value:+.2f}"
                  f"{mkt_str}{conf_mark}  ({reason})")
            print(f"      reason: {sig.rationale}")

            # 写日志
            record = {
                "ts": cycle_start.isoformat(),
                "symbol": sym,
                "side": side,
                "entry_price": sig.entry_price,
                "confidence": sig.confidence,
                "expected_value": sig.expected_value,
                "ml_prob_yes": prob,
                "dedup_reason": reason,
                "rationale": sig.rationale,
                "current_price": current_price,
                "factors": sig.factors,
            }
            if market:
                record["market"] = market
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # 弹窗 + 邮件（高置信度 + 通过去重）
            if enable_toast and sig.confidence >= 0.75:
                fire_notify(sig, record)

        # 等待下一轮（带心跳，让用户知道程序还活着）
        if _stop_event.is_set():
            break
        # 分段 sleep + 心跳显示（让用户看到倒计时）
        heartbeat_interval = max(5, poll_seconds // 12)  # 至少每 5 秒刷一次，最长 30s
        for remaining in range(poll_seconds, 0, -1):
            if _stop_event.is_set():
                break
            # 显示心跳（倒计时 + 时间戳 + 模型状态）
            if remaining % heartbeat_interval == 0 or remaining <= 10:
                mins, secs = divmod(remaining, 60)
                ts_now = datetime.now(tz=timezone(timedelta(hours=8))).strftime('%H:%M:%S')
                sys.stdout.write(
                    f"\r  [{ts_now}] "
                    f"alive | next scan in {mins:02d}:{secs:02d} | "
                    f"models=BTC+ETH | Ctrl+C to stop   "
                )
                sys.stdout.flush()
            time.sleep(1)
        # 心跳行清掉
        sys.stdout.write("\r" + " " * 100 + "\r")
        sys.stdout.flush()
        if _stop_event.is_set():
            break
        print(f"  [{now_sh().strftime('%H:%M:%S')}] → 进入下一轮扫描")


def main(argv=None) -> int:
    print("\033[2J\033[H", end="")  # 清屏，让 banner 干净
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--models", nargs="+", default=[
        "proofs/iter-14/BTCUSDT_xgb_v4_full.joblib",
        "proofs/iter-14/ETHUSDT_xgb_v4_full.joblib",
    ])
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--strike-offset", type=float, default=0.02)
    parser.add_argument("--event-minutes", type=int, default=30, choices=[30, 60])
    # iter-13: 默认阈值从 0.65/0.35 → 0.75/0.25（提高胜率、降低频率）
    parser.add_argument("--min-prob", type=float, default=0.80)
    parser.add_argument("--max-prob", type=float, default=0.20)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-market", action="store_true")
    parser.add_argument("--no-toast", action="store_true")
    parser.add_argument("--proxy", type=str, default=os.environ.get("HORIZON_PROXY", ""),
                        help="代理 URL（如 http://127.0.0.1:7890），空=直连")
    args = parser.parse_args(argv)

    model_paths = {}
    for mp in args.models:
        for sym in args.symbols:
            if sym in mp:
                model_paths[sym] = Path(mp)

    def handler(signum, frame):
        log.info("[live] 收到停止信号，优雅退出")
        _stop_event.set()
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    # Windows: 关 CMD 窗口 = 强制 kill，捕获不到信号。所以 run_live.bat 用 `pause` 让用户看到退出原因
    # 如果用户强制关窗，Python 进程会被立即终止，但下次双击会重新启动

    if args.once:
        # 单轮模式：用临时 stop event
        def stop_after():
            time.sleep(2)
            _stop_event.set()
        threading.Thread(target=stop_after, daemon=True).start()

    run_live(
        symbols=args.symbols,
        model_paths=model_paths,
        poll_seconds=args.interval,
        strike_offset_pct=args.strike_offset,
        event_minutes=args.event_minutes,
        min_prob_yes=args.min_prob,
        max_prob_yes=args.max_prob,
        use_real_market=not args.no_market,
        enable_toast=not args.no_toast,
        proxy_url=args.proxy,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
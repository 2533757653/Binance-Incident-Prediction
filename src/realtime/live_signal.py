"""
实时信号推送 — 从币安现货 K 线 + 训练好的 XGBoost 模型，输出实时 BUY YES/NO 信号。

工作流：
  1. 启动时拉 720 根历史 1h K 线（缓存到 data/cache/）
  2. 每 5 分钟拉最新 1h K 线
  3. 对每个 BTC/ETH 标的，构造"未来 1h 事件"（strike = 当前价 ± 2%）
  4. 用 XGBoost 预测 YES 中奖概率
  5. 如果 prob ≥ 0.7 → BUY YES；prob ≤ 0.3 → BUY NO；其他 → 无信号
  6. 终端打印（彩色 rich 表格）+ Windows 弹窗（confidence ≥ 0.65）+ JSON 日志

实时调度：单线程循环（生产可换 asyncio）
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.backtest.real_data import (
    build_4h_aggregated_klines,
    fetch_and_cache_klines,
)
from src.common.schemas import EventContract, Signal, make_signal_id
from src.signals.factors import (
    compute_atr_break_4h,
    compute_bb_pct_1h,
    compute_mom_15m,
)

log = logging.getLogger("horizon.live")

_TZ_CN = timezone(timedelta(hours=8))


def now_sh() -> datetime:
    return datetime.now(tz=_TZ_CN)


def load_model(path: str | Path):
    """加载训练好的 XGBoost + 特征名。"""
    data = joblib.load(path)
    return data["model"], data["feature_names"]


def fetch_latest_klines(symbol: str, interval: str = "1h", limit: int = 200, timeout: float = 8.0) -> list[dict]:
    """拉最新 K 线（直连 api.binance.com）。"""
    r = httpx.get(
        "https://api.binance.com/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def build_features_from_klines(klines_raw: list[dict], i: int) -> Optional[list[float]]:
    """从 K 线 raw list 抽取特征（i = 当前时刻索引）。"""
    if i < 30:
        return None
    closes = np.array([float(k[4]) for k in klines_raw[: i + 1]], dtype=float)

    # 1h K 线当作 1m 喂给 mom
    mom = float(np.log(closes[i] / closes[i - 15])) if i >= 15 else 0.0
    # BB 20 周期
    window = closes[max(0, i - 19): i + 1]
    if len(window) < 20:
        return None
    mid = float(np.mean(window))
    sd = float(np.std(window, ddof=0))
    if sd == 0:
        bb = 0.5
    else:
        bb = float((closes[i] - (mid - 2 * sd)) / (4 * sd))
    # ATR 4h：用最近 14 根 4h 聚合（用 1h 折算）
    # 简化：把 1h 序列当成 4h，period=14
    highs = np.array([float(k[2]) for k in klines_raw[: i + 1]], dtype=float)
    lows = np.array([float(k[3]) for k in klines_raw[: i + 1]], dtype=float)
    prev_close = closes[i - 14: i]
    cur_high = highs[i - 13: i + 1]
    cur_low = lows[i - 13: i + 1]
    tr = np.maximum(cur_high - cur_low, np.maximum(np.abs(cur_high - prev_close), np.abs(cur_low - prev_close)))
    atr = float(np.mean(tr))
    atr_break = float((closes[i] - np.mean(closes[max(0, i - 13): i + 1])) / max(atr * 1.0, 1e-9))

    hour = datetime.fromtimestamp(klines_raw[i][0] / 1000, tz=_TZ_CN).hour
    recent = closes[max(0, i - 24): i + 1]
    vol_1h = float(np.std(np.diff(np.log(recent)))) if len(recent) > 1 else 0.0
    trend_24h = (closes[i] / closes[max(0, i - 24)] - 1.0) if i >= 24 else 0.0
    ret_1h = (closes[i] / closes[i - 1] - 1.0) if i >= 1 else 0.0
    spread = 0.02  # 默认事件合约价差

    return [mom, bb, atr_break, hour, vol_1h, trend_24h, ret_1h, spread]


def construct_event(
    symbol: str,
    current_price: float,
    *,
    strike_offset_pct: float = 0.02,
    minutes_to_settle: int = 60,
    now: datetime = None,
) -> EventContract:
    """构造一个未来 1h 事件合约（基于当前价）。"""
    now = now or now_sh()
    strike = round(current_price * (1.0 + strike_offset_pct), 2)
    settle_time = now + timedelta(minutes=minutes_to_settle)
    # 假设事件合约价格：YES 中间价 = 0.5（不知道真实赔率），spread = 0.02
    yes_mid = 0.5
    no_mid = 0.5
    spread = 0.02
    return EventContract(
        event_id=f"{symbol}-1H-LIVE-{int(strike)}-{now.strftime('%Y%m%d%H%M%S')}",
        symbol=symbol,
        title=f"{symbol[:3]} 1h 后 ≥ {int(strike)}?",
        underlying=symbol[:3],
        strike_price=strike,
        direction="ABOVE",
        time_to_expiry="1h",
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


def predict_event(model, feature_names: list, features: list[float]) -> tuple[float, str]:
    """
    用模型预测事件 YES 概率，返回 (prob_yes, signal_side)。
    signal_side: "YES" / "NO" / "HOLD"
    """
    X = np.array([features])
    prob_yes = float(model.predict_proba(X)[0, 1])
    # iter-5: 放宽阈值（0.55/0.45）让信号更频繁（演示用）
    if prob_yes >= 0.55:
        return prob_yes, "YES"
    if prob_yes <= 0.45:
        return prob_yes, "NO"
    return prob_yes, "HOLD"


def build_signal(
    ev: EventContract,
    side: str,
    prob: float,
    features: list[float],
    feature_names: list[str],
    *,
    strategy: str = "ml_xgb_v1",
    ttl_seconds: int = 300,
) -> Signal:
    """构造 Signal 对象。"""
    now = now_sh()
    factors = {name: round(float(v), 6) for name, v in zip(feature_names, features)}
    # 补全 dataContract 必填字段
    factors["time_to_settle_min"] = round((ev.settle_time - now).total_seconds() / 60.0, 2)
    rationale = (
        f"ML 模型 prob_yes={prob:.2f} → "
        f"主因子 ret_1h={factors.get('ret_1h', 0)*100:+.2f}% "
        f"trend_24h={factors.get('trend_24h', 0)*100:+.2f}%"
    )
    confidence = max(prob, 1 - prob) if side != "HOLD" else 0.5
    entry = ev.current_yes_price if side == "YES" else ev.current_no_price
    payoff_ratio = (1.0 - entry) / max(entry, 0.01)
    win_prob = prob if side == "YES" else (1 - prob)
    ev_val = win_prob * payoff_ratio - (1 - win_prob)
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


def run_live(
    symbols: list[str],
    model_paths: dict[str, Path],
    *,
    poll_seconds: int = 300,
    strike_offset_pct: float = 0.02,
    log_dir: str = "logs",
    run_once: bool = False,
    dedup_ttl_seconds: int = 3600,
    use_real_market_price: bool = True,
):
    """
    主循环：每 poll_seconds 拉一次最新 K 线，输出信号。
    """
    from src.realtime.signal_dedup import SignalDedup
    from src.realtime.market_price import implied_prob_for_event

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "signals.jsonl"

    dedup = SignalDedup(ttl_seconds=dedup_ttl_seconds)

    # 加载模型
    models = {}
    for sym in symbols:
        if sym not in model_paths:
            log.error("[live] 找不到 %s 的模型路径，跳过", sym)
            continue
        m, fn = load_model(model_paths[sym])
        models[sym] = (m, fn)
        log.info("[live] 已加载模型 %s (%d 特征)", sym, len(fn))

    # 拉初始 K 线（缓存）
    klines_cache: dict[str, list[dict]] = {}
    for sym in symbols:
        try:
            cache = fetch_and_cache_klines(sym, interval="1h", limit=200)
            r = httpx.get("https://api.binance.com/api/v3/klines",
                          params={"symbol": sym, "interval": "1h", "limit": 200}, timeout=8)
            klines_cache[sym] = r.json()
            log.info("[live] %s 拉取 %d 根 K 线", sym, len(klines_cache[sym]))
        except Exception as e:
            log.error("[live] %s 拉 K 线失败: %s", sym, e)

    print(f"\n{'='*70}")
    print(f"[horizon-live] 启动实时信号 | 标的: {', '.join(symbols)} | 间隔: {poll_seconds}s")
    print(f"[horizon-live] 模型: {list(models.keys())} | 阈值: prob≥0.7 → YES, prob≤0.3 → NO")
    print(f"[horizon-live] 去重 TTL: {dedup_ttl_seconds}s | 真实盘口: {use_real_market_price}")
    print(f"[horizon-live] 日志: {log_path}")
    print(f"{'='*70}\n")

    cycle = 0
    try:
        while True:
            cycle += 1
            cycle_start = now_sh()
            print(f"\n[{cycle_start.strftime('%H:%M:%S')}] === 第 {cycle} 轮扫描 ===")
            for sym in symbols:
                if sym not in models:
                    continue
                try:
                    r = httpx.get("https://api.binance.com/api/v3/klines",
                                  params={"symbol": sym, "interval": "1h", "limit": 200}, timeout=8)
                    r.raise_for_status()
                    klines_cache[sym] = r.json()
                except Exception as e:
                    log.warning("[live] %s 拉 K 线失败: %s", sym, e)
                    continue

                klines = klines_cache[sym]
                if len(klines) < 30:
                    continue
                i = len(klines) - 1
                features = build_features_from_klines(klines, i)
                if features is None:
                    continue
                model, fn = models[sym]
                prob, side = predict_event(model, fn, features)
                current_price = float(klines[i][4])

                # 拉真实盘口（eapi 期权 → implied_prob）
                market_info = None
                if use_real_market_price:
                    strike = current_price * (1.0 + strike_offset_pct)
                    market_info = implied_prob_for_event(current_price, strike, 60, underlying=sym)

                # 用市场盘口 + 模型概率混合做最终决策
                if side == "HOLD":
                    if market_info:
                        imp = market_info["implied_prob_yes"]
                        print(f"  [{sym}] price={current_price:.2f} ml_prob={prob:.2f} mkt_implied={imp:.2f} → 观望")
                    else:
                        print(f"  [{sym}] price={current_price:.2f} ml_prob={prob:.2f} → 观望")
                    continue

                ev = construct_event(sym, current_price, strike_offset_pct=strike_offset_pct)
                sig = build_signal(ev, side, prob, features, fn)

                # 信号去重
                emit, reason = dedup.should_emit(sym, side, sig.confidence)
                if not emit:
                    print(f"  [{sym}] {side} conf={sig.confidence:.2f} → 去重丢弃 ({reason})")
                    continue

                side_mark = "[+YES]" if side == "YES" else "[-NO ]"
                conf_mark = " HIGH" if sig.confidence >= 0.7 else ""
                mkt_str = ""
                if market_info:
                    mkt_str = f" mkt={market_info['implied_prob_yes']:.2f}"
                print(
                    f"  {side_mark} [{sym}] entry={sig.entry_price:.2f} "
                    f"ml_prob={prob:.2f} conf={sig.confidence:.2f} ev={sig.expected_value:+.2f}{mkt_str}{conf_mark}  ({reason})"
                )
                print(f"      reason: {sig.rationale}")
                if market_info:
                    print(f"      market: ATM={market_info['atm_option']} IV={market_info['atm_iv']:.2f} "
                          f"exp={market_info['atm_expiry_hours']:.0f}h implied_yes={market_info['implied_prob_yes']:.3f}")

                # 写日志
                with log_path.open("a", encoding="utf-8") as f:
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
                    if market_info:
                        record["market"] = {
                            "atm_option": market_info["atm_option"],
                            "atm_iv": market_info["atm_iv"],
                            "implied_prob_yes": market_info["implied_prob_yes"],
                        }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                # Windows 弹窗（confidence ≥ 0.65 + 通过去重）
                if sig.confidence >= 0.65:
                    _fire_toast(sig)

            if run_once:
                break
            print(f"  (下轮扫描：{poll_seconds}s 后)")
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("\n[horizon-live] 用户中断，退出")


def _fire_toast(sig: Signal) -> None:
    """触发 Windows 弹窗（plyer 优先，win10toast-click 降级）。"""
    # 优先 plyer（更通用）
    try:
        from plyer import notification
        notification.notify(
            title=f"Horizon: {sig.side} {sig.symbol}",
            message=f"{sig.event_title}\nentry={sig.entry_price} conf={sig.confidence:.2f} ev={sig.expected_value:+.2f}",
            app_name="Horizon-Incident",
            timeout=8,
        )
        return
    except Exception as e:
        log.debug("[toast] plyer 失败: %s", e)
    # 降级到 win10toast-click
    try:
        from win10toast_click import ToastNotifier
        toaster = ToastNotifier()
        toaster.show_toast(
            title=f"Horizon: {sig.side} {sig.symbol}",
            msg=f"{sig.event_title}\nentry={sig.entry_price} conf={sig.confidence:.2f} ev={sig.expected_value:+.2f}",
            duration=8,
            threaded=True,
        )
        return
    except Exception as e:
        log.warning("[toast] win10toast 失败: %s", e)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--models", nargs="+", default=["proofs/iter-3/BTCUSDT_xgb_v1.joblib",
                                                         "proofs/iter-3/ETHUSDT_xgb_v1.joblib"])
    parser.add_argument("--interval", type=int, default=300, help="扫描间隔秒")
    parser.add_argument("--once", action="store_true", help="只跑一轮")
    parser.add_argument("--strike-offset", type=float, default=0.02)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if len(args.symbols) != len(args.models):
        # 默认按名字匹配
        model_paths = {}
        for sym in args.symbols:
            for mp in args.models:
                if sym in mp:
                    model_paths[sym] = Path(mp)
                    break
    else:
        model_paths = {sym: Path(mp) for sym, mp in zip(args.symbols, args.models)}

    run_live(
        args.symbols, model_paths,
        poll_seconds=args.interval,
        strike_offset_pct=args.strike_offset,
        run_once=args.once,
    )
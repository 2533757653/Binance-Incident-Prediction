#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实时信号 v4 —— 一体化观察窗（因果特征 / 无泄露 / 无 Binance）。

一个窗口全搞定：
- 双击启动 → 每 5 分钟扫描 BTC/ETH，打印心跳 + 交易日志
- 有把握的信号 → 弹窗提醒
- 每隔固定时间 → 自动核对历史信号对错，显示真实战绩 vs 保本线
- 关闭窗口 → 进程随之结束（前台运行，无需单独停止）

真实胜率(1年 walk-forward)：门槛0.66 时 BTC/ETH ~55%；保本 54%(盈亏比0.85)。
⚠ 只提示不下单；边际优势，务必小仓、看长期。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal as _signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")  # 保证中文不乱码
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.schemas import KlineBar
from src.data.sources import fetch_klines_rows
from src.features.causal_features import compute_live
from src.output.email_alert import EmailAlerter

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
log = logging.getLogger("horizon.live.v4")

_TZ = timezone(timedelta(hours=8))
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "proofs" / "honest"
SIGNAL_LOG_PATH = MODEL_DIR / "live_signals_v4.jsonl"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
TARGET = "1h"
HORIZON_MIN = {"30m": 30, "1h": 60}
SCAN_INTERVAL = 300          # 5 分钟一扫
DEDUP_SECONDS = 1800         # 同向信号 30 分钟不重复
LIVE_DAYS = 15               # 拉 15 天 5m 供指标预热 + 信号核对
B = 0.85
P_STAR = 1 / (1 + B)         # 保本胜率 54.05%


class C:
    G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; CY = "\033[96m"
    DIM = "\033[90m"; B_ = "\033[1m"; X = "\033[0m"


def now_sh() -> datetime:
    return datetime.now(tz=_TZ)


def entry_guidance(side: str, price: float, ts, horizon_min: int = 60) -> str:
    """给一条信号的入场指引：参考价 + 别追红线 + 尽快时限 + 结算时间。"""
    buf = price * 0.002  # 0.2% 缓冲（给晚几分钟的入场留窗口，又不至于追太远）
    if side.startswith("NO"):   # 预测跌：别在已经跌破后追
        cond = f"现价≥{price - buf:.0f} 可进（跌破就别追）"
    else:                        # 预测涨：别在已经涨过后追
        cond = f"现价≤{price + buf:.0f} 可进（涨过就别追）"
    until = (ts + timedelta(minutes=15)).strftime("%H:%M")
    settle = (ts + timedelta(minutes=horizon_min)).strftime("%H:%M")
    return f"参考价{price:.0f}｜{cond}｜{until}前尽快进｜约{settle}结算"


def _rows_to_klines(rows, symbol):
    out = []
    for k in rows:
        out.append(KlineBar(
            symbol=symbol, interval="5m",
            open_time=datetime.fromtimestamp(int(k[0]) / 1000, tz=_TZ),
            close_time=datetime.fromtimestamp(int(k[6]) / 1000, tz=_TZ),
            open=float(k[1]), high=float(k[2]), low=float(k[3]), close=float(k[4]),
            volume=float(k[5]), quote_volume=float(k[7]), trades_count=int(k[8])))
    return out


def _fire_toast(title, msg):
    try:
        from plyer import notification
        notification.notify(title=title, message=msg, app_name="Horizon-Incident", timeout=10)
    except Exception:
        pass


class LiveSignalV4:
    def __init__(self, threshold=0.66, enable_toast=True, enable_email=True, scoreboard_every=1800):
        self.threshold = threshold
        self.enable_toast = enable_toast
        self.scoreboard_every = scoreboard_every
        self.running = True
        self.models = {}
        self.dedup = {}
        self.history = []          # 每条: {ts,symbol,side,prob,price,target,resolved,won}
        self._last_board = 0.0
        _m = EmailAlerter() if enable_email else None
        self.mailer = _m if (_m and _m.enabled) else None

    # ── 载入 ──
    def load_models(self):
        ok = True
        for sym in SYMBOLS:
            p = MODEL_DIR / f"{sym}_{TARGET}.joblib"
            if not p.exists():
                print(f"{C.R}✗ 模型缺失 {p}（先跑 train_honest.py）{C.X}")
                ok = False
                continue
            self.models[sym] = joblib.load(p)["model"]
        return ok

    def load_history(self):
        if not SIGNAL_LOG_PATH.exists():
            return
        for line in SIGNAL_LOG_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                d.setdefault("resolved", False)
                d.setdefault("won", None)
                self.history.append(d)
            except Exception:
                pass

    def _save_history(self):
        """把信号 + 核对结果整体写盘（核对一次就永久记下，重启不丢）。"""
        try:
            with open(SIGNAL_LOG_PATH, "w", encoding="utf-8") as f:
                for h in self.history:
                    f.write(json.dumps(h, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ── 主循环 ──
    def run_forever(self, probe=0):
        print(C.B_ + "=" * 60 + C.X)
        print(f"{C.B_}  Horizon 事件合约信号 · 一体化观察窗{C.X}")
        print(f"  币种 {'/'.join(s.replace('USDT','') for s in SYMBOLS)} | 周期 {TARGET} | 门槛 ±{self.threshold}")
        print(f"  {C.DIM}保本胜率 {P_STAR*100:.1f}%(盈亏比{B})；只在很有把握时提示{C.X}")
        _ways = (["弹窗"] if self.enable_toast else []) + ([f"邮件→{self.mailer.to}"] if self.mailer else [])
        print(f"  {C.DIM}提醒方式：{' + '.join(_ways) or '仅窗口显示'}{C.X}")
        print(f"  {C.Y}⚠ 只提示不下单，边际优势，小仓观察。关闭本窗口即停止。{C.X}")
        print(C.B_ + "=" * 60 + C.X)
        self.load_history()
        if self.history:
            self._scoreboard(price_map={})  # 开场先显示历史战绩
        rounds = 0
        while self.running:
            t0 = time.time()
            price_map = self._scan()
            self._resolve(price_map)
            if time.time() - self._last_board >= self.scoreboard_every:
                self._scoreboard(price_map)
                self._last_board = time.time()
            rounds += 1
            if probe and rounds >= probe:
                self._scoreboard(price_map)
                break
            # 睡到下一个 5 分钟 K 线收盘后（对齐时钟边界 + 15s，确保每次都拿到新 K 线）
            next_wake = (int(time.time()) // SCAN_INTERVAL + 1) * SCAN_INTERVAL + 15
            while self.running and time.time() < next_wake:
                time.sleep(1)

    # ── 单次扫描 ──
    def _scan(self):
        price_map = {}
        print(f"\n{C.DIM}——— {now_sh():%m-%d %H:%M:%S} 扫描 ———{C.X}")
        for sym in SYMBOLS:
            if sym not in self.models:
                continue
            try:
                rows = fetch_klines_rows(sym, "5m", days=LIVE_DAYS)
            except Exception as e:
                print(f"  {C.R}{sym[:3]} 数据失败: {e}{C.X}")
                continue
            k5 = _rows_to_klines(rows, sym)[:-1]   # 丢弃正在形成的当根
            if len(k5) < 80:
                continue
            price_map[sym] = (np.array([int(b.open_time.timestamp() * 1000) for b in k5], dtype=np.int64),
                              np.array([b.close for b in k5], float))
            x, _ = compute_live(k5)
            prob = float(self.models[sym].predict_proba(x.reshape(1, -1))[0, 1])
            price = k5[-1].close
            self._decide(sym, prob, price)

        pend = sum(1 for h in self.history if not h.get("resolved"))
        done = [h for h in self.history if h.get("resolved")]
        wr = (sum(1 for h in done if h["won"]) / len(done)) if done else 0
        wrs = f"{wr*100:.0f}%" if done else "—"
        print(f"  {C.DIM}待核对 {pend} 条 | 已核对 {len(done)} 条 胜率 {wrs}{C.X}")
        return price_map

    def _decide(self, sym, prob, price):
        s = sym.replace("USDT", "")
        if prob >= self.threshold:
            side = "YES(涨)"; conf = prob
        elif prob <= 1 - self.threshold:
            side = "NO(跌)"; conf = 1 - prob
        else:
            print(f"  {s:4s} p={prob:.3f}  现价 {price:<10.2f}{C.DIM}无信号{C.X}")
            return
        key = f"{sym}_{side}"
        if key in self.dedup and time.time() - self.dedup[key] < DEDUP_SECONDS:
            print(f"  {s:4s} p={prob:.3f}  现价 {price:<10.2f}{C.DIM}({side} 已提示,去重){C.X}")
            return
        self.dedup[key] = time.time()
        color = C.G if "YES" in side else C.R
        print(f"  {C.B_}{color}★ {s:4s} {side:8s} 置信 {conf:.2f}  现价 {price:.2f}  p={prob:.3f}{C.X}")
        print(f"       {C.CY}↳ 入场：{entry_guidance(side, price, now_sh(), HORIZON_MIN.get(TARGET, 60))}{C.X}")
        if self.enable_toast:
            threading.Thread(target=_fire_toast,
                             args=(f"Horizon {s}", f"{side} 置信{conf:.2f} 现价{price:.0f}"),
                             daemon=True).start()
        if self.mailer:
            _subj = f"Horizon 信号 {s} {side} 置信{conf:.2f}"
            _body = (f"币种：{s}\n方向：{side}\n置信：{conf:.2f}\n现价：{price:.2f}\n"
                     f"入场指引：{entry_guidance(side, price, now_sh(), HORIZON_MIN.get(TARGET, 60))}\n"
                     f"模型概率 p：{prob:.3f}\n周期：{TARGET}\n"
                     f"时间：{now_sh():%Y-%m-%d %H:%M:%S}\n\n"
                     f"提示：保本胜率{P_STAR*100:.0f}%，仅供参考，小仓观察，不构成投资建议。")
            threading.Thread(target=self.mailer.send, args=(_subj, _body), daemon=True).start()
        entry = {"ts": now_sh().isoformat(), "symbol": sym, "target": TARGET,
                 "side": side, "confidence": round(conf, 4), "prob": round(prob, 4),
                 "price": price, "resolved": False, "won": None}
        self.history.append(entry)
        self._save_history()

    # ── 核对历史信号 ──
    def _resolve(self, price_map):
        now_ms = time.time() * 1000
        changed = False
        for h in self.history:
            if h.get("resolved"):
                continue
            sym = h["symbol"]
            hz = HORIZON_MIN.get(h.get("target", "1h"), 60)
            rms = datetime.fromisoformat(h["ts"]).timestamp() * 1000 + hz * 60000
            if now_ms < rms or sym not in price_map:
                continue
            oms, cl = price_map[sym]
            i = int(np.searchsorted(oms, int(rms), side="left"))
            if i >= len(cl):
                continue
            after = cl[i]
            is_yes = h["side"].startswith("YES")
            h["won"] = bool(after > h["price"]) if is_yes else bool(after < h["price"])
            h["price_after"] = round(float(after), 2)
            h["resolved"] = True
            changed = True
        if changed:
            self._save_history()  # 核对结果落盘

    # ── 战绩板 ──
    def _scoreboard(self, price_map):
        self._resolve(price_map)
        done = [h for h in self.history if h.get("resolved")]
        print(f"\n{C.B_}{C.CY}╔═══ 战绩板（保本 {P_STAR*100:.1f}%）═══╗{C.X}")
        for sym in SYMBOLS:
            d = [h for h in done if h["symbol"] == sym]
            n = len([h for h in self.history if h["symbol"] == sym])
            w = sum(1 for h in d if h["won"])
            if d:
                wr = w / len(d)
                ev = (1.85 * wr - 1) * 100
                col = C.G if wr > P_STAR else (C.Y if wr > P_STAR - 0.01 else C.R)
                print(f"  {sym.replace('USDT',''):4s} 信号{n:3d} 已核对{len(d):3d} "
                      f"胜率{col}{wr*100:5.1f}%{C.X} EV/笔{ev:+.1f}%")
            else:
                print(f"  {sym.replace('USDT',''):4s} 信号{n:3d} 已核对  0 {C.DIM}(还没到核对时间){C.X}")
        alld = done
        if alld:
            wr = sum(1 for h in alld if h["won"]) / len(alld)
            verdict = f"{C.G}✅盈利区{C.X}" if wr > P_STAR else (f"{C.Y}⚖约保本{C.X}" if wr > P_STAR - 0.01 else f"{C.R}❌亏损区{C.X}")
            note = f"  {C.DIM}(样本{len(alld)}条,<30看个乐){C.X}" if len(alld) < 30 else ""
            print(f"  {C.B_}合计 已核对{len(alld)} 胜率{wr*100:.1f}% → {verdict}{C.X}{note}")
        else:
            print(f"  {C.DIM}暂无已核对信号，挂着跑，信号到点(1h)后自动核对。{C.X}")
        # 最近信号
        recent = sorted(self.history, key=lambda h: h["ts"], reverse=True)[:6]
        if recent:
            print(f"  {C.DIM}最近：{C.X}")
            for h in recent:
                st = ("✓" if h["won"] else "✗") if h.get("resolved") else "…"
                t = datetime.fromisoformat(h["ts"]).strftime("%m-%d %H:%M")
                print(f"    {t} {h['symbol'].replace('USDT',''):4s} {h['side']:8s} 置信{h.get('confidence',0):.2f} {st}")
        print(f"{C.B_}{C.CY}╚{'═'*28}╝{C.X}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-toast", action="store_true")
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.66)
    ap.add_argument("--board-min", type=float, default=30, help="每隔几分钟显示战绩板")
    ap.add_argument("--probe", type=int, default=0)
    args = ap.parse_args()

    eng = LiveSignalV4(threshold=args.threshold, enable_toast=not args.no_toast,
                       enable_email=not args.no_email, scoreboard_every=args.board_min * 60)
    if not eng.load_models():
        input("按回车退出...")
        sys.exit(1)

    def _stop(sig, frame):
        eng.running = False
    _signal.signal(_signal.SIGINT, _stop)
    try:
        _signal.signal(_signal.SIGTERM, _stop)
    except Exception:
        pass

    try:
        eng.run_forever(probe=args.probe)
    except KeyboardInterrupt:
        pass
    print(f"\n{C.DIM}已停止。{C.X}")


if __name__ == "__main__":
    main()

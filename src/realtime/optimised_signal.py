#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""优化版信号（高层策略层，继承现有引擎，不修改任何底层文件）。

在 live_signal_v4 之上叠加：
1) BTC/ETH 协同过滤：只在两个币【同向】时才出信号（回测胜率更高、更少而精）。
2) 平注 + 凯利建议：纸面资金曲线用平注，另给凯利公式建议下注比例（不做马丁）。

独立日志 optimised_signals.jsonl，独立窗口 optimised.bat；horizon.bat 原封不动。
"""
from __future__ import annotations

import argparse
import json
import signal as _signal
import sys
import threading
import time

import numpy as np

# —— 复用底层，全部 import，不改动 ——
from src.realtime.live_signal_v4 import (
    LiveSignalV4, C, now_sh, _rows_to_klines, _fire_toast, entry_guidance,
    fetch_klines_rows, compute_live,
    SYMBOLS, TARGET, HORIZON_MIN, B, P_STAR,
    SCAN_INTERVAL, DEDUP_SECONDS, LIVE_DAYS, MODEL_DIR,
)

OPT_LOG = MODEL_DIR / "optimised_signals.jsonl"


def _dir(prob: float, t: float) -> int:
    return 1 if prob >= t else (-1 if prob <= 1 - t else 0)


class OptimisedSignal(LiveSignalV4):
    def __init__(self, threshold=0.62, **kw):
        super().__init__(threshold=threshold, **kw)
        self.log_path = OPT_LOG          # 独立日志

    # —— 日志改用独立文件（覆盖底层，不影响 horizon 的日志）——
    def load_history(self):
        if not self.log_path.exists():
            return
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                d.setdefault("resolved", False); d.setdefault("won", None)
                self.history.append(d)
            except Exception:
                pass

    def _save_history(self):
        try:
            with open(self.log_path, "w", encoding="utf-8") as f:
                for h in self.history:
                    f.write(json.dumps(h, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # —— 协同扫描：先算两个币，再联合决策 ——
    def _scan(self):
        print(f"\n{C.DIM}——— {now_sh():%m-%d %H:%M:%S} 扫描（协同版）———{C.X}")
        price_map, info = {}, {}
        for sym in SYMBOLS:
            if sym not in self.models:
                continue
            try:
                rows = fetch_klines_rows(sym, "5m", days=LIVE_DAYS)
            except Exception as e:
                print(f"  {C.R}{sym[:3]} 数据失败: {e}{C.X}")
                continue
            k5 = _rows_to_klines(rows, sym)[:-1]
            if len(k5) < 80:
                continue
            price_map[sym] = (np.array([int(b.open_time.timestamp() * 1000) for b in k5], dtype=np.int64),
                              np.array([b.close for b in k5], float))
            x, _ = compute_live(k5)
            prob = float(self.models[sym].predict_proba(x.reshape(1, -1))[0, 1])
            info[sym] = (prob, k5[-1].close)

        dirs = {}
        for sym in SYMBOLS:
            if sym in info:
                prob, price = info[sym]
                d = _dir(prob, self.threshold)
                dirs[sym] = d
                tag = f"{C.G}涨↑{C.X}" if d > 0 else (f"{C.R}跌↓{C.X}" if d < 0 else f"{C.DIM}—{C.X}")
                print(f"  {sym.replace('USDT',''):4s} p={prob:.3f}  现价 {price:<10.2f} 倾向 {tag}")

        ds = [dirs.get(s, 0) for s in SYMBOLS]
        if len(ds) >= 2 and ds[0] != 0 and all(d == ds[0] for d in ds):
            d = ds[0]
            word = "YES(涨)" if d > 0 else "NO(跌)"
            col = C.G if d > 0 else C.R
            print(f"  {C.B_}{col}★ 协同信号！BTC+ETH 同向 {word}{C.X}")
            for sym in SYMBOLS:
                self._emit_coord(sym, info[sym][0], info[sym][1], d)
        else:
            print(f"  {C.DIM}未协同（两币方向不一致）→ 观望{C.X}")

        pend = sum(1 for h in self.history if not h.get("resolved"))
        done = [h for h in self.history if h.get("resolved")]
        wr = (sum(1 for h in done if h["won"]) / len(done)) if done else 0
        print(f"  {C.DIM}待核对 {pend} | 已核对 {len(done)} 胜率 {(f'{wr*100:.0f}%' if done else '—')}{C.X}")
        return price_map

    def _emit_coord(self, sym, prob, price, d):
        side = "YES(涨)" if d > 0 else "NO(跌)"
        conf = prob if d > 0 else 1 - prob
        key = f"{sym}_{side}"
        if key in self.dedup and time.time() - self.dedup[key] < DEDUP_SECONDS:
            return
        self.dedup[key] = time.time()
        s = sym.replace("USDT", "")
        print(f"     {C.B_}→ {s} {side} 置信{conf:.2f} 现价{price:.2f}{C.X}")
        print(f"       {C.CY}↳ 入场：{entry_guidance(side, price, now_sh(), HORIZON_MIN.get(TARGET, 60))}{C.X}")
        if self.enable_toast:
            threading.Thread(target=_fire_toast,
                             args=(f"Horizon协同 {s}", f"{side} 置信{conf:.2f}（BTC+ETH同向）"),
                             daemon=True).start()
        if self.mailer:
            subj = f"Horizon协同 {s} {side} 置信{conf:.2f}"
            body = (f"【协同信号】BTC 与 ETH 同向确认\n币种：{s}\n方向：{side}\n置信：{conf:.2f}\n"
                    f"现价：{price:.2f}\n入场指引：{entry_guidance(side, price, now_sh(), HORIZON_MIN.get(TARGET, 60))}\n"
                    f"周期：{TARGET}\n时间：{now_sh():%Y-%m-%d %H:%M:%S}\n\n"
                    f"提示：协同=更少更准；仍仅供参考，平注小仓，不构成投资建议。")
            threading.Thread(target=self.mailer.send, args=(subj, body), daemon=True).start()
        self.history.append({"ts": now_sh().isoformat(), "symbol": sym, "target": TARGET,
                             "side": side, "confidence": round(conf, 4), "prob": round(prob, 4),
                             "price": price, "bet": 1.0, "resolved": False, "won": None})
        self._save_history()

    # —— 战绩板：加平注纸面盈亏 + 凯利建议 ——
    def _scoreboard(self, price_map):
        self._resolve(price_map)
        done = [h for h in self.history if h.get("resolved")]
        print(f"\n{C.B_}{C.CY}╔═══ 协同版战绩（保本 {P_STAR*100:.1f}%）═══╗{C.X}")
        for sym in SYMBOLS:
            d = [h for h in done if h["symbol"] == sym]
            n = len([h for h in self.history if h["symbol"] == sym])
            w = sum(1 for h in d if h["won"])
            if d:
                wr = w / len(d)
                col = C.G if wr > P_STAR else (C.Y if wr > P_STAR - 0.01 else C.R)
                print(f"  {sym.replace('USDT',''):4s} 信号{n:3d} 已核对{len(d):3d} 胜率{col}{wr*100:5.1f}%{C.X}")
            else:
                print(f"  {sym.replace('USDT',''):4s} 信号{n:3d} 已核对  0 {C.DIM}(等到点核对){C.X}")
        if done:
            wins = sum(1 for h in done if h["won"])
            wr = wins / len(done)
            flat = sum(B if h["won"] else -1.0 for h in done)   # 平注纸面盈亏(单位)
            kelly = max(0.0, (wr * (1 + B) - 1) / B)             # 全凯利比例
            verdict = f"{C.G}✅盈利{C.X}" if wr > P_STAR else (f"{C.Y}⚖保本{C.X}" if wr > P_STAR - 0.01 else f"{C.R}❌亏损{C.X}")
            print(f"  {C.B_}合计 已核对{len(done)} 胜率{wr*100:.1f}% → {verdict}{C.X}")
            print(f"  {C.CY}平注纸面盈亏 {flat:+.1f} 单位{C.X}  "
                  f"{C.DIM}凯利建议每笔≈本金 {kelly*100:.1f}%（新手用一半 {kelly*50:.1f}%）{C.X}")
            if len(done) < 30:
                print(f"  {C.DIM}(样本{len(done)}条,<30 仅供参考，凯利比例还不稳){C.X}")
        else:
            print(f"  {C.DIM}暂无已核对信号。协同较严，信号更少，挂着等。{C.X}")
        print(f"{C.B_}{C.CY}╚{'═'*30}╝{C.X}")

    def run_forever(self, probe=0):
        print(C.B_ + "=" * 60 + C.X)
        print(f"{C.B_}  Horizon 优化版 · BTC/ETH 协同 + 平注/凯利{C.X}")
        print(f"  只在 BTC 与 ETH 【同向】时出信号（更少更准） | 门槛 ±{self.threshold}")
        _ways = (["弹窗"] if self.enable_toast else []) + ([f"邮件→{self.mailer.to}"] if self.mailer else [])
        print(f"  {C.DIM}提醒：{' + '.join(_ways) or '仅窗口'} | 保本 {P_STAR*100:.1f}%(盈亏比{B}){C.X}")
        print(f"  {C.Y}⚠ 纸面观察·不下单·平注小仓；关闭窗口即停止{C.X}")
        print(C.B_ + "=" * 60 + C.X)
        self.load_history()
        if self.history:
            self._scoreboard({})
        rounds = 0
        while self.running:
            t0 = time.time()
            pm = self._scan()
            self._resolve(pm)
            if time.time() - self._last_board >= self.scoreboard_every:
                self._scoreboard(pm); self._last_board = time.time()
            rounds += 1
            if probe and rounds >= probe:
                self._scoreboard(pm); break
            next_wake = (int(time.time()) // SCAN_INTERVAL + 1) * SCAN_INTERVAL + 15
            while self.running and time.time() < next_wake:
                time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-toast", action="store_true")
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.62)
    ap.add_argument("--board-min", type=float, default=30)
    ap.add_argument("--probe", type=int, default=0)
    args = ap.parse_args()

    eng = OptimisedSignal(threshold=args.threshold, enable_toast=not args.no_toast,
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

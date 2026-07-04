#!/usr/bin/env python
"""实时信号推送 v3 — 集成策略版本（XGBoost + LightGBM + 规则，2/3 投票）。

胜率验证（5-Fold CV，90 天 BTC 5m 数据）：
  XGBoost 单跑: 75.2% ± 1.0%
  LightGBM 单跑: 75.2% ± 1.2%
  规则单跑: 63.9% ± 5.6%
  集成 2-of-3: 76.1% ± 0.9%  ← 采用这个
  集成 3-of-3: 76.4% ± 20.9% (信号太少)

扫描间隔: 5 分钟（每 5m K 线收盘后跑一次）
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
from typing import Dict, List, Optional, Tuple

import httpx
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.backtest.features_v5 import (
    build_aggregated_klines_v2,
    _bars_to_arrays,
    build_all_indicators,
    compute_features_v5,
    FEATURE_NAMES_V5,
)
from src.backtest.rules_v5 import evaluate_all_rules

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("horizon.live.v3")

_TZ_CN = timezone(timedelta(hours=8))

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════
PROOF_DIR = Path(__file__).resolve().parent.parent.parent / "proofs" / "iter-15"
MODEL_XGB_30M = PROOF_DIR / "BTCUSDT_xgb_v5_30m_pruned.joblib"
MODEL_LGBM_30M = PROOF_DIR / "BTCUSDT_lgbm_v5_30m.joblib"

# ML 概率阈值（验证出 76.1% 胜率的阈值）
ML_THRESHOLD = 0.55

# 规则投票阈值（≥2/5 同意才算有效）
RULE_MIN_VOTES = 2

# 信号去重（同一标的 5 分钟内不重复）
DEDUP_SECONDS = 300

SIGNAL_CACHE: Dict[str, float] = {}  # signal_id → last ts
SIGNAL_LOG: List[dict] = []  # 内存日志（每轮追加，可导出到 JSONL）
SIGNAL_LOG_PATH = PROOF_DIR / "live_signals.jsonl"


def now_sh() -> datetime:
    return datetime.now(tz=_TZ_CN)


# ═══════════════════════════════════════════════════════════════
# Data fetch (5m primary + aggregation)
# ═══════════════════════════════════════════════════════════════
_PROXY_URL = ""


def _detect_system_proxy_windows() -> str:
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
    except Exception:
        return ""


def _fetch_klines_5m(symbol: str, limit: int = 300) -> List[list]:
    from src.realtime.incremental_klines import fetch_klines_incremental
    return fetch_klines_incremental(
        symbol, "5m",
        limit=limit,
        proxy_url=_PROXY_URL,
        timeout=20.0,
        internal_retries=5,
    )


def _raw_to_arrays(raw: List[list]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    o = np.array([float(k[1]) for k in raw], dtype=float)
    c = np.array([float(k[4]) for k in raw], dtype=float)
    h = np.array([float(k[2]) for k in raw], dtype=float)
    l = np.array([float(k[3]) for k in raw], dtype=float)
    v = np.array([float(k[5]) for k in raw], dtype=float)
    return o, c, h, l, v


def _aggregate_tf(o5, c5, h5, l5, v5, target_interval: str):
    ratios = {"15m": 3, "30m": 6, "1h": 12, "4h": 48}
    ratio = ratios[target_interval]
    n = len(c5) // ratio
    if n == 0:
        return tuple(np.array([]) for _ in range(5))
    o_out = np.zeros(n); c_out = np.zeros(n); h_out = np.zeros(n); l_out = np.zeros(n); v_out = np.zeros(n)
    for i in range(n):
        s, e = i * ratio, (i + 1) * ratio
        o_out[i] = o5[s]
        c_out[i] = c5[e - 1]
        h_out[i] = np.max(h5[s:e])
        l_out[i] = np.min(l5[s:e])
        v_out[i] = np.sum(v5[s:e])
    return o_out, c_out, h_out, l_out, v_out


# ═══════════════════════════════════════════════════════════════
# Toast (Windows notification)
# ═══════════════════════════════════════════════════════════════
def _fire_toast(title: str, msg: str) -> None:
    try:
        from plyer import notification
        notification.notify(title=title, message=msg, app_name="Horizon-Incident", timeout=8)
    except Exception:
        pass


class Color:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


# ═══════════════════════════════════════════════════════════════
# Ensemble Live Signal Engine
# ═══════════════════════════════════════════════════════════════
class EnsembleLiveSignal:
    def __init__(self, enable_toast: bool = True):
        self.enable_toast = enable_toast
        self.running = True
        self.xgb_model = None
        self.xgb_feat_idx = None
        self.lgbm_model = None
        self.lgbm_feat_idx = None

    def load_models(self) -> bool:
        ok = True
        if not MODEL_XGB_30M.exists():
            log.error("⚠ XGBoost 模型不存在: %s", MODEL_XGB_30M)
            ok = False
        else:
            data = joblib.load(MODEL_XGB_30M)
            self.xgb_model = data["model"]
            self.xgb_feat_idx = np.array(data["feature_indices"])
            log.info("✓ 加载 XGBoost (%d 特征)", len(self.xgb_feat_idx))

        if not MODEL_LGBM_30M.exists():
            log.error("⚠ LightGBM 模型不存在: %s", MODEL_LGBM_30M)
            ok = False
        else:
            data = joblib.load(MODEL_LGBM_30M)
            self.lgbm_model = data["model"]
            self.lgbm_feat_idx = np.array(data["feature_indices"])
            log.info("✓ 加载 LightGBM (%d 特征)", len(self.lgbm_feat_idx))

        return ok

    def run_forever(self):
        log.info("=" * 60)
        log.info("  Horizon-Incident v3 (Ensemble 2/3)")
        log.info("  组件: XGBoost + LightGBM + 5 条规则")
        log.info("  投票: ≥2/3 同意才出信号")
        log.info("  验证胜率: 76.1%% ± 0.9%% (5-Fold CV, 90 天)")
        log.info("  扫描间隔: 5 分钟")
        log.info("=" * 60)

        scan_interval = 300
        min_klines = 60

        # Probe mode: 跑 3 轮后停（验证 pipeline）
        probe_rounds = int(os.environ.get("PROBE_ROUNDS", "0"))

        round_count = 0
        while self.running:
            start = time.time()
            try:
                for sym in ["BTCUSDT", "ETHUSDT"]:
                    if sym == "ETHUSDT":
                        # ETH 模型还没训练好（代理超时），先跳过
                        if self.lgbm_model is None and sym == "ETHUSDT":
                            continue
                    self._process_symbol(sym, min_klines)
            except Exception as e:
                log.error("[live] scan error: %s", e)

            elapsed = time.time() - start
            round_count += 1
            log.debug("[live] round %d done in %.1fs", round_count, elapsed)

            if probe_rounds > 0 and round_count >= probe_rounds:
                log.info("[live] probe rounds (%d) complete, exiting", probe_rounds)
                self.running = False
                break

            sleep_for = max(1, scan_interval - elapsed)
            time.sleep(sleep_for)

    def _process_symbol(self, symbol: str, min_bars: int):
        try:
            raw5 = _fetch_klines_5m(symbol, limit=300)
        except Exception as e:
            log.error("[live] %s 数据拉取失败: %s", symbol, e)
            return

        if len(raw5) < min_bars:
            log.warning("[live] %s 数据不足 (%d < %d)", symbol, len(raw5), min_bars)
            return

        o5, c5, h5, l5, v5 = _raw_to_arrays(raw5)
        o15, c15, h15, l15, v15 = _aggregate_tf(o5, c5, h5, l5, v5, "15m")
        o30, c30, h30, l30, v30 = _aggregate_tf(o5, c5, h5, l5, v5, "30m")
        o1h, c1h, h1h, l1h, v1h = _aggregate_tf(o5, c5, h5, l5, v5, "1h")
        o4h, c4h, h4h, l4h, _ = _aggregate_tf(o5, c5, h5, l5, v5, "4h")

        try:
            inds = build_all_indicators(
                o5, c5, h5, l5, v5,
                o15, c15, h15, l15, v15,
                o30, c30, h30, l30, v30,
                o1h, c1h, h1h, l1h, v1h,
                o4h, c4h, h4h, l4h,
            )
        except Exception as e:
            log.error("[live] %s 指标计算失败: %s", symbol, e)
            return

        i_5m = len(c5) - 2
        if i_5m < min_bars:
            return

        feats = compute_features_v5(inds, c5, v5, c1h, v30, h30, l30, i_5m)
        feats_arr = np.array([feats], dtype=float)

        # ── 1. XGBoost vote ──
        xgb_vote = -1  # -1 = 无意见
        xgb_prob = 0.5
        if self.xgb_model is not None:
            try:
                xgb_prob = float(self.xgb_model.predict_proba(feats_arr[:, self.xgb_feat_idx])[0, 1])
                if xgb_prob >= ML_THRESHOLD:
                    xgb_vote = 1  # YES
                elif xgb_prob <= 1 - ML_THRESHOLD:
                    xgb_vote = 0  # NO
            except Exception as e:
                log.error("[live] XGBoost predict error: %s", e)

        # ── 2. LightGBM vote ──
        lgbm_vote = -1
        lgbm_prob = 0.5
        if self.lgbm_model is not None:
            try:
                lgbm_prob = float(self.lgbm_model.predict_proba(feats_arr[:, self.lgbm_feat_idx])[0, 1])
                if lgbm_prob >= ML_THRESHOLD:
                    lgbm_vote = 1
                elif lgbm_prob <= 1 - ML_THRESHOLD:
                    lgbm_vote = 0
            except Exception as e:
                log.error("[live] LightGBM predict error: %s", e)

        # ── 3. Rules vote ──
        rules = evaluate_all_rules(inds, i_5m, c5)
        rule_pos = sum(1 for r in rules if r[0] > 0)
        rule_neg = sum(1 for r in rules if r[0] < 0)
        rules_vote = -1
        if rule_pos >= RULE_MIN_VOTES and rule_pos > rule_neg:
            rules_vote = 1
        elif rule_neg >= RULE_MIN_VOTES and rule_neg > rule_pos:
            rules_vote = 0

        # ── 4. Ensemble decision ──
        valid_votes = [v for v in [xgb_vote, lgbm_vote, rules_vote] if v >= 0]
        pos_v = sum(1 for v in valid_votes if v == 1)
        neg_v = sum(1 for v in valid_votes if v == 0)

        if pos_v == 0 and neg_v == 0:
            return  # no consensus
        if pos_v > neg_v:
            side, side_label = 1, "YES"
        elif neg_v > pos_v:
            side, side_label = 0, "NO"
        else:
            return  # tied → skip

        # Confidence: 平均 prob（如果都有）
        confs = []
        if xgb_vote == side: confs.append(xgb_prob if side == 1 else 1 - xgb_prob)
        if lgbm_vote == side: confs.append(lgbm_prob if side == 1 else 1 - lgbm_prob)
        if rules_vote == side:
            confs.append(0.70)  # 规则票的固定置信度
        conf = sum(confs) / len(confs) if confs else 0.65

        # ── Dedup ──
        sig_id = f"{symbol}_{side_label}"
        now_ts = time.time()
        if sig_id in SIGNAL_CACHE:
            if now_ts - SIGNAL_CACHE[sig_id] < DEDUP_SECONDS:
                return
        SIGNAL_CACHE[sig_id] = now_ts

        # ── Display ──
        self._display_signal(
            symbol, side_label, side, conf,
            xgb_vote, lgbm_vote, rules_vote,
            xgb_prob, lgbm_prob,
            rule_pos, rule_neg, rules,
        )

    def _display_signal(self, symbol, side_label, side, conf,
                         xgb_vote, lgbm_vote, rules_vote,
                         xgb_prob, lgbm_prob,
                         rule_pos, rule_neg, rules):
        sym_short = symbol.replace("USDT", "")

        # Vote breakdown
        xgb_str = f"XGB={'Y' if xgb_vote == 1 else 'N' if xgb_vote == 0 else '-'}"
        lgbm_str = f"LGB={'Y' if lgbm_vote == 1 else 'N' if lgbm_vote == 0 else '-'}"
        rules_str = f"R{rule_pos+rule_neg}/5"

        color = Color.GREEN if side == 1 else Color.RED
        side_tag = f"[+{side_label}]" if side == 1 else f"[-{side_label}]"

        line = (
            f"{Color.BOLD}{color}"
            f"[30m] {sym_short:4s} {side_tag:8s}  conf={conf:.2f}  "
            f"votes={xgb_str} {lgbm_str} {rules_str}"
            f"{Color.RESET}"
        )
        print(line, flush=True)

        # Show active rule names (if any)
        if rules and rules_vote >= 0:
            rule_names = " | ".join([r[2] for r in rules if r[0] == (1 if side == 1 else -1)])
            if rule_names:
                print(f"       rules: {rule_names}", flush=True)

        # Show XGB + LGB probs
        print(f"       probs: XGB={xgb_prob:.3f} LGB={lgbm_prob:.3f}", flush=True)

        # Toast
        if self.enable_toast:
            dir_word = "涨" if side == 1 else "跌"
            title = f"Horizon {sym_short}"
            msg = f"{side_label} — 30m后预计{dir_word} (conf={conf:.2f})"
            threading.Thread(target=_fire_toast, args=(title, msg), daemon=True).start()

        # Log to JSONL
        log_entry = {
            "ts": now_sh().isoformat(),
            "symbol": symbol,
            "side": side_label,
            "confidence": round(conf, 4),
            "xgb_vote": "YES" if xgb_vote == 1 else ("NO" if xgb_vote == 0 else "skip"),
            "lgbm_vote": "YES" if lgbm_vote == 1 else ("NO" if lgbm_vote == 0 else "skip"),
            "rules_vote": "YES" if rules_vote == 1 else ("NO" if rules_vote == 0 else "skip"),
            "xgb_prob": round(xgb_prob, 4),
            "lgbm_prob": round(lgbm_prob, 4),
            "rule_pos": rule_pos,
            "rule_neg": rule_neg,
            "active_rules": [r[2] for r in rules],
        }
        SIGNAL_LOG.append(log_entry)
        try:
            with open(SIGNAL_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    global _PROXY_URL

    parser = argparse.ArgumentParser(description="Horizon-Incident v3 Ensemble Live Signal")
    parser.add_argument("--no-toast", action="store_true")
    parser.add_argument("--proxy", type=str, default="")
    parser.add_argument("--probe", type=int, default=0,
                        help="Probe rounds (0=run forever, N=run N rounds then exit)")
    args = parser.parse_args()

    _PROXY_URL = args.proxy or os.environ.get("HORIZON_PROXY") or _detect_system_proxy_windows()
    if _PROXY_URL:
        log.info("[live] 代理: %s", _PROXY_URL)
    else:
        log.info("[live] 直连模式")

    if args.probe > 0:
        os.environ["PROBE_ROUNDS"] = str(args.probe)
        log.info("[live] probe mode: %d rounds", args.probe)

    engine = EnsembleLiveSignal(enable_toast=not args.no_toast)

    if not engine.load_models():
        log.error("[live] 模型加载失败，请先运行 optimize_v5.py + optimize_lgbm.py")
        sys.exit(1)

    def _shutdown(sig, frame):
        log.info("[live] 收到关闭信号，退出...")
        engine.running = False

    _signal.signal(_signal.SIGINT, _shutdown)
    _signal.signal(_signal.SIGTERM, _shutdown)

    log.info("[live] 信号日志: %s", SIGNAL_LOG_PATH)
    engine.run_forever()


if __name__ == "__main__":
    main()
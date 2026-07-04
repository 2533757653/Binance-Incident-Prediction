"""因果日线特征引擎（含恐慌贪婪指数衍生因子 + 丰富量能因子）。

设计原则（吸取 iter-14/15 数据泄露教训）：
- **无前视**：第 i 根日线的特征只用索引 <= i 的信息（pandas 滚动窗口天然因果）。
- 目标 y = 第 i+1 根收盘 > 第 i 根收盘（次日涨/跌，二元）。
- F&G 按 ts<=bar 开盘时刻对齐（见 fear_greed.align_fng），预测的是下一根 → 天然留出缓冲。

列名前缀用于消融实验：
  px_*  价格 / 趋势 / 波动 / 高阶统计
  vol_* 量能（OBV / 资金流 / 量价相关）
  fng_* 恐慌贪婪指数族（本轮新增，**价格无关**）
  tm_*  时间 / 季节

思路借鉴自用户旧因子引擎草稿（F&G 衍生因子、量能因子、市场状态），
但全部重写为因果安全版本。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.fear_greed import align_fng, fetch_fng


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1 / n, adjust=False).mean()
    al = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = ag / (al + 1e-12)
    return 100 - 100 / (1 + rs)


def _streak(flag: pd.Series) -> pd.Series:
    """连续为真的计数（因果）：...0,1,2,3,0,1..."""
    f = flag.astype(int)
    grp = (f == 0).cumsum()
    return f.groupby(grp).cumsum()


# F&G 族列名（用于消融时识别）
FNG_PREFIX = "fng_"


def compute_daily_matrix(
    klines: List, use_fng: bool = True, warmup: int = 200
) -> Tuple[pd.DataFrame, np.ndarray, List]:
    """构建日线特征矩阵。

    Returns: (df_features, y, dates)
      - df_features: 每行一根日线的全部特征（已切掉 warmup 与最后一根无目标行）
      - y: 次日涨(1)/跌(0)
      - dates: 每行对应的 open_time
    """
    o = pd.Series([k.open for k in klines], dtype=float)
    h = pd.Series([k.high for k in klines], dtype=float)
    l = pd.Series([k.low for k in klines], dtype=float)
    c = pd.Series([k.close for k in klines], dtype=float)
    v = pd.Series([k.volume for k in klines], dtype=float)
    times = [k.open_time for k in klines]

    df = pd.DataFrame(index=range(len(klines)))
    ret1 = c.pct_change()

    # ════════ px_* 价格 / 趋势 ════════
    df["px_ret1"] = ret1
    df["px_ret3"] = c.pct_change(3)
    df["px_ret5"] = c.pct_change(5)
    df["px_ret10"] = c.pct_change(10)
    for p in (5, 10, 20, 50, 100, 200):
        df[f"px_maratio_{p}"] = c / c.rolling(p).mean() - 1
    df["px_ma5_20"] = c.rolling(5).mean() / c.rolling(20).mean() - 1
    df["px_ma20_50"] = c.rolling(20).mean() / c.rolling(50).mean() - 1
    df["px_ma50_200"] = c.rolling(50).mean() / c.rolling(200).mean() - 1
    df["px_rsi14"] = _rsi(c, 14) / 100.0
    df["px_rsi7"] = _rsi(c, 7) / 100.0
    macd = _ema(c, 12) - _ema(c, 26)
    df["px_macd_hist"] = (macd - _ema(macd, 9)) / c
    df["px_mom20"] = c / c.shift(20) - 1
    # 布林
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    df["px_bb_pos"] = (c - (m20 - 2 * s20)) / (4 * s20 + 1e-12)
    df["px_bb_width"] = 4 * s20 / (m20 + 1e-12)
    # ATR%
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["px_atr14"] = tr.rolling(14).mean() / c
    # 位置 / 回撤
    df["px_pos20"] = (c - l.rolling(20).min()) / (h.rolling(20).max() - l.rolling(20).min() + 1e-12)
    df["px_pos50"] = (c - l.rolling(50).min()) / (h.rolling(50).max() - l.rolling(50).min() + 1e-12)
    df["px_dd60"] = c / c.rolling(60).max() - 1
    # 波动 / 高阶统计
    df["px_hv20"] = ret1.rolling(20).std()
    df["px_volcone"] = ret1.rolling(5).std() / (ret1.rolling(20).std() + 1e-12)
    df["px_skew20"] = ret1.rolling(20).skew()
    df["px_kurt20"] = ret1.rolling(20).kurt()
    df["px_autocorr20"] = ret1.rolling(20).apply(lambda x: pd.Series(x).autocorr(1), raw=False)

    # ════════ vol_* 量能 ════════
    vma20 = v.rolling(20).mean()
    df["vol_ratio20"] = v / (vma20 + 1e-12)
    df["vol_z20"] = (v - vma20) / (v.rolling(20).std() + 1e-12)
    obv = (np.sign(ret1.fillna(0)) * v).cumsum()
    df["vol_obv_slope"] = obv.diff(5) / (vma20 * 5 + 1e-12)
    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    mfv = mfm * v
    df["vol_cmf20"] = mfv.rolling(20).sum() / (v.rolling(20).sum() + 1e-12)
    df["vol_pv_corr20"] = c.rolling(20).corr(v)
    df["vol_pv_reson"] = c.pct_change(5) * v.pct_change(5)
    df["vol_trend5"] = v.pct_change(5)

    # ════════ fng_* 恐慌贪婪指数族（价格无关，本轮新增）════════
    if use_fng:
        fng = pd.Series(align_fng(times, fetch_fng()), dtype=float)
        ma5, ma20f, sd20f = fng.rolling(5).mean(), fng.rolling(20).mean(), fng.rolling(20).std()
        df["fng_level"] = fng / 100.0
        df["fng_ma5"] = ma5 / 100.0
        df["fng_ma20"] = ma20f / 100.0
        df["fng_dev"] = (fng - ma20f) / 100.0
        df["fng_chg1"] = fng.diff(1) / 100.0
        df["fng_chg3"] = fng.diff(3) / 100.0
        df["fng_chg7"] = fng.diff(7) / 100.0
        df["fng_vol20"] = sd20f / 100.0
        df["fng_z"] = (fng - ma20f) / (sd20f + 1e-9)
        df["fng_extreme_fear"] = (fng <= 25).astype(float)
        df["fng_extreme_greed"] = (fng >= 75).astype(float)
        df["fng_fear_streak"] = _streak(fng <= 25) / 10.0
        df["fng_greed_streak"] = _streak(fng >= 75) / 10.0
        df["fng_pct90"] = fng.rolling(90).apply(
            lambda a: float(np.mean(a[:-1] < a[-1])) if len(a) > 1 else 0.5, raw=True)
        # 从极端恐慌反弹（昨日<=25 且今日回升）——经典反向做多信号
        df["fng_rev_up"] = ((fng.shift(1) <= 25) & (fng.diff() > 0)).astype(float)
        # F&G 与价格 5 日方向背离
        df["fng_vs_price"] = np.sign(fng.diff(5).fillna(0)) * np.sign(c.pct_change(5).fillna(0))
        # 综合情绪（RSI + F&G，均去中心化）
        df["fng_composite"] = ((df["px_rsi14"] - 0.5) + (fng / 100.0 - 0.5)) / 2.0

    # ════════ tm_* 时间 / 季节 ════════
    dow = pd.Series([t.weekday() for t in times], dtype=float)
    mon = pd.Series([t.month for t in times], dtype=float)
    df["tm_dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["tm_dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["tm_mon_sin"] = np.sin(2 * np.pi * mon / 12)
    df["tm_mon_cos"] = np.cos(2 * np.pi * mon / 12)

    # ════════ 目标 + 清洗 ════════
    y_full = (c.shift(-1) > c).astype(int)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    n = len(klines)
    valid = list(range(warmup, n - 1))  # 掐掉 warmup + 最后一根（无次日）
    X = df.iloc[valid].reset_index(drop=True)
    y = y_full.iloc[valid].to_numpy(dtype=int)
    dates = [times[i] for i in valid]
    return X, y, dates


def feature_groups(cols: List[str]) -> dict:
    """按前缀归组，便于消融/可视化。"""
    g = {"px": [], "vol": [], "fng": [], "tm": []}
    for cidx in cols:
        pre = cidx.split("_", 1)[0]
        g.get(pre, g.setdefault(pre, [])).append(cidx)
    return g

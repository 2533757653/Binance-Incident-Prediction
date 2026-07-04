"""
Builder-F · 可信度评分（Platt scaling）。

目标：把 SignalGenerator 输出的 raw confidence 用历史"预测置信度 vs 实际胜率"做
      Platt scaling 校准，使 confidence 数值更接近真实胜率。

算法：用 sklearn.linear_model.LogisticRegression 拟合 sigmoid(a*raw + b)。

注意：iter-1 的 SignalGenerator 已经把 confidence = agreement × spread_penalty × time_penalty。
      这一置信度通常**高估**真实胜率，需要校准。
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# 数据容器
# ============================================================
@dataclass
class CalibrationSample:
    """一条历史样本：(raw_confidence, 实际胜负 0/1)"""

    raw_confidence: float
    outcome: int   # 1 = 信号方向正确；0 = 信号方向错误


# ============================================================
# 主类
# ============================================================
class ConfidenceCalibrator:
    """
    Platt scaling 校准器。

    校准公式：p_calibrated = 1 / (1 + exp(a * raw + b))
    训练：用 sklearn LogisticRegression 拟合 sigmoid 曲线。
    """

    def __init__(self, model_path: str = "data/cache/calibration.json"):
        self.model_path = Path(model_path)
        self.a: float = 1.0     # 默认恒等映射
        self.b: float = 0.0
        self._fitted: bool = False
        self._n_train: int = 0

    # --------------------------------------------------------
    # 训练
    # --------------------------------------------------------
    def fit(
        self,
        signals_history: Sequence,           # 任意带 .confidence 属性的对象
        outcomes: Sequence[int],             # 0/1
    ) -> "ConfidenceCalibrator":
        """
        从历史 (raw_confidence, actual_outcome) 拟合 Platt scaling。

        要求：
            - len(signals_history) == len(outcomes) >= 100（iter-2 硬要求）
            - outcomes 必须是 0/1 整数
        """
        n = len(signals_history)
        if n != len(outcomes):
            raise ValueError(f"signals_history and outcomes length mismatch: {n} vs {len(outcomes)}")
        if n < 100:
            raise ValueError(f"need at least 100 samples for Platt scaling, got {n}")

        # 1) 抽 raw confidence
        raws: List[float] = []
        labels: List[int] = []
        for sig, y in zip(signals_history, outcomes):
            try:
                raw = float(sig.confidence)
            except AttributeError:
                raise AttributeError(f"signal object has no .confidence: {type(sig).__name__}")
            raws.append(raw)
            labels.append(int(y))

        # 2) 用 sklearn LogisticRegression 拟合（logit space 线性）
        from sklearn.linear_model import LogisticRegression
        import numpy as np

        X = np.array(raws, dtype=float).reshape(-1, 1)
        y = np.array(labels, dtype=int)

        # 强制 L2 正则 + 大 C（Platt 标准用法，正则弱一点效果更接近经典 Platt）
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        clf.fit(X, y)

        # 3) 提取 Platt 参数：p = 1/(1+exp(a*x+b)) → a = coef_[0], b = intercept_[0]
        self.a = float(clf.coef_[0][0])
        self.b = float(clf.intercept_[0])
        self._fitted = True
        self._n_train = n

        logger.info(
            "calibration fit done: n=%d a=%.4f b=%.4f (pos_rate=%.3f)",
            n, self.a, self.b, float(np.mean(y)),
        )
        return self

    # --------------------------------------------------------
    # 推理
    # --------------------------------------------------------
    def calibrate(self, raw_confidence: float) -> float:
        """
        把 raw_confidence 映射到校准后的 [0, 1] 置信度。

        默认（未训练）：恒等映射。
        训练后：1 / (1 + exp(a * raw + b))。
        """
        try:
            z = self.a * float(raw_confidence) + self.b
        except Exception:
            return float(raw_confidence)
        # 防溢出
        if z > 50:
            return 0.0
        if z < -50:
            return 1.0
        p = 1.0 / (1.0 + math.exp(z))
        return float(max(0.0, min(1.0, p)))

    # --------------------------------------------------------
    # 工具：批量校准
    # --------------------------------------------------------
    def calibrate_batch(self, raws: Sequence[float]) -> List[float]:
        return [self.calibrate(float(r)) for r in raws]

    # --------------------------------------------------------
    # 评估：Brier score（越小越好）
    # --------------------------------------------------------
    @staticmethod
    def brier_score(preds: Sequence[float], outcomes: Sequence[int]) -> float:
        """Brier = mean((p - y)^2)"""
        if len(preds) != len(outcomes):
            raise ValueError("preds/outcomes length mismatch")
        if not preds:
            return 0.0
        s = 0.0
        for p, y in zip(preds, outcomes):
            s += (float(p) - float(y)) ** 2
        return s / len(preds)

    # --------------------------------------------------------
    # 持久化
    # --------------------------------------------------------
    def save(self) -> Path:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "a": self.a,
            "b": self.b,
            "n_train": self._n_train,
            "fitted": self._fitted,
        }
        self.model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("calibration saved to %s", self.model_path)
        return self.model_path

    def load(self) -> "ConfidenceCalibrator":
        if not self.model_path.exists():
            logger.warning("calibration file not found: %s (using identity)", self.model_path)
            return self
        try:
            payload = json.loads(self.model_path.read_text(encoding="utf-8"))
            self.a = float(payload["a"])
            self.b = float(payload["b"])
            self._n_train = int(payload.get("n_train", 0))
            self._fitted = bool(payload.get("fitted", True))
        except Exception as e:
            logger.warning("calibration load failed (%s); keeping defaults", e)
        return self


# ============================================================
# 便捷函数：构造样本
# ============================================================
def samples_from_trades(trades: Iterable) -> List[CalibrationSample]:
    """
    从 BacktestRun.records (TradeRecord) 构造 CalibrationSample 列表。
    """
    out: List[CalibrationSample] = []
    for tr in trades:
        try:
            out.append(CalibrationSample(
                raw_confidence=float(tr.signal.confidence),
                outcome=1 if tr.won else 0,
            ))
        except AttributeError:
            continue
    return out

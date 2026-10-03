"""攻击者集合 + 特征构建（纯 numpy，离线可跑）。

特征矩阵列布局（固定契约，K = n_classes）：
  0 loss | 1 confidence | 2 entropy | 3 margin | 4 correct | 5..5+K-1 各类概率 p_c

完整概率向量是成员推理在多类设定下的关键特征：成员信息分布于各类置信度中，
单标量 loss 抓不全 —— 这也是 Shokri 等真实 MIA 在 CIFAR-10/100 上能显著超越
loss 阈值基线的原因。

攻击者：
  - loss_threshold (基线 / Yeom blind)：仅用 loss，score = -loss
  - shadow_all    (旗舰 Shokri 影子模型攻击)：多影子模型积累「特征→成员」
                  监督，迁移到目标；逻辑回归融合全部特征（含完整概率向量）
  - shadow_loss_only  (消融①)：影子攻击仅用 loss
  - shadow_no_correct (消融②)：去除 correct 特征

作者：晨星
"""

from __future__ import annotations

import numpy as np

from core.errors import AttackError


# ——————————————————— 特征构建 ———————————————————
def build_features(target, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(target.predict_proba(X), 1e-12, 1.0 - 1e-12)
    n = y.shape[0]
    loss = -np.log(p[np.arange(n), y.astype(int)])
    confidence = p.max(axis=1)
    entropy = -(p * np.log(p)).sum(axis=1)
    sp = np.sort(p, axis=1)
    margin = sp[:, -1] - sp[:, -2]
    correct = (p.argmax(axis=1) == y.astype(int)).astype(np.float64)
    return np.column_stack([loss, confidence, entropy, margin, correct, p]).astype(np.float64)


# ——————————————————— 纯 numpy 逻辑回归 ———————————————————
class NumpyLogistic:
    """确定性二分类逻辑回归（全批量梯度下降；不做内部标准化，由调用方统一缩放）。"""

    def __init__(self, lr: float = 0.05, epochs: int = 300, seed: int = 0):
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.w: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpyLogistic":
        Xb = np.hstack([np.asarray(X, dtype=np.float64), np.ones((X.shape[0], 1))])
        y = np.asarray(y, dtype=np.float64).ravel()
        w = np.zeros(Xb.shape[1])
        m = Xb.shape[0]
        for _ in range(self.epochs):
            z = Xb @ w
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            grad = (Xb.T @ (p - y)) / m
            w = w - self.lr * grad
        self.w = w
        return self

    def decision(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise AttackError("逻辑回归未训练")
        Xb = np.hstack([np.asarray(X, dtype=np.float64), np.ones((X.shape[0], 1))])
        return Xb @ self.w


# ——————————————————— 攻击者 ———————————————————
class LossThresholdAttack:
    """基线（Yeom 2018 blind）：仅用 loss，score = -loss。"""

    name = "loss_threshold"

    def fit(self, feats: np.ndarray, labels: np.ndarray) -> None:
        if feats.shape[1] < 1:
            raise AttackError("loss_threshold 需要 loss 特征")

    def score(self, feats: np.ndarray) -> np.ndarray:
        return -feats[:, 0]


class ShadowAttack:
    """Shokri 影子模型攻击（旗舰 / 消融变体）。

    fit 接收「影子模型积累的 (特征, 成员标签)」数据；score 接收目标样本特征。
    用纯 numpy 逻辑回归作为攻击分类器。

    域偏移处理：标准化统计由「影子 + 目标特征合并分布」统一估计（仅特征缩放，
    不含成员标签，无泄漏），影子与目标特征在同一尺度空间，避免分布错位导致攻击劣于基线。
    """

    def __init__(self, name: str, cols: list[int], seed: int = 0, lr: float = 0.05, epochs: int = 300):
        self.name = name
        self.cols = list(cols)
        self.seed = seed
        self.lr = lr
        self.epochs = epochs
        self.clf: NumpyLogistic | None = None
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        if self._mu is None or self._sd is None:
            raise AttackError("未设置标准化统计量")
        return (np.asarray(X, dtype=np.float64) - self._mu[self.cols]) / self._sd[self.cols]

    def set_standardize(self, ref: np.ndarray) -> None:
        r = np.asarray(ref, dtype=np.float64)
        self._mu = r.mean(axis=0)
        self._sd = r.std(axis=0) + 1e-8

    def fit(self, feats: np.ndarray, labels: np.ndarray) -> None:
        if self._mu is None:
            self.set_standardize(feats)
        Xz = self._standardize(feats[:, self.cols])
        self.clf = NumpyLogistic(lr=self.lr, epochs=self.epochs, seed=self.seed)
        self.clf.fit(Xz, labels)

    def score(self, feats: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise AttackError("影子攻击未训练")
        Xz = self._standardize(feats[:, self.cols])
        return self.clf.decision(Xz)


# 列布局常量
COL_LOSS, COL_CONF, COL_ENT, COL_MARGIN, COL_CORRECT = range(5)


def make_attacks(seed: int = 0, n_classes: int = 10) -> list:
    """构造旗舰 + 基线 + 消融攻击集合。"""
    prob_cols = list(range(5, 5 + n_classes))
    full = [COL_LOSS, COL_CONF, COL_ENT, COL_MARGIN, COL_CORRECT] + prob_cols
    return [
        LossThresholdAttack(),
        ShadowAttack("shadow_all", full, seed=seed + 1),
        ShadowAttack("shadow_loss_only", [COL_LOSS], seed=seed + 2),
        ShadowAttack("shadow_no_correct", [COL_LOSS, COL_CONF, COL_ENT, COL_MARGIN] + prob_cols, seed=seed + 3),
    ]

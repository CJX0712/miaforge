"""合成成员推理数据集生成器（固定 seed，可复现）。

数据生成过程（DGP）：
  类条件高斯，类均值沿一条随机方向等距排布，间距 = snr（以噪声 std=1 为单位）。
  snr 越大类越可分（攻击越难，因模型更易泛化）；snr 越小越重叠（攻击越易但模型也难学）。
  成员与非成员来自同一总体（i.i.d.），符合成员推理的标准设定。

作者：晨星
"""

from __future__ import annotations

import numpy as np

from core.errors import DataError


class SyntheticMiaGenerator:
    """沿单方向的类条件高斯 DGP。"""

    def __init__(
        self,
        n_features: int = 20,
        n_classes: int = 2,
        snr: float = 1.5,
        class_spread: float = 1.8,
        seed: int = 0,
    ):
        if n_features < 1:
            raise DataError("n_features >= 1")
        if n_classes < 2:
            raise DataError("n_classes >= 2")
        if snr <= 0:
            raise DataError("snr > 0")
        if class_spread < 0:
            raise DataError("class_spread >= 0")
        self.n_features = n_features
        self.n_classes = n_classes
        self.snr = snr
        self.class_spread = class_spread
        self.seed = seed

    def _class_means(self, rng: np.random.Generator) -> np.ndarray:
        direction = rng.standard_normal(self.n_features)
        direction /= np.linalg.norm(direction)
        centers = []
        for c in range(self.n_classes):
            offset = (c - (self.n_classes - 1) / 2.0) * self.snr
            centers.append(offset * direction)
        return np.asarray(centers, dtype=np.float64)

    def _class_noise_scales(self, rng: np.random.Generator) -> np.ndarray:
        """每类固有难度不同（后类更噪声）→ 制造「类难度混淆」，
        使原始全局 loss 阈值次优，而用完整概率向量（类条件校准）的攻击
        可学出逐类边界，成为真实增益来源。确定性（由 seed 决定）。
        class_spread 控制跨度（0 = 等难度）。"""
        if self.n_classes <= 1:
            return np.ones(1, dtype=np.float64)
        frac = np.linspace(0.0, 1.0, self.n_classes)
        return (1.0 + self.class_spread * frac).astype(np.float64)

    def generate(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        if n < 1:
            raise DataError("n >= 1")
        rng = np.random.default_rng(self.seed)
        means = self._class_means(rng)
        scales = self._class_noise_scales(rng)
        y = rng.integers(0, self.n_classes, size=n)
        noise = rng.standard_normal((n, self.n_features)) * scales[y][:, None]
        X = means[y] + noise
        return X.astype(np.float64), y.astype(np.int64)

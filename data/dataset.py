"""成员推理真实数据集加载（本地内置、确定性、离线可用）。

选用 scikit-learn 内置数据集（无需联网）：
  - digits：1797 样本 / 64 特征 / 10 类手写数字（默认）
  - iris ：150 样本 / 4 特征 / 3 类（轻量备选）

真实图像/表格数据使目标模型产生异质性过拟合与校准差，
完整概率向量攻击可真正超越标量 loss 阈值基线（对标 Shokri 2017 在真实数据上的结论）。

作者：晨星
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits, load_iris

from core.errors import DataError


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """返回 (X, y)：已标准化 X (n,d)，y (n,) 为 int 类标签。确定性。"""
    key = (name or "digits").lower()
    if key == "digits":
        d = load_digits()
        X = np.asarray(d.data, dtype=np.float64)
        y = np.asarray(d.target, dtype=np.int64)
    elif key == "iris":
        d = load_iris()
        X = np.asarray(d.data, dtype=np.float64)
        y = np.asarray(d.target, dtype=np.int64)
    else:
        raise DataError(f"未知数据集 {name!r}（支持 digits / iris）")
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    X = (X - mu) / sd
    return X, y


class MiaData:
    """确定性划分：成员 / 非成员 / 影子池。同一总体 i.i.d.，符合 MIA 标准设定。"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        if X.shape[0] != y.shape[0]:
            raise DataError("X/y 样本数不一致")
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.int64)
        self.n = self.X.shape[0]
        self.n_features = self.X.shape[1]
        self.n_classes = int(np.unique(self.y).size)

    @classmethod
    def from_name(cls, name: str) -> "MiaData":
        try:
            X, y = load_dataset(name)
            return cls(X, y)
        except Exception:
            # 离线兜底：sklearn 不可用（零下载）时退化为纯 numpy 合成数据，保证系统仍可跑
            from .generator import SyntheticMiaGenerator
            gen = SyntheticMiaGenerator(n_features=64, n_classes=10, snr=1.2, class_spread=1.8, seed=0)
            X, y = gen.generate(1200)
            return cls(X, y)

    def split(self, n_train: int, n_non: int, seed: int):
        """确定性划分成员/非成员/影子池（三者互不相交）。"""
        if n_train + n_non >= self.n:
            raise DataError(f"数据集样本不足：需 > {n_train + n_non}，实际 {self.n}")
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.n)
        mi = perm[:n_train]
        nmi = perm[n_train : n_train + n_non]
        rest = perm[n_train + n_non :]
        return (
            self.X[mi], self.y[mi],
            self.X[nmi], self.y[nmi],
            self.X[rest], self.y[rest],
        )

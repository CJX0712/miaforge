"""MiaForge 关键抽象（Protocol），用于单测 mock 与后端可插拔。

作者：晨星
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TargetModel(Protocol):
    """目标模型：被攻击的黑盒，输出概率。"""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def loss(self, X: np.ndarray, y: np.ndarray) -> float: ...


@runtime_checkable
class Attack(Protocol):
    """攻击者：输入特征矩阵，输出「越像成员分数越高」的 score。"""

    name: str

    def fit(self, feats: np.ndarray, labels: np.ndarray) -> None: ...

    def score(self, feats: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class DataGenerator(Protocol):
    """合成数据生成器（固定 seed 可复现）。"""

    def generate(self, n: int) -> tuple[np.ndarray, np.ndarray]: ...

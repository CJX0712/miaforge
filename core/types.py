"""MiaForge 核心数据类型（dataclass）。

作者：晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AttackResult:
    """单个攻击方法的评测结果。"""

    name: str
    auc: float
    accuracy: float
    n_eval: int


@dataclass
class Split:
    """泄漏安全的攻击训练/评测切分（成员与非成员各自半分）。"""

    X_m_tr: np.ndarray
    y_m_tr: np.ndarray
    X_m_te: np.ndarray
    y_m_te: np.ndarray
    X_nm_tr: np.ndarray
    y_nm_tr: np.ndarray
    X_nm_te: np.ndarray
    y_nm_te: np.ndarray


@dataclass
class SeedRecord:
    """单 seed 运行记录。"""

    seed: int
    methods: dict = field(default_factory=dict)  # name -> auc
    accuracy: dict = field(default_factory=dict)  # name -> accuracy
    target_train_loss: float = 0.0
    target_eval_loss: float = 0.0
    overfit_gap: float = 0.0


@dataclass
class Aggregate:
    """多 seed 聚合（mean ± std）。"""

    name: str
    mean: float
    std: float
    n: int

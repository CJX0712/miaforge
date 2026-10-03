"""全局确定性控制。

唯一入口 ``set_all(seed)``：一次性设齐 numpy 全局种子与 stdlib random 种子，
并把派生子种子的约定固定下来（组件用 ``seed + offset`` 派生独立 Generator）。

作者：晨星
"""

from __future__ import annotations

import random

import numpy as np

_STATE: dict = {}


def set_all(seed: int | None):
    """设齐全部随机源。返回当前 seed 信息字典。"""
    if seed is None:
        return {"seed": None}
    np.random.seed(seed)
    random.seed(seed)
    _STATE["seed"] = seed
    _STATE["rng"] = np.random.default_rng(seed)
    return {"seed": seed}


def derive_rng(seed: int | None, offset: int = 0):
    """从主 seed 派生一个独立、确定性的 numpy Generator。"""
    if seed is None:
        seed = 0
    return np.random.default_rng(int(seed) + int(offset))

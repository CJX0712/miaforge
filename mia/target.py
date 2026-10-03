"""纯 numpy 多层感知机（目标模型），确定性、零重型依赖。

- 全批量 Adam 优化交叉熵 + softmax
- 训练前仅用成员训练集统计量做标准化（防止尺度漂移）
- 该模型在小样本高容量设定下会过拟合 → 产生成员信号

作者：晨星
"""

from __future__ import annotations

import numpy as np

from core.errors import TargetError


class NumpyMLP:
    """确定性 numpy MLP（用于目标模型与参考模型）。"""

    def __init__(
        self,
        hidden: tuple[int, ...] = (32,),
        n_classes: int = 2,
        epochs: int = 300,
        lr: float = 0.01,
        seed: int = 0,
        verbose: bool = False,
    ):
        self.hidden = tuple(hidden)
        self.n_classes = n_classes
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.verbose = verbose
        self.Ws: list[np.ndarray] = []
        self.bs: list[np.ndarray] = []
        self._means: np.ndarray | None = None
        self._stds: np.ndarray | None = None

    # —— 内部工具 ——
    @staticmethod
    def _xavier(rng: np.random.Generator, n_in: int, n_out: int) -> np.ndarray:
        return rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / (n_in + n_out))

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_deriv(x: np.ndarray) -> np.ndarray:
        return (x > 0.0).astype(np.float64)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _forward(self, Xz: np.ndarray):
        acts: list[np.ndarray] = [Xz]
        pre: list[np.ndarray] = []
        A = Xz
        for i in range(len(self.Ws)):
            z = A @ self.Ws[i] + self.bs[i]
            pre.append(z)
            # 末层 softmax，其余层 ReLU
            A = self._relu(z) if i < len(self.Ws) - 1 else self._softmax(z)
            acts.append(A)
        return acts, pre

    # —— 训练 ——
    def fit(self, X: np.ndarray, y: np.ndarray) -> NumpyMLP:
        if X.ndim != 2:
            raise TargetError("X 必须是二维")
        n, d = X.shape
        if n != y.shape[0]:
            raise TargetError("X/y 样本数不一致")
        rng = np.random.default_rng(self.seed)
        layers = [d, *self.hidden, self.n_classes]
        self.Ws = [
            self._xavier(rng, layers[i], layers[i + 1]) for i in range(len(layers) - 1)
        ]
        self.bs = [np.zeros(layers[i + 1]) for i in range(len(layers) - 1)]

        self._means = X.mean(axis=0)
        self._stds = X.std(axis=0) + 1e-8
        Xz = (X - self._means) / self._stds

        Yoh = np.zeros((n, self.n_classes), dtype=np.float64)
        Yoh[np.arange(n), y.astype(int)] = 1.0

        betas = (0.9, 0.999)
        eps = 1e-8
        # Adam 时刻计数从 1 开始（偏差修正项 betas**t 要求首次更新用 t=1）
        mW = [np.zeros_like(w) for w in self.Ws]
        vW = [np.zeros_like(w) for w in self.Ws]
        mb = [np.zeros_like(b) for b in self.bs]
        vb = [np.zeros_like(b) for b in self.bs]

        # 全批量 Adam（向量化，逐 epoch 一次前向/反向；小样本下充分过拟合）
        for t in range(1, self.epochs + 1):
            acts, pre = self._forward(Xz)
            dZ = acts[-1] - Yoh  # (n, K) 交叉熵 + softmax 梯度
            dA = dZ
            gW: list[np.ndarray] = []
            gb: list[np.ndarray] = []
            for li in range(len(self.Ws) - 1, -1, -1):
                A_prev = acts[li]
                gW.insert(0, A_prev.T @ dA)
                gb.insert(0, dA.sum(axis=0))
                if li > 0:
                    dA = (dA @ self.Ws[li].T) * self._relu_deriv(pre[li - 1])
            for li in range(len(self.Ws)):
                mW[li] = betas[0] * mW[li] + (1 - betas[0]) * gW[li]
                vW[li] = betas[1] * vW[li] + (1 - betas[1]) * (gW[li] ** 2)
                mhat = mW[li] / (1 - betas[0] ** t)
                vhat = vW[li] / (1 - betas[1] ** t)
                self.Ws[li] -= self.lr * mhat / (np.sqrt(vhat) + eps)
                mb[li] = betas[0] * mb[li] + (1 - betas[0]) * gb[li]
                vb[li] = betas[1] * vb[li] + (1 - betas[1]) * (gb[li] ** 2)
                mbhat = mb[li] / (1 - betas[0] ** t)
                vbhat = vb[li] / (1 - betas[1] ** t)
                self.bs[li] -= self.lr * mbhat / (np.sqrt(vbhat) + eps)
        return self

    # —— 推理 ——
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._means is None:
            raise TargetError("模型未训练")
        Xz = (X - self._means) / self._stds
        acts, _ = self._forward(Xz)
        return acts[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        p = self.predict_proba(X)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        n = y.shape[0]
        return float(-np.log(p[np.arange(n), y.astype(int)]).mean())

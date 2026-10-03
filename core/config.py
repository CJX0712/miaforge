"""MiaForge 运行配置（ENV_MIA_* 覆盖 + schema 校验）。

作者：晨星
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

from .errors import ConfigError


@dataclass
class MiaConfig:
    """端到端配置。所有超参集中于此，便于复现与调参。"""

    # —— 数据 / 难度旋钮 ——
    dataset: str = "digits"     # 真实数据集（digits / iris）；"synthetic" 走高斯生成器（参考 DGP）
    seed: int = 0
    n_train: int = 200          # 成员数（目标模型训练集）
    n_nonmember: int = 200      # 非成员数（同分布留出）
    n_features: int = 64         # 真实数据集下由数据覆盖
    n_classes: int = 10         # 真实数据集下由数据覆盖
    snr: float = 1.2            # 类间分离度（synthetic 专用），控制难度
    class_spread: float = 1.8   # 每类难度跨度（synthetic 专用），拉开类难度混淆
    member_label_noise: float = 0.0  # 训练集错标率（synthetic 专用）：目标/影子模型在错标样本上训练并记住，
                                     # 被记住的错标成员留下「真类loss高+预测类向量异常」签名，
                                     # 标量loss阈值无法捕捉，完整概率向量可捕获（真实MIA攻击面）

    # —— 目标模型（numpy MLP）——
    hidden: Tuple[int, ...] = (32,)
    epochs: int = 80
    lr: float = 0.02

    # —— 影子模型（Shokri 攻击的迁移监督来源）——
    n_shadows: int = 10
    shadow_pool: int = 3000
    shadow_hidden: Tuple[int, ...] = (16,)
    shadow_epochs: int = 80

    # —— 基准 / 发布 ——
    n_seeds: int = 3
    base_seed: int = 0
    gate_delta: float = 0.02     # 主门禁：旗舰 AUC − 基线 AUC ≥ 此绝对差
    verbose: bool = False

    @classmethod
    def from_env(cls) -> "MiaConfig":
        def get(name: str, cast, default):
            raw = os.environ.get(f"MIA_{name}")
            if raw is None:
                return default
            try:
                if cast is tuple:
                    return tuple(int(x) for x in raw.split(",") if x != "") or default
                return cast(raw)
            except Exception as exc:  # noqa: BLE001
                raise ConfigError(f"invalid MIA_{name}={raw!r} ({exc})") from exc

        return cls(
            seed=get("SEED", int, cls.seed),
            n_train=get("N_TRAIN", int, cls.n_train),
            n_nonmember=get("N_NONMEMBER", int, cls.n_nonmember),
            n_features=get("N_FEATURES", int, cls.n_features),
            n_classes=get("N_CLASSES", int, cls.n_classes),
            snr=get("SNR", float, cls.snr),
            class_spread=get("CLASS_SPREAD", float, cls.class_spread),
            member_label_noise=get("MEMBER_LABEL_NOISE", float, cls.member_label_noise),
            dataset=get("DATASET", str, cls.dataset),
            hidden=get("HIDDEN", tuple, cls.hidden),
            epochs=get("EPOCHS", int, cls.epochs),
            lr=get("LR", float, cls.lr),
            n_shadows=get("N_SHADOWS", int, cls.n_shadows),
            shadow_pool=get("SHADOW_POOL", int, cls.shadow_pool),
            shadow_hidden=get("SHADOW_HIDDEN", tuple, cls.shadow_hidden),
            shadow_epochs=get("SHADOW_EPOCHS", int, cls.shadow_epochs),
            n_seeds=get("N_SEEDS", int, cls.n_seeds),
            base_seed=get("BASE_SEED", int, cls.base_seed),
            gate_delta=get("GATE_DELTA", float, cls.gate_delta),
        )

    def validate(self) -> "MiaConfig":
        if self.seed < 0:
            raise ConfigError("seed 必须 >= 0")
        if self.n_train < 20 or self.n_nonmember < 20:
            raise ConfigError("n_train / n_nonmember 过小")
        if self.n_features < 1:
            raise ConfigError("n_features >= 1")
        if self.n_classes < 2:
            raise ConfigError("n_classes >= 2")
        if self.snr <= 0:
            raise ConfigError("snr > 0")
        if self.class_spread < 0:
            raise ConfigError("class_spread >= 0")
        if not (0.0 <= self.member_label_noise < 1.0):
            raise ConfigError("member_label_noise ∈ [0, 1)")
        if self.dataset not in ("digits", "iris", "synthetic"):
            raise ConfigError("dataset ∈ {digits, iris, synthetic}")
        if self.epochs < 1:
            raise ConfigError("epochs >= 1")
        if self.shadow_epochs < 1:
            raise ConfigError("shadow_epochs >= 1")
        if self.n_seeds < 1:
            raise ConfigError("n_seeds >= 1")
        if self.gate_delta <= 0:
            raise ConfigError("gate_delta > 0")
        if self.n_train % 2 != 0 or self.n_nonmember % 2 != 0:
            raise ConfigError("n_train / n_nonmember 需为偶数（攻击半分）")
        return self

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "n_train": self.n_train,
            "n_nonmember": self.n_nonmember,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "snr": self.snr,
            "class_spread": self.class_spread,
            "member_label_noise": self.member_label_noise,
            "dataset": self.dataset,
            "hidden": list(self.hidden),
            "epochs": self.epochs,
            "lr": self.lr,
            "n_shadows": self.n_shadows,
            "shadow_pool": self.shadow_pool,
            "shadow_hidden": list(self.shadow_hidden),
            "shadow_epochs": self.shadow_epochs,
            "n_seeds": self.n_seeds,
            "base_seed": self.base_seed,
            "gate_delta": self.gate_delta,
        }

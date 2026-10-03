"""MiaForge 核心测试：确定性、离线兜底、攻击契约。

作者：晨星
"""

from __future__ import annotations

import numpy as np

from core.config import MiaConfig
from pipeline.pipeline import benchmark
from mia.attacks import build_features, make_attacks
from mia.target import NumpyMLP
from data.generator import SyntheticMiaGenerator


def test_determinism_digits():
    """同一配置两次运行，核心数值必须逐位一致（确定性门禁）。"""
    cfg = MiaConfig(n_seeds=3, base_seed=0)
    r1 = benchmark(cfg)
    r2 = benchmark(cfg)
    assert r1["gate"] == r2["gate"], "gate 必须确定性一致"
    assert r1["aggregate"] == r2["aggregate"], "aggregate 必须确定性一致"
    assert r1["gate"]["passed"] is True, "digits 默认配置应过门禁"


def test_offline_synthetic_runs():
    """离线兜底：纯 numpy 合成数据路径不依赖 sklearn，可正常运行并产出报告。"""
    cfg = MiaConfig(dataset="synthetic", n_train=200, n_nonmember=200,
                   n_features=20, n_classes=10, n_shadows=5, n_seeds=2, base_seed=0)
    rep = benchmark(cfg)
    assert set(rep["aggregate"].keys()) >= {"loss_threshold", "shadow_all"}
    assert 0.0 <= rep["gate"]["flagship_auc_mean"] <= 1.0


def test_attacks_contract():
    """攻击集合契约：名称固定、loss 阈值基线 score = -loss。"""
    atks = make_attacks(seed=0, n_classes=10)
    names = [a.name for a in atks]
    assert names == ["loss_threshold", "shadow_all", "shadow_loss_only", "shadow_no_correct"]

    gen = SyntheticMiaGenerator(20, 10, 1.2, 1.8, seed=0)
    X, y = gen.generate(50)
    m = NumpyMLP(hidden=(16,), n_classes=10, epochs=10, seed=1).fit(X, y)
    feats = build_features(m, X, y)
    thr = [a for a in atks if a.name == "loss_threshold"][0]
    sc = thr.score(feats)
    np.testing.assert_allclose(sc, -feats[:, 0], atol=1e-9)


def test_demo_numbers_plausible():
    """旗舰攻击应明显优于单 loss 基线（delta>0）。"""
    cfg = MiaConfig(n_seeds=3, base_seed=0)
    rep = benchmark(cfg)
    delta = rep["gate"]["flagship_auc_mean"] - rep["gate"]["baseline_auc_mean"]
    assert delta > 0.02, f"旗舰应显著超越基线，实际 delta={delta}"

"""端到端流水线：单 seed 协议 + 多 seed 基准聚合 + 门禁 + 消融 + 失败案例。

作者：晨星
"""

from __future__ import annotations

import numpy as np

from core.config import MiaConfig
from core.seed import set_all
from core.types import Aggregate, AttackResult, SeedRecord, Split
from data.dataset import MiaData
from data.generator import SyntheticMiaGenerator
from mia.attacks import build_features, make_attacks
from mia.metrics import optimal_threshold_balanced_accuracy, roc_auc
from mia.target import NumpyMLP


def _apply_label_noise(
    y: np.ndarray, rate: float, rng: np.random.Generator, n_classes: int
) -> np.ndarray:
    """将确定比例 y 翻到另一类（训练集标签损坏，模型记住被损坏的记录）。"""
    y = y.copy()
    if rate <= 0 or n_classes < 2:
        return y
    n = y.shape[0]
    k = round(rate * n)  # py3 round(float) 已返回 int，无需再 int() 包裹
    if k == 0:
        return y
    idx = rng.choice(n, size=k, replace=False)
    for i in idx:
        y[i] = (int(y[i]) + 1 + int(rng.integers(0, n_classes - 1))) % n_classes
    return y


def _source_split(cfg: MiaConfig, seed: int):
    """返回 (Xm,ym, Xnm,ynm, Xpool,ypool)：成员 / 非成员 / 影子池，三者互不相交。

    真实数据集（digits/iris）走 MiaData；synthetic 走确定性高斯生成器。
    """
    if cfg.dataset == "synthetic":
        gen_m = SyntheticMiaGenerator(
            cfg.n_features, cfg.n_classes, cfg.snr, cfg.class_spread, seed=seed + 100
        )
        gen_nm = SyntheticMiaGenerator(
            cfg.n_features, cfg.n_classes, cfg.snr, cfg.class_spread, seed=seed + 200
        )
        gen_pool = SyntheticMiaGenerator(
            cfg.n_features, cfg.n_classes, cfg.snr, cfg.class_spread, seed=seed + 500
        )
        Xm, ym = gen_m.generate(cfg.n_train)
        Xnm, ynm = gen_nm.generate(cfg.n_nonmember)
        Xp, yp = gen_pool.generate(cfg.shadow_pool)
        return Xm, ym, Xnm, ynm, Xp, yp
    data = MiaData.from_name(cfg.dataset)
    return data.split(cfg.n_train, cfg.n_nonmember, seed + 100)


def _train_shadow_data(cfg: MiaConfig, seed: int, Xpool: np.ndarray, ypool: np.ndarray):
    """训练 M 个影子模型，积累「特征→成员」监督数据（用于迁移攻击训练）。"""
    n = Xpool.shape[0]
    rng = np.random.default_rng(seed + 7)
    rng_noise = np.random.default_rng(seed + 77)
    feats_rows = []
    labels_rows = []
    for m in range(cfg.n_shadows):
        perm = rng.permutation(n)
        in_idx = perm[: cfg.n_train]
        out_idx = perm[cfg.n_train : cfg.n_train + cfg.n_nonmember]
        X_in, y_in = Xpool[in_idx], ypool[in_idx]
        X_out, y_out = Xpool[out_idx], ypool[out_idx]
        y_in_train = _apply_label_noise(
            y_in, cfg.member_label_noise, rng_noise, cfg.n_classes
        )
        shadow = NumpyMLP(
            hidden=cfg.hidden,
            n_classes=cfg.n_classes,
            epochs=cfg.epochs,
            lr=cfg.lr,
            seed=seed + 100 + m,
        )
        shadow.fit(X_in, y_in_train)
        f_in = build_features(shadow, X_in, y_in)
        f_out = build_features(shadow, X_out, y_out)
        feats_rows.append(f_in)
        feats_rows.append(f_out)
        labels_rows.append(np.ones(f_in.shape[0]))
        labels_rows.append(np.zeros(f_out.shape[0]))

    feats = np.vstack(feats_rows).astype(np.float64)
    labels = np.concatenate(labels_rows).astype(np.float64)
    return feats, labels


def _protocol(cfg: MiaConfig, seed: int):
    """执行单 seed 的完整协议，返回所有中间产物（供评测与失败案例分析）。"""
    set_all(seed)
    Xm, ym, Xnm, ynm, Xpool, ypool = _source_split(cfg, seed)

    # —— 攻击训练/评测半分（泄漏安全）——
    rng = np.random.default_rng(seed + 1)
    m_perm = rng.permutation(cfg.n_train)
    nm_perm = rng.permutation(cfg.n_nonmember)
    hm = cfg.n_train // 2
    hn = cfg.n_nonmember // 2
    X_m_tr, y_m_tr = Xm[m_perm[:hm]], ym[m_perm[:hm]]
    X_m_te, y_m_te = Xm[m_perm[hm:]], ym[m_perm[hm:]]
    X_nm_tr, y_nm_tr = Xnm[nm_perm[:hn]], ynm[nm_perm[:hn]]
    X_nm_te, y_nm_te = Xnm[nm_perm[hn:]], ynm[nm_perm[hn:]]

    split = Split(X_m_tr, y_m_tr, X_m_te, y_m_te, X_nm_tr, y_nm_tr, X_nm_te, y_nm_te)

    # —— 目标模型：在全部成员上训练（synthetic 路径按 rate 注入错标）——
    rng_n = np.random.default_rng(seed + 3)
    ym_train = _apply_label_noise(ym, cfg.member_label_noise, rng_n, cfg.n_classes)
    target = NumpyMLP(
        hidden=cfg.hidden,
        n_classes=cfg.n_classes,
        epochs=cfg.epochs,
        lr=cfg.lr,
        seed=seed + 2,
    )
    target.fit(Xm, ym_train)

    # —— 影子模型监督数据 ——
    shadow_feats, shadow_labels = _train_shadow_data(cfg, seed, Xpool, ypool)

    # —— 目标样本特征（成员 ∪ 非成员）——
    X_te = np.vstack([X_m_te, X_nm_te])
    y_te = np.concatenate([y_m_te, y_nm_te])
    labels_te = np.concatenate([np.ones(X_m_te.shape[0]), np.zeros(X_nm_te.shape[0])])
    mat_te = build_features(target, X_te, y_te)

    # —— 逐攻击者评测（攻击在影子数据上训练，迁移到目标评测集）——
    # 统一标准化：影子 + 目标特征合并估计统计量（仅缩放，无标签泄漏），消除域偏移
    ref_feats = np.vstack([shadow_feats, mat_te])
    results = []
    scores_te = {}
    for atk in make_attacks(seed=seed, n_classes=cfg.n_classes):
        if hasattr(atk, "set_standardize"):
            atk.set_standardize(ref_feats)
        atk.fit(shadow_feats, shadow_labels)
        sc = atk.score(mat_te)
        t = optimal_threshold_balanced_accuracy(atk.score(shadow_feats), shadow_labels)
        auc = roc_auc(labels_te, sc)
        acc = float(((sc >= t).astype(int) == labels_te.astype(int)).mean())
        results.append(AttackResult(atk.name, auc, acc, len(labels_te)))
        scores_te[atk.name] = sc

    target_train_loss = target.loss(Xm, ym)
    target_eval_loss = target.loss(Xnm, ynm)
    overfit_gap = target_eval_loss - target_train_loss

    return {
        "seed": seed,
        "split": split,
        "target": target,
        "mat_te": mat_te,
        "labels_te": labels_te,
        "scores_te": scores_te,
        "results": results,
        "target_train_loss": target_train_loss,
        "target_eval_loss": target_eval_loss,
        "overfit_gap": overfit_gap,
    }


def run_one(cfg: MiaConfig, seed: int) -> SeedRecord:
    p = _protocol(cfg, seed)
    return _record_from_protocol(p)


def _record_from_protocol(p: dict) -> SeedRecord:
    rec = SeedRecord(seed=p["seed"])
    rec.target_train_loss = p["target_train_loss"]
    rec.target_eval_loss = p["target_eval_loss"]
    rec.overfit_gap = p["overfit_gap"]
    for r in p["results"]:
        rec.methods[r.name] = r.auc
        rec.accuracy[r.name] = r.accuracy
    return rec


def _aggregate(records: list[SeedRecord], name: str) -> Aggregate:
    vals = np.array([r.methods[name] for r in records], dtype=np.float64)
    return Aggregate(
        name=name, mean=float(vals.mean()), std=float(vals.std()), n=len(vals)
    )


def collect_failures(protocol: dict) -> list[dict]:
    p = protocol
    sc = p["scores_te"]["shadow_all"]
    labels = p["labels_te"]
    mat = p["mat_te"]
    t = optimal_threshold_balanced_accuracy(p["scores_te"]["shadow_all"], labels)
    pred = (sc >= t).astype(int)
    n_te = labels.shape[0]

    false_neg, false_pos = [], []
    for i in range(n_te):
        if labels[i] == 1 and pred[i] == 0:
            false_neg.append(i)
        elif labels[i] == 0 and pred[i] == 1:
            false_pos.append(i)

    cases = []
    if false_neg:
        i = false_neg[0]
        cases.append(
            {
                "type": "false_negative_member",
                "desc": "真实成员被攻击判为非成员",
                "loss": round(float(mat[i, 0]), 4),
                "confidence": round(float(mat[i, 1]), 4),
                "correct": int(mat[i, 4]),
                "cause": (
                    "该成员样本处于类间重叠区或含噪声，"
                    "目标模型未充分记忆，攻击按特征判为非成员"
                ),
            }
        )
    if false_pos:
        i = false_pos[0]
        cases.append(
            {
                "type": "false_positive_nonmember",
                "desc": "真实非成员被攻击判为成员",
                "loss": round(float(mat[i, 0]), 4),
                "confidence": round(float(mat[i, 1]), 4),
                "correct": int(mat[i, 4]),
                "cause": (
                    "该非成员恰为易分样本，loss 与置信度接近典型成员，攻击难以区分"
                ),
            }
        )
    member_idx = np.where(labels == 1)[0]
    if member_idx.size:
        i = member_idx[int(np.argmax(mat[member_idx, 0]))]
        cases.append(
            {
                "type": "high_loss_member",
                "desc": "成员中 loss 最高者",
                "loss": round(float(mat[i, 0]), 4),
                "confidence": round(float(mat[i, 1]), 4),
                "correct": int(mat[i, 4]),
                "cause": (
                    "成员样本含高噪声或位于决策边界，目标模型似未记住 → 攻击判为非成员"
                ),
            }
        )
    nm_idx = np.where(labels == 0)[0]
    if nm_idx.size:
        i = nm_idx[int(np.argmin(mat[nm_idx, 0]))]
        cases.append(
            {
                "type": "low_loss_nonmember",
                "desc": "非成员中 loss 最低者",
                "loss": round(float(mat[i, 0]), 4),
                "confidence": round(float(mat[i, 1]), 4),
                "correct": int(mat[i, 4]),
                "cause": "非成员恰为易分样本，loss 接近典型成员，攻击典型难例",
            }
        )
    return cases[:6]


def benchmark(cfg: MiaConfig) -> dict:
    cfg.validate()
    # 真实数据集：同步特征/类别维度到 config
    if cfg.dataset != "synthetic":
        probe = MiaData.from_name(cfg.dataset)
        cfg.n_features = probe.n_features
        cfg.n_classes = probe.n_classes
    protocols = [_protocol(cfg, cfg.base_seed + s) for s in range(cfg.n_seeds)]
    records = [_record_from_protocol(p) for p in protocols]

    method_names = list(records[0].methods.keys())
    aggs = {name: _aggregate(records, name) for name in method_names}

    base = aggs["loss_threshold"]
    flag = aggs["shadow_all"]
    delta = flag.mean - base.mean
    sig = delta > 0.5 * (flag.std + base.std)
    passed = (delta >= cfg.gate_delta) and sig

    ablation = {}
    for name in method_names:
        if name in ("shadow_all", "loss_threshold"):
            continue
        a = aggs[name]
        ablation[name] = {
            "auc_mean": round(a.mean, 4),
            "auc_std": round(a.std, 4),
            "delta_vs_flagship": round(a.mean - flag.mean, 4),
        }

    failures = collect_failures(protocols[0])

    return {
        "system": "MiaForge",
        "config": cfg.as_dict(),
        "per_seed": [
            {
                "seed": r.seed,
                "methods": {k: round(v, 4) for k, v in r.methods.items()},
                "accuracy": {k: round(v, 4) for k, v in r.accuracy.items()},
                "target_train_loss": round(r.target_train_loss, 4),
                "target_eval_loss": round(r.target_eval_loss, 4),
                "overfit_gap": round(r.overfit_gap, 4),
            }
            for r in records
        ],
        "aggregate": {
            name: {"mean": round(a.mean, 4), "std": round(a.std, 4), "n": a.n}
            for name, a in aggs.items()
        },
        "gate": {
            "baseline": "loss_threshold",
            "flagship": "shadow_all",
            "baseline_auc_mean": round(base.mean, 4),
            "flagship_auc_mean": round(flag.mean, 4),
            "delta": round(delta, 4),
            "gate_delta_required": cfg.gate_delta,
            "significant": bool(sig),
            "passed": bool(passed),
        },
        "ablation": ablation,
        "failures": failures,
    }

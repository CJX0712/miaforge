"""攻击评测指标（纯 numpy，确定性，含并列处理）。

作者：晨星
"""

from __future__ import annotations

import numpy as np


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC。y_true: 1=成员, 0=非成员；score 越大越像成员。

    采用平均秩法处理并列（与 sklearn roc_auc_score 在 tie 上一致）。
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if y.size != s.size or y.size == 0:
        raise ValueError("y_true / scores 长度必须一致且非空")
    n_pos = float(y.sum())
    n_neg = float(y.size - n_pos)
    if n_pos == 0.0 or n_neg == 0.0:
        return 0.5

    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    yy = y[order]

    ranks = np.empty(y.size, dtype=np.float64)
    i = 0
    while i < y.size:
        j = i
        while j < y.size - 1 and ss[j + 1] == ss[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2.0 + 1.0  # 1-based 平均秩
        i = j + 1

    sum_ranks_pos = ranks[yy == 1.0].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc)


def balanced_accuracy_at_threshold(
    scores: np.ndarray, y_true: np.ndarray, threshold: float
) -> float:
    """给定阈值（score>=threshold 判为成员）的均衡准确率。"""
    pred = (np.asarray(scores) >= threshold).astype(np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    tp = ((pred == 1) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    p = max(float((y == 1).sum()), 1.0)
    n = max(float((y == 0).sum()), 1.0)
    return float(tp / p * 0.5 + tn / n * 0.5)


def optimal_threshold_balanced_accuracy(
    scores: np.ndarray, y_true: np.ndarray
) -> float:
    """在训练集上搜索使均衡准确率最高的阈值（仅用训练侧，避免泄漏到评测侧）。"""
    s = np.asarray(scores, dtype=np.float64)
    # 候选阈值：各 score 与相邻中值
    cand = np.unique(np.concatenate([s, (s[:-1] + s[1:]) / 2.0]))
    best_t, best_acc = cand[0], -1.0
    for t in cand:
        acc = balanced_accuracy_at_threshold(s, y_true, t)
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return float(best_t)

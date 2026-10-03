# MiaForge 架构说明

作者：晨星

## 1. 设计原则

- **离线 + 确定性**：无外部下载（默认用 `sklearn` 本地内置数据集；缺失时纯 numpy 合成兜底），同 seed 逐位复现。
- **单向无环**：`core → data → mia → pipeline → cli`，无循环依赖。
- **统一语义**：所有攻击者输出「分数越大越像成员」，AUC 跨方法可比。
- **泄漏安全**：成员/非成员在协议起始即固定切分；攻击阈值仅在影子侧搜索，绝不接触评测侧标签。

## 2. 模块职责

### core/
| 文件 | 职责 |
|------|------|
| `config.py` | `MiaConfig` 集中所有超参；`ENV_MIA_*` 覆盖；schema 校验（含 `validate()`） |
| `errors.py` | 错误码体系 `E1xx~E5xx`，所有异常携带稳定 code |
| `seed.py` | `set_all(seed)` 设齐随机源；`derive_rng` 派生子 Generator |
| `types.py` | `AttackResult / Split / SeedRecord / Aggregate` 数据结构 |
| `interfaces.py` | `TargetModel / Attack / DataGenerator` Protocol（可插拔后端） |

### data/
- `dataset.py`：`MiaData` 确定性划分成员/非成员/影子池（三者互不相交）；`load_dataset` 加载 `digits`/`iris`（标准化后返回）；`from_name` 带**离线兜底**（sklearn 缺失 → 纯 numpy 合成）。
- `generator.py`：`SyntheticMiaGenerator` 类条件高斯 DGP（含每类难度跨度），作为参考级 DGP 与离线兜底数据源（`dataset="synthetic"` 可选）。

### mia/
- `target.py`：`NumpyMLP` 纯 numpy MLP（全批量 Adam + softmax 交叉熵），确定性。
- `attacks.py`：`build_features` 固定特征契约；`LossThresholdAttack`（基线）、`ShadowAttack`（旗舰 + 消融变体），攻击分类器为纯 numpy 逻辑回归。
- `metrics.py`：`roc_auc`（平均秩法处理并列）、均衡准确率、最优阈值搜索。

### pipeline/
- `_protocol(cfg, seed)`：单 seed 完整协议 —— 数据切分 → 训练目标 → 训练影子积累监督 → 提取目标特征 → 逐攻击评测。
- `benchmark(cfg)`：多 seed 聚合 → 门禁 → 消融 → 失败案例收集。
- `collect_failures`：复用 `_protocol` 产物（不重训），提取 ≥3 条典型误例 + 归因。

### cli.py / examples/run_demo.py
- `cli.py`：`benchmark` / `run` / `demo` 三个子命令。
- `run_demo.py`：多 seed 基准 + 二次运行**确定性逐位自检**，落盘 `benchmark.json`。

## 3. 数据流（单 seed）

```
MiaData.split(seed) → (Xm,ym) 成员 ∪ (Xnm,ynm) 非成员 ∪ (Xpool,ypool) 影子池
        │
        ├─ 成员半分：训练侧 / 评测侧（泄漏安全）
        ├─ 目标模型 NumpyMLP.fit(Xm)  →  build_features(目标, X_te) = mat_te
        │
        └─ 影子池 → K 个 NumpyMLP 各在 (in/out) 上训练 → 累积 (feats, 标签) 监督
                    │
                    ▼
        统一标准化(影子+目标) → 每个攻击 fit(影子监督) → score(mat_te) → roc_auc
```

## 4. 关键不变量（手写实现校验点）

- **特征契约**：`build_features` 列布局固定（loss/conf/entropy/margin/correct/概率向量），下游严格按索引取用。
- **确定性**：`set_all` + 零初始化逻辑回归 + 固定派生 seed，二次运行 `benchmark` 核心 AUC 逐位一致。
- **门禁**：旗舰 AUC 均值 − 基线 AUC 均值 ≥ `gate_delta`(0.02) 且 `delta > ½(σ₁+σ₂)` 显著。
- **消融单调**：`shadow_loss_only`（仅 loss）应 ≤ 旗舰；去 `correct` 不应显著优于旗舰（一致佐证增益来自概率向量而非单特征）。

## 5. 失败案例结构

每条含：`type`（false_negative_member / false_positive_nonmember / high_loss_member / low_loss_nonmember）、`loss / confidence / correct` 取值与**归因说明**，供可解释性分析。

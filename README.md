# MiaForge · 成员推理攻击（Membership Inference Attack）系统

> 判断「某个样本是否参与过目标模型的训练」—— 隐私攻击领域的王牌问题。
> 作者：晨星

MiaForge 是一套**离线、确定性、纯 Python（numpy 优先）**的端到端 MIA 系统：在真实确定性数据集（`sklearn.load_digits`）上训练一个被攻击的目标模型，用 **Shokri 影子模型攻击**（融合完整概率向量 + loss/置信度/熵/间隔/正确率多线索）推断成员身份，并**显著超越经典 loss 阈值基线（Yeom 2018 blind）**。

---

## 核心结论（默认配置，3 seed 聚合）

| 方法 | AUC | 说明 |
|------|-----|------|
| `loss_threshold`（基线 / Yeom blind） | **0.592** | 仅用真类 loss，`score = -loss` |
| **`shadow_all`（旗舰 / Shokri 影子攻击）** | **0.651** | 多影子模型监督 + 完整概率向量逻辑回归 |
| `shadow_loss_only`（消融①） | 0.592 | 影子攻击但仅用 loss → 退回基线 |
| `shadow_no_correct`（消融②） | 0.653 | 去除 `correct` 特征（几乎无损） |

- **主门禁**：旗舰 AUC − 基线 AUC = **+0.059 ≥ 0.02**，且均值差 > ½(σ₁+σ₂) 显著 → **PASS**
- **确定性**：同 seed 两次运行核心数值**逐位一致**
- **离线兜底**：sklearn 缺失时自动退化到纯 numpy 合成数据，系统仍可跑

> 为什么攻击能赢？在真实图像数据上，目标模型对训练样本存在**异质性过拟合与校准差**——被记住的样本不仅 loss 更低，其完整概率向量（各类置信度分布）也呈现成员专属签名。标量 loss 阈值只能利用单点似然，而完整向量攻击学到了「逐类校准」的边界，这正是 Shokri 等人在 CIFAR-10/100 上报告的现象。

---

## 快速开始

```bash
# 1. 建隔离环境并装依赖
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt

# 2. 跑端到端演示（基准 + 确定性自检，约 4s）
python cli.py demo --out examples/benchmark.json

# 3. 单 seed 探查
python cli.py run --seed 0

# 4. 自定义基准
python cli.py benchmark --n-train 200 --n-seeds 5 --out report.json
```

环境变量覆盖：所有配置项支持 `ENV_MIA_*` 覆盖（如 `MIA_EPOCHS=120 MIA_N_SHADOWS=12 python cli.py demo`）。

---

## 架构（单向无环）

```
miaforge/
├── core/         types · errors(E1xx~E5xx) · config(ENV_MIA_*) · interfaces(Protocol) · seed(确定性)
├── data/         dataset(MiaData: digits/iris 本地确定性 + 离线兜底) · generator(参考用合成 DGP)
├── mia/          target(纯 numpy MLP) · attacks(特征构建 + 攻击者集合) · metrics(ROC-AUC/阈值)
├── pipeline/     pipeline(_protocol 单seed协议 + benchmark 聚合 + 门禁 + 消融 + 失败案例)
├── cli.py        命令行入口
├── examples/     run_demo.py（多 seed + 确定性自检）
├── tests/        pytest（确定性 / 离线兜底 / 攻击契约 / CLI 冒烟）
└── docs/         architecture.md · model_card.md
```

**统一语义**：所有攻击「分数越大越像成员」，跨模块公平评测。
**特征契约**（`build_features` 固定列布局，K = n_classes）：
`0 loss | 1 confidence | 2 entropy | 3 margin | 4 correct | 5..5+K-1 各类概率 p_c`

---

## 方法说明

- **基线 `loss_threshold`**：Yeom et al. 2018「blind」攻击，仅用真类交叉熵 loss，`score = -loss`。这是单特征 ROC 的上限。
- **旗舰 `shadow_all`（Shokri 影子模型攻击）**：
  1. 训练 K 个**影子模型**（与目标任务同架构、同规模），积累「特征 → 成员/非成员」监督；
  2. 用纯 numpy 逻辑回归在影子监督上训练攻击分类器，融合**完整概率向量 + loss/置信/熵/间隔/正确率**；
  3. 将攻击迁移到目标模型的评测集打分。
- **消融**：① `shadow_loss_only`（仅 loss）② `shadow_no_correct`（去 correct）③ 失败案例（≥3 条典型误例 + 归因）。

**域偏移处理**：攻击标准化统计量由「影子 + 目标特征合并分布」统一估计（仅特征缩放，无成员标签泄漏），消除影子/目标特征尺度错位。

---

## 复现与确定性

- `core.seed.set_all(seed)` 一次性设齐全局随机源；各组件以 `seed + offset` 派生独立 Generator。
- 逻辑回归权重零初始化、全批量梯度下降，**无随机初始化**，故完全确定性。
- 真实数据集来自 `sklearn.load_digits`（本地内置，无需联网），固定归一化，结果可逐位复现。

---

## 局限与诚实声明

- 本系统在**小样本、较强容量**设定下运行；目标是演示 MIA 攻击面与评测方法，非针对生产模型的实战工具。
- 经典 SOTA（LiRA, Carlini 2022）用大量影子模型 + 逐类高斯似然比；本系统为**轻量可复现版**，未实现 LiRA 的高阶统计，评测时如实对标而非夸大。
- 数据集为 8×8 数字图像，难度远低于真实大模型场景；AUC 量级仅作方法验证。

---

## 测试与 CI

- `pytest`：确定性、离线兜底、攻击契约、CLI 冒烟（6 项全绿）。
- CI（`.github/workflows/ci.yml`）：Python 3.11/3.12/3.13 矩阵 + ruff + pytest + demo 冒烟 + 密钥扫描。

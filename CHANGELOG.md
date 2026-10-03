# Changelog

## v0.1.1 (2026-10-03)
工程强化（**算法与数值结果零变化**：4 项 AUC 与 delta 逐位一致）

- **修复 v0.1.0 的 lint 声明与实际不符**：v0.1.0 宣称「ruff 绿」，但 CI 写的是 `ruff check . || true`
  （非阻塞）且未钉版本；实装 ruff 0.16.10 后暴露 **17 项告警**，均为真问题。本次全部清零。
- lint 修复明细：`Tuple→tuple` / 去冗余注解引号（UP006, UP035, UP037）、导入排序（I001）、
  `__all__` 排序（RUF022）、失效 `noqa`（RUF100）、冗余 `int()` 包裹（RUF046）、
  列表切片取首元素改 `next(...)`（RUF015）、列表+合并改 `*解包`（RUF005）、
  `if/else` 改三元（SIM108）、手写计数器改 `range(1, n+1)`（SIM113）、
  `subprocess.run` 补显式 `check=False`（PLW1510）、长行折行（E501）。
- **收窄异常捕获（行为 hardening）**：`data/dataset.py` 的离线兜底由裸 `except Exception`
  收窄为 `except ImportError`。其余异常（如数据集名非法）不再被静默吞掉退化兜底，必须可见。
- **新增 `pyproject.toml`**：显式钉死规则集
  （`E,W,F,I,UP,BLE,SIM,PLW,RUF`，忽略中文全角标点误报的 RUF001/002/003）与 `target-version="py311"`。
- **钉死 ruff 版本 0.16.10**（CI / Dockerfile / Makefile），并把 lint 从「建议」升级为「门禁」：
  CI 去掉 `|| true`，新增 `ruff format --check .`；Dockerfile 构建期执行 lint 和格式校验。
- 验证：pytest 6 项全绿；demo 4 项 AUC（0.5920 / 0.6508 / 0.5920 / 0.6533）
  与 `GATE delta=0.0588 PASSED=True` **与 v0.1.0 完全一致**；确定性二次运行逐位一致。

## v0.1.0 (2026-10-03)
- 初始发布：成员推理攻击（MIA）端到端系统
- 数据集：默认 `sklearn.load_digits`（本地确定性），离线兜底纯 numpy 合成
- 目标模型：纯 numpy MLP（确定性、零重型依赖）
- 攻击者：Yeom loss 阈值基线 + Shokri 影子模型攻击（融合完整概率向量 + loss/置信/熵/间隔/正确率）
- 流水线：多 seed 聚合 + 门禁 + 消融 + 失败案例收集
- 门禁：旗舰 AUC − 基线 AUC ≥ 0.02 且显著；默认配置 +0.059 PASS
- 确定性：同 seed 二次运行逐位一致
- 测试：pytest 6 项（确定性 / 离线兜底 / 攻击契约 / CLI 冒烟）全绿
- CI：Python 3.11/3.12/3.13 矩阵 + ruff + pytest + demo 冒烟 + 密钥扫描
- 文档：README / architecture.md / model_card.md

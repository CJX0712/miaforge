# Changelog

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

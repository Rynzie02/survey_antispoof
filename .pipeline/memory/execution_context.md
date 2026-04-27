# Execution Context
_最后同步：2026-04-27T08:46:00Z_

## 当前任务

**ID:** exp-001（待分配正式ID）
**标题:** 两阶段串行保护扰动实验
**状态:** 待启动
**详细说明:**
- 阶段1：PGD优化 δ₁，损失 L₁ = L_clone(x + δ₁)，使说话人embedding偏移
- 阶段2：在 δ₁ 基础上优化 δ₂，损失 L₂ = L_clone(Purify(x + δ₁ + δ₂))，净化器为 AudioPure
- DiffAttack deviated-reconstruction loss：最大化 forward/reverse 扩散中间点差异，破坏净化重建

## 实验配置
- ε = 0.02（L∞约束）
- iterations = 100
- α = 0.6（说话人攻击权重），β = 0.4（净化失效权重）
- 净化器：AudioPure（score-based diffusion，arXiv:2310.14270）
- 数据集：VoxCeleb，100 samples
- 成功标准：ASR > 90%，PPR > 70%

## 决策树
1. 先跑阶段1 baseline → 确认 ASR > 90%
2. 加入阶段2 → 验证净化后 PPR > 70%
3. 若阶段2梯度计算过慢 → 考虑用 DDIM 加速（减少 timestep）

## 上下文积累诊断
- AudioPure 需要 score-based model 权重，安装前确认路径
- DiffAttack loss 需要在扩散中间步（t=T/2附近）计算偏差，非端到端

## 待处理的 Agent 反馈
无

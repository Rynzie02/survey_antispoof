# Project Truth
_最后同步：2026-04-27T08:46:00Z_

## 研究主题
语音对抗攻击：TTS防克隆 + 净化失效

## 研究目标
**防御者视角**：在自己的音频上加入保护扰动 δ，使得：
1. 攻击者无法用 TTS/VC 模型克隆说话人身份
2. 攻击者试图用扩散净化去除扰动后，克隆仍然失效

即：SafeSpeech/AntiFake 的升级版——同时防克隆 + 防净化。

## 当前阶段
ideation_completed → 技术路径已确认，准备进入实验阶段

## 已确认决策
- 2026-04-12: 选择双目标保护扰动作为核心研究方向（Idea 1）
- 2026-04-12: 完成文献调研（28篇），识别5个研究空白
- 2026-04-13: 净化器替换为 AudioPure（score-based diffusion）
- 2026-04-14: 损失权重调整 α=0.6（ASR），β=0.4（PPR）
- 2026-04-27: 确认防御者视角，确认两阶段串行优化路径，DiffAttack deviated-reconstruction loss 作为阶段2核心方法

## 阶段进展摘要

### Survey
完成28篇文献调研，核心论文：
- arXiv:2310.14270 (DAP / AudioPure)
- arXiv:2311.16124 (DiffAttack：deviated-reconstruction loss)
- SafeSpeech USENIX 2025
- AntiFake CCS 2023
- De-AntiFake ICML 2025

### Ideation
生成5个创新方向，选定 Idea 1：**两阶段串行保护扰动生成框架**

**阶段1**：生成防克隆扰动 δ₁
- 目标：使 TTS/VC encoder 无法提取正确 speaker embedding
- 方法：PGD + L_clone_fail（AntiFake/SafeSpeech 路线）
- 损失：L₁ = L_clone(x + δ₁)

**阶段2**：在 δ₁ 基础上加入净化鲁棒性
- 目标：扩散净化后扰动仍然有效
- 方法：模拟 forward+reverse 扩散过程，优化净化后克隆失效损失
- 损失：L₂ = L_clone(Purify(x + δ₁ + δ₂))
- 核心借鉴：DiffAttack deviated-reconstruction loss（让 forward/reverse 中间点差异变大）

**约束**：||δ₁ + δ₂||_∞ ≤ ε

### Experiment
代码框架已实现（experiments/dual_target_attack/），配置：
- ε=0.02, iterations=100, α=0.6, β=0.4
- 净化器：AudioPure
- 数据集：VoxCeleb 100 samples
- 状态：待运行

### Publication
未开始

## 当前最佳实验结果
暂无（实验未运行）

## 方向调整记录
- 2026-04-27: 明确两阶段策略：阶段1仅偏移说话人特征，阶段2同时偏移说话人特征+破坏扩散净化（DiffAttack方法）

## 风险 / 阻塞项
- AudioPure 环境依赖尚未安装完成
- 两阶段联合优化的梯度计算效率待验证

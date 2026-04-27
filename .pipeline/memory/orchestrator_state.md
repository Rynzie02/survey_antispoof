# Orchestrator State
_最后同步：2026-04-27T08:46:00Z_

## 全局进度看板

| 阶段 | 状态 | 完成/总计 | 备注 |
|------|------|----------|------|
| Survey | done | 1/1 | 28篇文献，gap分析完成 |
| Ideation | done | 1/1 | 选定两阶段串行框架 |
| Experiment | active | 0/? | 代码框架就绪，待运行 |
| Publication | pending | 0/? | 未开始 |

## 当前活跃任务
- 实验代码框架已生成（experiments/dual_target_attack/）
- 待安装 AudioPure 依赖并运行首轮实验

## 最近完成任务
- 2026-04-14: 调整损失权重 α=0.6, β=0.4
- 2026-04-13: 替换净化器为 AudioPure，迁移至 uv 环境
- 2026-04-12: 实现实验代码框架
- 2026-04-12: 确定 Idea 1 为核心方向
- 2026-04-12: 完成文献调研（28篇）

## 决策点
- 两阶段策略已确认：阶段1偏移说话人特征，阶段2加入 DiffAttack deviated-reconstruction loss
- 下一步需决定：是否先跑阶段1 baseline，再加阶段2

## 下一步建议
1. 安装 AudioPure 环境依赖
2. 运行阶段1 baseline（纯防克隆，无净化鲁棒性）
3. 加入阶段2损失，对比两阶段效果

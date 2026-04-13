# Experiment initialized: Dual-Target Adversarial Attack

**Date**: 2026-04-12
**Status**: Code implementation complete, ready for execution

## Experiment Configuration

- **Attack Type**: Dual-target PGD
- **Epsilon**: 0.02
- **Iterations**: 100
- **Loss Weights**: α=0.5, β=0.5
- **Dataset**: VoxCeleb (100 samples)
- **Success Criteria**: ASR > 90%, PPR > 70%

## Experiment Runs

### Config Update — 2026-04-13
- **环境管理**: 迁移至 uv，新建 `pyproject.toml`，删除 `requirements.txt`
- **净化器**: 替换为 AudioPure（score-based diffusion，arXiv:2310.14270）
- **接口**: `load_purification_model(model_type='audiopure')` → `AudioPureWrapper`
- **攻击参数**: ε=0.02, iterations=100, α=β=0.5（不变）
- **状态**: 代码已更新，待安装 AudioPure 后运行

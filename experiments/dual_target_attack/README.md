# Dual-Target Adversarial Attack Experiments

This directory contains the implementation of the dual-target adversarial optimization framework for audio adversarial attacks.

## Overview

The framework simultaneously optimizes two objectives:
1. **Speaker Recognition Attack**: Make speaker recognition systems misclassify audio
2. **Purification Robustness**: Ensure adversarial perturbations survive diffusion-based purification

## Project Structure

```
dual_target_attack/
├── models/
│   ├── speaker_model.py      # Speaker recognition model (ResNet-based)
│   └── purification.py        # Diffusion-based purification model
├── attacks/
│   ├── dual_pgd.py           # Dual-target PGD attack
│   └── utils.py              # Attack utility functions
├── data/
│   └── dataset.py            # VoxCeleb dataset loader
├── evaluation/
│   └── metrics.py            # Evaluation metrics (ASR, PPR, SNR, PESQ, STOI)
├── results/
│   ├── logs/                 # Experiment logs
│   ├── checkpoints/          # Model checkpoints
│   └── figures/              # Visualization figures
├── config.py                 # Configuration file
├── main.py                   # Main experiment script
└── requirements.txt          # Python dependencies
```

## Installation

```bash
uv python pin 3.11
uv sync
```

The speaker encoder path now expects the PyPI `coqui-tts` package under Python 3.11.
The bundled `experiments/De-AntiFake/PhonePuRe/encoder_models/TTS` copy is only used as a fallback.

## Usage

### Basic Experiment

Run the main experiment with default configuration:

```bash
uv run python main.py
```

### Custom Configuration

Modify `config.py` or pass parameters:

```python
from config import config

# Update configuration
config.update(
    epsilon=0.02,
    num_iterations=100,
    alpha=0.5,
    beta=0.5
)
```

### Running Different Attack Types

```python
# Single-target attack (baseline)
python main.py --attack_type single

# Dual-target attack
python main.py --attack_type dual

# Adaptive weight attack
python main.py --attack_type adaptive
```

## Key Components

### 1. Dual-Target Loss Function

```
L_total = α · L_speaker + β · L_purification
```

- **L_speaker**: Cross-entropy loss for speaker misclassification
- **L_purification**: Cosine similarity loss to maintain adversarial effect after purification
- **α, β**: Weight parameters (default: 0.5, 0.5)

### 2. Attack Algorithm

```
for i in 1 to N_iter:
    L_total = α · L_speaker(x_adv) + β · L_purification(x_adv)
    grad = ∇_{x_adv} L_total
    x_adv = x_adv + step_size · sign(grad)
    x_adv = clip(x_adv, x_clean - ε, x_clean + ε)
```

### 3. Evaluation Metrics

- **ASR (Attack Success Rate)**: Percentage of successful attacks (target > 90%)
- **PPR (Post-Purification Robustness)**: Percentage of attacks surviving purification (target > 70%)
- **SNR (Signal-to-Noise Ratio)**: Perturbation magnitude
- **PESQ/STOI**: Perceptual audio quality

## Experiment Phases

1. **Phase 1**: Environment setup and model preparation
2. **Phase 2**: Baseline experiments (single-target attack)
3. **Phase 3**: Dual-target attack experiments
4. **Phase 4**: Comparison and analysis

## Expected Results

- **Single-target attack**: High ASR (~95%), Low PPR (~30%)
- **Dual-target attack**: High ASR (>90%), High PPR (>70%)

## Notes

- The current implementation uses dummy data for demonstration
- Replace with actual VoxCeleb dataset for real experiments
- Pretrained models should be placed in `models/pretrained/`
- GPU is recommended for faster experiments

## Citation

If you use this code, please cite:

```
@article{your_paper_2026,
  title={Dual-Target Adversarial Optimization Framework for Audio Attacks},
  author={Your Name},
  journal={arXiv preprint},
  year={2026}
}
```

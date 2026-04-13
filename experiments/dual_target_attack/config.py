"""
Configuration file for dual-target adversarial attack experiments
"""
import torch

class Config:
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    data_root = './data/voxceleb'
    num_samples = 100  # Number of test samples
    sample_rate = 16000
    audio_length = 3.0  # seconds

    # Speaker Recognition Model
    speaker_model_type = 'resnet34'  # or 'ecapa_tdnn', 'xvector'
    speaker_model_path = './models/pretrained/speaker_resnet34.pth'
    num_speakers = 1251  # VoxCeleb1
    embedding_dim = 512

    # Purification Model
    purification_type = 'diffusion'  # or 'denoising_ae'
    diffusion_steps = 50  # T in paper
    noise_schedule = 'linear'  # or 'cosine'
    purification_model_path = './models/pretrained/diffusion_purifier.pth'

    # Attack Parameters
    attack_type = 'dual_pgd'
    epsilon = 0.02  # Perturbation budget (relative to audio amplitude)
    num_iterations = 100
    step_size = None  # Will be set to epsilon / num_iterations * 2

    # Loss weights
    alpha = 0.5  # Weight for speaker recognition loss
    beta = 0.5   # Weight for purification robustness loss
    weight_strategy = 'fixed'  # 'fixed', 'adaptive', or 'staged'

    # Evaluation
    target_asr = 0.90  # Target attack success rate
    target_ppr = 0.70  # Target post-purification robustness

    # Training/Experiment
    batch_size = 8
    num_workers = 4
    seed = 42

    # Logging
    log_dir = './results/logs'
    checkpoint_dir = './results/checkpoints'
    figure_dir = './results/figures'
    log_interval = 10

    def __init__(self):
        if self.step_size is None:
            self.step_size = self.epsilon / self.num_iterations * 2

    def update(self, **kwargs):
        """Update config with keyword arguments"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        if 'epsilon' in kwargs or 'num_iterations' in kwargs:
            self.step_size = self.epsilon / self.num_iterations * 2

# Create default config
config = Config()

"""
Configuration file for dual-target adversarial attack experiments
"""

import torch


class Config:
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    # data_root = "/mnt/data/wht/test_set_example_clean/audio"
    data_root = "/mnt/data/wht/voxceleb1/samples_100"
    num_samples = 100  # Number of test samples
    sample_rate = 16000
    audio_length = 3.0  # seconds

    # Speaker Model (ECAPA-TDNN, local weights)
    speaker_model_type = "ecapa"
    speaker_model_path = "./models/ecapa_tdnn_pretrained"
    embedding_dim = 192

    # Purification Model (De-AntiFake DiffWave, first-stage)
    purification_type = "deantifake"
    purification_model_path = "../De-AntiFake/checkpoints/purification.pkl"
    purification_reverse_timestep = 5
    purification_step_stride = 1  # reverse denoising stride, keep =1 for full quality
    purification_capture_stride = 1  # interval for capturing intermediate nodes in loss
    purification_use_checkpoint = True

    # Attack Parameters
    attack_type = "dual_pgd"
    epsilon = 0.03  # Perturbation budget (relative to audio amplitude)
    num_iterations = 30
    step_size = None  # Will be set to epsilon / num_iterations * 2

    # Loss weights
    alpha = 0.05  # Weight for speaker recognition loss
    beta = 0.95  # Weight for purification robustness loss
    weight_strategy = "fixed"  # 'fixed', 'adaptive', or 'staged'

    # Evaluation
    target_asr = 0.90  # Target attack success rate
    target_ppr = 0.50  # Target post-purification robustness
    ppr_threshold = 0.5  # source_sim below this → attack survived purification

    # Training/Experiment
    batch_size = 1
    use_targeted = (
        False  # True: pull toward target speaker; False: push away from source
    )
    eot_samples = 1  # EOT: number of diffusion samples to average over
    num_workers = 4
    seed = 42

    # Logging
    log_dir = "./results/logs"
    checkpoint_dir = "./results/checkpoints"
    figure_dir = "./results/figures"
    log_interval = 4

    def __init__(self):
        if self.step_size is None:
            self.step_size = self.epsilon / self.num_iterations * 2

    def update(self, **kwargs):
        """Update config with keyword arguments"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        if "epsilon" in kwargs or "num_iterations" in kwargs:
            self.step_size = self.epsilon / self.num_iterations * 2


# Create default config
config = Config()

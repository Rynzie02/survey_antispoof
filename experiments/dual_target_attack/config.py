"""
Configuration file for dual-target adversarial attack experiments
"""

from pathlib import Path

import torch


class Config:
    _base_dir = Path(__file__).resolve().parent

    # Device
    gpu_id = 0
    device = torch.device(
        f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    # Data
    # Set your dataset path here. `run_diffattack.sh` will use this value directly.
    data_root = "/mnt/wht/tts_related/antipurify/voxceleb1/samples_1000"
    num_samples = 32  # Number of test samples
    sample_rate = 16000
    audio_length = 3.0  # seconds

    # Speaker Model for attack optimization: "ecapa" or "tortoise"
    speaker_model_type = "tortoise"
    speaker_model_path = str(_base_dir / "models" / "ecapa_tdnn_pretrained")
    embedding_dim = 1024  # 192 for ecapa, 1024 for tortoise

    # Speaker Model for evaluation metrics
    eval_speaker_model_type = "ecapa"
    eval_speaker_model_path = str(_base_dir / "models" / "ecapa_tdnn_pretrained")
    xvector_model_type = "xvector"
    xvector_model_path = "/home/wht/pyproj/tts_related/asv/spkrec-xvect-voxceleb"

    # Purification Model (De-AntiFake DiffWave, first-stage)
    purification_type = "deantifake"
    purification_model_path = str(
        _base_dir.parent / "De-AntiFake" / "checkpoints" / "purification.pkl"
    )
    purification_reverse_timestep = 5
    purification_step_stride = 1  # reverse denoising stride, keep =1 for full quality
    purification_capture_stride = 1  # interval for capturing intermediate nodes in loss
    purification_use_checkpoint = True

    # Attack Parameters
    attack_type = "dual_pgd"
    epsilon = 0.05  # Perturbation budget (relative to audio amplitude)
    num_iterations = 60
    step_size = None  # Will be set to epsilon / num_iterations * 2

    # Loss weights
    alpha = 0.4  # Weight for speaker recognition loss
    beta = 0.6  # Weight for purification robustness loss
    weight_strategy = "fixed"  # 'fixed', 'adaptive', or 'staged'

    # Evaluation
    target_asr = 0.90  # Target attack success rate
    target_ppr = 0.50  # Target post-purification robustness
    ppr_threshold = 0.5  # source_sim below this → attack survived purification
    ecapa_sva_threshold = 0.75
    xvector_sva_threshold = 0.75
    resemblyzer_sva_threshold = 0.75

    # Training/Experiment
    batch_size = 4
    use_targeted = (
        False  # True: pull toward target speaker; False: push away from source
    )
    eot_samples = 1  # EOT: number of diffusion samples to average over
    num_workers = 0
    seed = 42

    # Logging
    save_audio = False  # set False to skip saving adv/purified wav files
    log_dir = str(_base_dir / "results" / "logs")
    checkpoint_dir = str(_base_dir / "results" / "checkpoints")
    figure_dir = str(_base_dir / "results" / "figures")
    audio_output_dir = str(_base_dir / "results" / "audio")
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

"""
Configuration file for dual-target adversarial attack experiments
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.WARNING)


def _find_tts_related_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if candidate.name == "tts_related":
            return candidate
    return start.parents[2]


def _resolve_path(env_var: str, *candidates: Path) -> str:
    override = os.environ.get(env_var)
    if override:
        return override
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


class Config:
    _base_dir = Path(__file__).resolve().parent
    _tts_related_root = _find_tts_related_root(_base_dir)

    # Device
    gpu_id = 1
    device = torch.device(
        f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    # Data
    # Set your dataset path here. `run_diffattack.sh` will use this value directly.
    # data_root = "/mnt/wht/tts_related/antipurify/voxceleb1/samples_1000"
    # data_root = "/mnt/data/wht/voxceleb1/test_800"
    data_root = "/mnt/wht/exp/test_900"
    num_samples = 64  # Number of test samples
    sample_rate = 16000
    audio_length = 3.0  # seconds 

    # Speaker Model for attack optimization: "ecapa" or "tortoise"
    speaker_model_type = "vits+tortoise+wavlm+campp"
    speaker_model_path = str(_base_dir / "models" / "ecapa_tdnn_pretrained")
    campp_model_path = _resolve_path(
        "DUAL_ATTACK_CAMPP_MODEL_PATH",
        _tts_related_root
        / "E2E-VGuard"
        / "checkpoints"
        / "CosyVoice"
        / "base_models"
        / "CosyVoice-300M"
        / "campplus.onnx",
        _base_dir / "models" / "campp" / "campplus.onnx",
    )
    embedding_dim = 192  # 192 for ecapa, 1024 for tortoise

    # Speaker Model for evaluation metrics
    eval_speaker_model_type = "ecapa"
    eval_speaker_model_path = str(_base_dir / "models" / "ecapa_tdnn_pretrained")
    xvector_model_type = "xvector"
    xvector_model_path = _resolve_path(
        "DUAL_ATTACK_XVECTOR_MODEL_PATH",
        _tts_related_root / "asv" / "spkrec-xvect-voxceleb",
        _base_dir / "models" / "xvector_tdnn_pretrained",
    )

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
    attack_type = "diffattack"
    epsilon = 0.03  # Perturbation budget (relative to audio amplitude)
    num_iterations = 300
    step_size = None  # Will be set to epsilon / num_iterations * 2

    # Loss weights
    alpha = 0.2  # Weight for speaker recognition loss
    beta = 0.8  # Weight for purification robustness loss
    weight_strategy = "staged"  # 'fixed', 'adaptive', or 'staged'
    staged_direct_ratio = 0.65  # staged: first 75% iters direct-only, last 25% add purification

    # Evaluation
    target_asr = 0.90  # Target attack success rate
    target_ppr = 0.50  # Target post-purification robustness
    ppr_threshold = 0.5  # source_sim below this -> attack survived purification
    
    # === CMU-100 tail-20 calibration results ===
    # Raw-cosine EER thresholds:
    # ECAPA:       EER=0.0002  threshold=0.4962
    # XVector:     EER=0.0391  threshold=0.9545
    # Resemblyzer: EER=0.0123  threshold=0.7651
    #
    # ECAPA/x-vector metrics use (cosine + 1) / 2, so keep thresholds on
    # the same score scale as the values being compared.
    ecapa_sva_threshold = 0.7481
    xvector_sva_threshold = 0.9772
    resemblyzer_sva_threshold = 0.7651

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
    audio_output_dir = "/mnt/data/wht/antispoof/audio_fulltrace"
    audio_output_include_run_ts = True
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

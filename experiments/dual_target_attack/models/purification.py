"""
De-AntiFake DiffWave purification wrapper (first-stage model)
Loads checkpoints/purification.pkl via De-AntiFake's create_diffwave_model
"""
import sys
import os
import argparse
import torch
import torch.nn as nn

# Make De-AntiFake importable
_DEANTIFAKE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../De-AntiFake/PhonePuRe')
)
if _DEANTIFAKE not in sys.path:
    sys.path.insert(0, _DEANTIFAKE)


class DeAntiFakePurifier(nn.Module):
    """Wraps De-AntiFake's DiffWave DDPM as a drop-in purifier."""

    def __init__(self, model_path: str, config_path: str,
                 reverse_timestep: int = 25, device: str = 'cuda'):
        super().__init__()
        from purification_models.diffwave_ddpm import create_diffwave_model

        args = argparse.Namespace(
            diffwav_path=model_path,
            ddpm_config=config_path,
            defense='DDPM',
            t=reverse_timestep,
        )
        self.diffwave = create_diffwave_model(
            args, model_path, config_path, reverse_timestep
        )
        self.diffwave.eval().to(device)

    def purify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, samples) waveform in [-1, 1]
        Returns:
            purified waveform, same shape
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)          # (B, 1, T)
        out = self.diffwave(x)
        return out.squeeze(1)           # (B, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.purify(x)


def load_purification_model(model_type='deantifake', model_path=None, device='cuda'):
    config_path = os.path.join(
        os.path.dirname(__file__),
        '../../De-AntiFake/PhonePuRe/configs/config.json'
    )
    if model_path is None:
        raise ValueError("model_path must point to purification.pkl")
    model = DeAntiFakePurifier(model_path, config_path, device=device)
    model.eval().to(device)
    return model

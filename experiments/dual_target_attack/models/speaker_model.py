"""
Coqui YourTTS speaker encoder wrapper
Uses De-AntiFake's ModelManager(coqui) interface
"""
import sys
import os
import torch
import torch.nn as nn

_PHONEPURE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../De-AntiFake/PhonePuRe')
)
if _PHONEPURE not in sys.path:
    sys.path.insert(0, _PHONEPURE)

_RTVC = os.path.join(_PHONEPURE, 'encoder_models', 'rtvc')
if _RTVC not in sys.path:
    sys.path.insert(0, _RTVC)

class CoquiSpeakerEncoder(nn.Module):
    """Wraps Coqui YourTTS speaker_manager as a speaker encoder."""

    def __init__(self, device='cuda'):
        super().__init__()
        from attack_vc_data_utils import load_coqui_model
        self.speaker_manager, _, _, self.device = load_coqui_model()
        self.embedding_dim = 512

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, samples) waveform, 16kHz
        Returns:
            (batch, embedding_dim) L2-normalized speaker embedding
        """
        encoder = self.speaker_manager.encoder

        if x.ndim != 2:
            raise ValueError(f"Expected audio batch with shape (batch, samples), got {tuple(x.shape)}")

        # Coqui's public compute_embedding() runs under inference_mode(), which severs
        # the autograd graph and breaks PGD. Rebuild the encoder forward path here
        # so gradients can flow back to the waveform.
        if getattr(encoder, "use_torch_spec", False):
            with torch.autocast("cuda", enabled=False):
                features = encoder.torch_spec(x)
                features = encoder.instancenorm(features).transpose(1, 2)
        else:
            features = encoder.instancenorm(x).transpose(1, 2)

        embeddings = encoder.layers(features)
        if getattr(encoder, "use_lstm_with_projection", False):
            embeddings = embeddings[:, -1]

        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns embeddings (used as logits proxy for loss computation)."""
        return self.get_embedding(x)


def load_speaker_model(model_path=None, num_speakers=None, device='cuda'):
    model = CoquiSpeakerEncoder(device=device)
    model.eval()
    return model

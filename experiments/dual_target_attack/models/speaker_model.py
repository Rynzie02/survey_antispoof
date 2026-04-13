"""
Coqui YourTTS speaker encoder wrapper
Uses De-AntiFake's ModelManager(coqui) interface
"""
import sys
import os
import torch
import torch.nn as nn

_PHONEPURE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../../De-AntiFake/PhonePuRe')
)
if _PHONEPURE not in sys.path:
    sys.path.insert(0, _PHONEPURE)


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
        embeddings = []
        for i in range(x.shape[0]):
            emb = self.speaker_manager.encoder.compute_embedding(x[i].unsqueeze(0))
            embeddings.append(emb)
        return torch.cat(embeddings, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns embeddings (used as logits proxy for loss computation)."""
        return self.get_embedding(x)


def load_speaker_model(model_path=None, num_speakers=None, device='cuda'):
    model = CoquiSpeakerEncoder(device=device)
    model.eval()
    return model

"""
Coqui YourTTS speaker encoder wrapper
Loads the vendored Coqui TTS package directly to keep gradients intact
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

_PHONEPURE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../De-AntiFake/PhonePuRe")
)
_TTS_ROOT = os.path.join(_PHONEPURE, "encoder_models", "TTS")
if _TTS_ROOT not in sys.path:
    sys.path.insert(0, _TTS_ROOT)

COQUI_YOURTTS_PATH = "tts_models/multilingual/multi-dataset/your_tts"


def load_coqui_speaker_manager(device: str = "cuda"):
    from TTS.api import TTS

    use_gpu = str(device).startswith("cuda") and torch.cuda.is_available()
    runtime_device = "cuda" if use_gpu else "cpu"
    coqui_tts = TTS(model_name=COQUI_YOURTTS_PATH, progress_bar=True, gpu=use_gpu)
    speaker_manager = coqui_tts.synthesizer.tts_model.speaker_manager
    return speaker_manager, runtime_device


class CoquiSpeakerEncoder(nn.Module):
    """Wraps Coqui YourTTS speaker_manager as a speaker encoder."""

    def __init__(self, device="cuda"):
        super().__init__()
        self.speaker_manager, self.device = load_coqui_speaker_manager(device=device)
        self.embedding_dim = 512
        self.speaker_manager.encoder.requires_grad_(False)
        self.speaker_manager.encoder.eval()

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, samples) waveform, 16kHz
        Returns:
            (batch, embedding_dim) L2-normalized speaker embedding
        """
        encoder = self.speaker_manager.encoder

        if x.ndim != 2:
            raise ValueError(
                f"Expected audio batch with shape (batch, samples), got {tuple(x.shape)}"
            )

        # Coqui's public compute_embedding() runs under inference_mode(), which severs
        # the autograd graph and breaks PGD. Rebuild the encoder forward path here
        # so gradients can flow back to the waveform.
        if getattr(encoder, "use_torch_spec", False):
            if x.is_cuda:
                autocast_context = torch.autocast("cuda", enabled=False)
            else:
                autocast_context = torch.autocast("cpu", enabled=False)
            with autocast_context:
                x = encoder.torch_spec(x)

        if getattr(encoder, "log_input", False):
            x = (x + 1e-6).log()

        x = encoder.instancenorm(x).unsqueeze(1)

        x = encoder.conv1(x)
        x = encoder.relu(x)
        x = encoder.bn1(x)
        x = encoder.layer1(x)
        x = encoder.layer2(x)
        x = encoder.layer3(x)
        x = encoder.layer4(x)

        x = x.reshape(x.size(0), -1, x.size(-1))
        w = encoder.attention(x)

        if encoder.encoder_type == "SAP":
            x = torch.sum(x * w, dim=2)
        else:  # ASP
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
            x = torch.cat((mu, sg), 1)

        embeddings = encoder.fc(x.view(x.size(0), -1))
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns embeddings (used as logits proxy for loss computation)."""
        return self.get_embedding(x)


def load_speaker_model(model_path=None, num_speakers=None, device="cuda"):
    model = CoquiSpeakerEncoder(device=device)
    model.eval()
    return model

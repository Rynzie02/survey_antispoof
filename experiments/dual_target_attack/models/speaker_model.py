"""
ECAPA-TDNN speaker encoder wrapper (SpeechBrain, local weights)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

ECAPA_SAVEDIR = os.path.join(os.path.dirname(__file__), "ecapa_tdnn_pretrained")


class ECAPASpeakerEncoder(nn.Module):
    def __init__(self, model_path=None, device="cuda"):
        super().__init__()
        from speechbrain.inference.classifiers import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        model_dir = model_path or ECAPA_SAVEDIR
        hparams_path = os.path.join(model_dir, "hyperparams.yaml")
        source = model_dir if os.path.exists(hparams_path) else "speechbrain/spkrec-ecapa-voxceleb"

        run_opts = {"device": str(device)}
        self._classifier = EncoderClassifier.from_hparams(
            source=source,
            savedir=model_dir,
            # The local checkpoint directory may already contain symlinks
            # (for example to Hugging Face cache files). Re-linking inside the
            # same directory can trip SpeechBrain's self-symlink guard, so for
            # repo-local weights we read the files in place.
            local_strategy=(
                LocalStrategy.NO_LINK
                if source == model_dir
                else LocalStrategy.SYMLINK
            ),
            run_opts=run_opts,
        )
        self._encoder = self._classifier.mods.embedding_model
        self._encoder.requires_grad_(False)
        self._encoder.eval()
        self.embedding_dim = 192

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, samples) waveform, 16kHz
        Returns:
            (batch, 192) L2-normalized speaker embedding
        """
        if x.ndim != 2:
            raise ValueError(f"Expected (batch, samples), got {tuple(x.shape)}")

        # compute fbank features via SpeechBrain's compute_features + mean_var_norm
        feats = self._classifier.mods.compute_features(x)  # (B, T, 80)
        feats = self._classifier.mods.mean_var_norm(
            feats, torch.ones(x.shape[0], device=x.device)
        )

        embeddings = self._encoder(feats)  # (B, 1, 192)
        embeddings = embeddings.squeeze(1)  # (B, 192)
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_embedding(x)


def load_speaker_model(model_path=None, num_speakers=None, device="cuda"):
    model = ECAPASpeakerEncoder(model_path=model_path, device=device)
    model.eval()
    return model

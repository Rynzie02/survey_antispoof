"""
Speaker encoder wrappers for adversarial attacks.
Supports: ECAPA-TDNN / x-vector (SpeechBrain), Tortoise-TTS autoregressive conditioning encoder.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

ECAPA_SAVEDIR = os.path.join(os.path.dirname(__file__), "ecapa_tdnn_pretrained")
XVECTOR_SAVEDIR = os.path.join(os.path.dirname(__file__), "xvector_tdnn_pretrained")


class _SpeechBrainSpeakerEncoder(nn.Module):
    def __init__(self, model_path=None, device="cuda", default_savedir=None,
                 hf_source=None, embedding_dim=None):
        super().__init__()
        from speechbrain.inference.classifiers import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        model_dir = model_path or default_savedir
        os.makedirs(model_dir, exist_ok=True)
        hparams_path = os.path.join(model_dir, "hyperparams.yaml")
        source = model_dir if os.path.exists(hparams_path) else hf_source

        run_opts = {"device": str(device)}
        self._classifier = EncoderClassifier.from_hparams(
            source=source,
            overrides={"pretrained_path": model_dir},
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
        self.embedding_dim = embedding_dim

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


class ECAPASpeakerEncoder(_SpeechBrainSpeakerEncoder):
    def __init__(self, model_path=None, device="cuda"):
        super().__init__(
            model_path=model_path,
            device=device,
            default_savedir=ECAPA_SAVEDIR,
            hf_source="speechbrain/spkrec-ecapa-voxceleb",
            embedding_dim=192,
        )


class XVectorSpeakerEncoder(_SpeechBrainSpeakerEncoder):
    def __init__(self, model_path=None, device="cuda"):
        super().__init__(
            model_path=model_path,
            device=device,
            default_savedir=XVECTOR_SAVEDIR,
            hf_source="speechbrain/spkrec-xvect-voxceleb",
            embedding_dim=512,
        )


class TortoiseSpeakerEncoder(nn.Module):
    """
    Wraps tortoise-tts autoregressive conditioning_encoder as a differentiable speaker encoder.
    Input: (B, T) waveform at 22050 Hz
    Output: (B, 1024) L2-normalized conditioning latent
    """

    def __init__(self, model_path=None, device="cuda", input_sr=16000):
        super().__init__()
        from tortoise.models.arch_util import TorchMelSpectrogram
        from tortoise.models.autoregressive import UnifiedVoice

        ckpt_path = model_path or "/home/wht/pyproj/tts_related/AntiFake/tortoise/autoregressive.pth"
        autoregressive = UnifiedVoice(
            max_mel_tokens=604, max_text_tokens=402, max_conditioning_inputs=2,
            layers=30, model_dim=1024, heads=16, number_text_tokens=255,
            start_text_token=255, checkpointing=False, train_solo_embeddings=False
        )
        state = torch.load(ckpt_path, map_location="cpu")
        autoregressive.load_state_dict(state, strict=False)

        conditioning_encoder = autoregressive.conditioning_encoder
        del autoregressive, state  # free the large model immediately

        self._mel = TorchMelSpectrogram()
        self._conditioning_encoder = conditioning_encoder
        self._conditioning_encoder.requires_grad_(False)
        self._conditioning_encoder.eval()
        self._conditioning_encoder.to(device)
        self._device = device
        self._input_sr = input_sr
        self._target_sr = 22050
        self._cond_length = 132300  # samples at 22050 Hz
        self.embedding_dim = 1024

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T) waveform at 22050 Hz
        Returns:
            (B, 1024) L2-normalized conditioning latent
        """
        if x.ndim != 2:
            raise ValueError(f"Expected (batch, samples), got {tuple(x.shape)}")

        if self._input_sr != self._target_sr:
            x = torchaudio.functional.resample(x, self._input_sr, self._target_sr)

        gap = x.shape[-1] - self._cond_length
        if gap < 0:
            x = F.pad(x, (0, -gap))
        elif gap > 0:
            x = x[:, :self._cond_length]

        mels = []
        for i in range(x.shape[0]):
            m = self._mel(x[i].unsqueeze(0)).squeeze(0)  # (80, T')
            mels.append(m)
        mel = torch.stack(mels, dim=0)  # (B, 80, T')
        latent = self._conditioning_encoder(mel)          # (B, 1024)
        return F.normalize(latent, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_embedding(x)


def load_speaker_model(model_path=None, model_type=None, num_speakers=None, device="cuda", input_sr=16000):
    mtype = model_type or "ecapa"
    if mtype == "tortoise":
        model = TortoiseSpeakerEncoder(device=device, input_sr=input_sr)
    elif mtype == "xvector":
        model = XVectorSpeakerEncoder(model_path=model_path, device=device)
    else:
        model = ECAPASpeakerEncoder(model_path=model_path, device=device)
    model.eval()
    return model

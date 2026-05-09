"""
Speaker encoder wrappers for adversarial attacks.
Supports: ECAPA-TDNN / x-vector (SpeechBrain), Tortoise-TTS autoregressive conditioning encoder,
          VITS posterior encoder, WavLM (microsoft/wavlm-base-plus-sv).
"""

import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi

VITS_DIR = os.path.join(os.path.dirname(__file__), "vits")
VITS_CKPT = os.path.join(VITS_DIR, "pretrained_ljs.pth")
VITS_CONFIG = os.path.join(VITS_DIR, "ljs_base.json")

ECAPA_SAVEDIR = os.path.join(os.path.dirname(__file__), "ecapa_tdnn_pretrained")
XVECTOR_SAVEDIR = os.path.join(os.path.dirname(__file__), "xvector_tdnn_pretrained")

LOGGER = logging.getLogger(__name__)


def _find_tts_related_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if candidate.name == "tts_related":
            return candidate
    return start.parents[4]


def _resolve_tortoise_ckpt(model_path=None) -> str:
    if model_path:
        return model_path

    override = os.environ.get("DUAL_ATTACK_TORTOISE_CKPT")
    if override:
        return override

    models_dir = Path(__file__).resolve().parent
    tts_related_root = _find_tts_related_root(models_dir)
    candidate = tts_related_root / "AntiFake" / "tortoise" / "autoregressive.pth"
    return str(candidate)


def _resolve_campp_onnx(model_path=None) -> str:
    if model_path:
        return str(model_path)

    override = os.environ.get("DUAL_ATTACK_CAMPP_MODEL_PATH")
    if override:
        return override

    models_dir = Path(__file__).resolve().parent
    tts_related_root = _find_tts_related_root(models_dir)
    candidates = [
        tts_related_root
        / "E2E-VGuard"
        / "checkpoints"
        / "CosyVoice"
        / "base_models"
        / "CosyVoice-300M"
        / "campplus.onnx",
        models_dir / "campp" / "campplus.onnx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _patch_onnx2torch_campp():
    # CosyVoice CAM++ uses dynamic ONNX shapes that onnx2torch can rank-infer
    # as -2 for some 1D temporal layers.
    import onnx2torch.node_converters.average_pool as onnx2torch_average_pool
    import onnx2torch.node_converters.batch_norm as onnx2torch_batch_norm

    onnx2torch_average_pool._AVGPOOL_CLASS_FROM_SPATIAL_RANK.setdefault(
        -2, torch.nn.AvgPool1d
    )
    onnx2torch_batch_norm._BN_CLASS_FROM_SPATIAL_RANK.setdefault(
        -2, torch.nn.BatchNorm1d
    )


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


class VITSSpeakerEncoder(nn.Module):
    """
    VITS posterior encoder (enc_q) as a differentiable speaker encoder.
    Input: (B, T) waveform at 16kHz
    Output: (B, 192) L2-normalized mean-pooled latent
    """

    def __init__(self, config_path=None, checkpoint_path=None, device="cuda", input_sr=16000):
        super().__init__()
        _models_dir = os.path.dirname(VITS_DIR)  # models/
        if VITS_DIR not in sys.path:
            sys.path.insert(0, VITS_DIR)
        if _models_dir not in sys.path:
            sys.path.insert(0, _models_dir)
        from text.symbols import symbols
        from vits.models import SynthesizerTrn
        import vits.utils as utils

        hps = utils.get_hparams_from_file(config_path or VITS_CONFIG)
        net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
        ckpt = torch.load(checkpoint_path or VITS_CKPT, map_location="cpu")
        state = ckpt.get("model", ckpt)
        net_g.load_state_dict(state, strict=False)

        self._enc_q = net_g.enc_q
        self._enc_q.requires_grad_(False)
        self._enc_q.eval()
        self._enc_q.to(device)

        self._hps = hps
        self._device = device
        self._input_sr = input_sr
        self._target_sr = hps.data.sampling_rate
        self.embedding_dim = hps.model.inter_channels

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        from vits.mel_processing import spectrogram_torch
        import vits.commons as commons

        if x.ndim != 2:
            raise ValueError(f"Expected (batch, samples), got {tuple(x.shape)}")

        if self._input_sr != self._target_sr:
            x = torchaudio.functional.resample(x, self._input_sr, self._target_sr)

        hps = self._hps.data
        specs, masks = [], []
        for i in range(x.shape[0]):
            spec = spectrogram_torch(
                x[i].unsqueeze(0),
                hps.filter_length, hps.sampling_rate,
                hps.hop_length, hps.win_length,
                center=False,
            )  # (1, freq, T)
            specs.append(spec.squeeze(0))

        max_len = max(s.shape[-1] for s in specs)
        spec_padded = torch.zeros(len(specs), specs[0].shape[0], max_len, device=self._device)
        lengths = torch.zeros(len(specs), dtype=torch.long, device=self._device)
        for i, s in enumerate(specs):
            spec_padded[i, :, :s.shape[-1]] = s
            lengths[i] = s.shape[-1]

        z, _, _, mask = self._enc_q(spec_padded, lengths)  # (B, C, T)
        # mean pool over valid frames
        emb = (z * mask).sum(-1) / mask.sum(-1).clamp(min=1)  # (B, C)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_embedding(x)


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

        ckpt_path = _resolve_tortoise_ckpt(model_path)
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


WAVLM_SAVEDIR = os.path.join(os.path.dirname(__file__), "wavlm_pretrained")


class WavLMSpeakerEncoder(nn.Module):
    """
    WavLM-based speaker encoder using microsoft/wavlm-base-plus-sv.
    Input: (B, T) waveform at 16kHz
    Output: (B, 256) L2-normalized speaker embedding
    """

    def __init__(self, model_path=None, device="cuda", input_sr=16000):
        super().__init__()
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

        source = model_path or WAVLM_SAVEDIR
        if not os.path.isdir(source) or not os.listdir(source):
            source = "microsoft/wavlm-base-plus-sv"

        self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(source)
        self._model = WavLMForXVector.from_pretrained(source)
        self._model.requires_grad_(False)
        self._model.eval()
        self._model.to(device)

        self._device = device
        self._input_sr = input_sr
        self._target_sr = 16000
        self.embedding_dim = 512

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected (batch, samples), got {tuple(x.shape)}")

        if self._input_sr != self._target_sr:
            x = torchaudio.functional.resample(x, self._input_sr, self._target_sr)

        # Run through model directly (keeps gradients for attack)
        outputs = self._model(input_values=x)
        emb = outputs.embeddings  # (B, 256)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_embedding(x)


class CAMPlusSpeakerEncoder(nn.Module):
    """
    CosyVoice CAM++ speaker encoder converted from campplus.onnx.
    Input: (B, T) waveform
    Output: (B, 192) L2-normalized speaker embedding
    """

    def __init__(self, model_path=None, device="cuda", input_sr=16000):
        super().__init__()
        from onnx2torch import convert

        _patch_onnx2torch_campp()

        onnx_path = Path(_resolve_campp_onnx(model_path))
        if not onnx_path.exists():
            raise FileNotFoundError(f"Missing CAM++ ONNX model: {onnx_path}")

        self._model = convert(str(onnx_path))
        self._model.requires_grad_(False)
        self._model.eval()
        self._model.to(device)

        self._device = torch.device(device)
        self._input_sr = input_sr
        self._target_sr = 16000
        self.embedding_dim = 192

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected (batch, samples), got {tuple(x.shape)}")

        if self._input_sr != self._target_sr:
            x = torchaudio.functional.resample(x, self._input_sr, self._target_sr)
        x = x.to(self._device)

        features = []
        for i in range(x.shape[0]):
            feature = kaldi.fbank(
                x[i].unsqueeze(0),
                num_mel_bins=80,
                dither=0,
                sample_frequency=self._target_sr,
            )
            feature = feature - feature.mean(dim=0, keepdim=True)
            features.append(feature)

        max_frames = max(feature.shape[0] for feature in features)
        batch_feature = torch.zeros(
            len(features),
            max_frames,
            80,
            device=x.device,
            dtype=features[0].dtype,
        )
        for i, feature in enumerate(features):
            batch_feature[i, : feature.shape[0]] = feature

        emb = self._model(batch_feature)
        if isinstance(emb, (tuple, list)):
            emb = emb[0]
        if emb.ndim == 3 and emb.shape[1] == 1:
            emb = emb.squeeze(1)
        elif emb.ndim > 2:
            emb = emb.reshape(emb.shape[0], -1)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_embedding(x)


class EnsembleSpeakerEncoder(nn.Module):
    """
    Ensemble of speaker encoders with potentially different embedding dims.
    get_embedding() returns the first encoder's embedding (for API compatibility).
    Use compute_loss() for the actual ensemble loss during attacks.
    """

    def __init__(self, encoders: list):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.embedding_dim = encoders[0].embedding_dim

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoders[0].get_embedding(x)

    def compute_loss(self, emb_clean_list: list, x_adv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb_clean_list: list of pre-computed clean embeddings, one per encoder
            x_adv: (B, T) adversarial waveform
        Returns:
            sum of -cosine_similarity losses across all encoders
        """
        loss = torch.tensor(0.0, device=x_adv.device)
        for enc, emb_clean in zip(self.encoders, emb_clean_list):
            emb_adv = enc.get_embedding(x_adv)
            loss = loss + (-F.cosine_similarity(emb_clean, emb_adv, dim=1).mean())
        return loss

    def get_clean_embeddings(self, x: torch.Tensor) -> list:
        """Returns list of clean embeddings, one per encoder."""
        return [enc.get_embedding(x) for enc in self.encoders]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_embedding(x)


def _build_ecapa_fallback(model_path=None, device="cuda", reason=""):
    fallback_path = (
        model_path if model_path and os.path.isdir(model_path) else ECAPA_SAVEDIR
    )
    LOGGER.warning(
        "Falling back to ECAPA speaker encoder%s",
        f" because {reason}" if reason else "",
    )
    return ECAPASpeakerEncoder(model_path=fallback_path, device=device)


def load_speaker_model(
    model_path=None,
    model_type=None,
    num_speakers=None,
    device="cuda",
    input_sr=16000,
    campp_model_path=None,
):
    mtype = model_type or "ecapa"
    if mtype in (
        "vits+tortoise",
        "vits+tortoise+ecapa",
        "vits+tortoise+ecapa+wavlm",
        "vits+tortoise+wavlm",
        "vits+tortoise+wavlm+ecapa",
        "vits+tortoise+wavlm+campp",
    ):
        encoders = []

        if os.path.exists(VITS_CKPT):
            try:
                encoders.append(VITSSpeakerEncoder(device=device, input_sr=input_sr))
            except (ImportError, ModuleNotFoundError, FileNotFoundError) as exc:
                LOGGER.warning("Skipping VITS speaker encoder: %s", exc)
        else:
            LOGGER.warning(
                "Skipping VITS speaker encoder: missing checkpoint %s",
                VITS_CKPT,
            )

        tortoise_ckpt = _resolve_tortoise_ckpt()
        if os.path.exists(tortoise_ckpt):
            try:
                encoders.append(
                    TortoiseSpeakerEncoder(
                        model_path=tortoise_ckpt,
                        device=device,
                        input_sr=input_sr,
                    )
                )
            except (ImportError, ModuleNotFoundError, FileNotFoundError) as exc:
                LOGGER.warning("Skipping Tortoise speaker encoder: %s", exc)
        else:
            LOGGER.warning(
                "Skipping Tortoise speaker encoder: missing checkpoint %s",
                tortoise_ckpt,
            )

        if mtype in (
            "vits+tortoise+wavlm",
            "vits+tortoise+wavlm+ecapa",
            "vits+tortoise+wavlm+campp",
        ):
            try:
                encoders.append(WavLMSpeakerEncoder(device=device, input_sr=input_sr))
            except Exception as exc:
                LOGGER.warning("Skipping WavLM speaker encoder: %s", exc)

        if mtype == "vits+tortoise+wavlm+ecapa":
            try:
                encoders.append(ECAPASpeakerEncoder(model_path=model_path, device=device))
            except Exception as exc:
                LOGGER.warning("Skipping ECAPA speaker encoder: %s", exc)

        if mtype == "vits+tortoise+wavlm+campp":
            try:
                encoders.append(
                    CAMPlusSpeakerEncoder(
                        model_path=campp_model_path,
                        device=device,
                        input_sr=input_sr,
                    )
                )
            except Exception as exc:
                LOGGER.warning("Skipping CAM++ speaker encoder: %s", exc)

        if mtype == "vits+tortoise+ecapa":
            try:
                encoders.append(ECAPASpeakerEncoder(model_path=model_path, device=device))
            except Exception as exc:
                LOGGER.warning("Skipping ECAPA speaker encoder: %s", exc)

        if mtype == "vits+tortoise+ecapa+wavlm":
            try:
                encoders.append(ECAPASpeakerEncoder(model_path=model_path, device=device))
            except Exception as exc:
                LOGGER.warning("Skipping ECAPA speaker encoder: %s", exc)
            try:
                encoders.append(WavLMSpeakerEncoder(device=device, input_sr=input_sr))
            except Exception as exc:
                LOGGER.warning("Skipping WavLM speaker encoder: %s", exc)

        if len(encoders) >= 2:
            model = EnsembleSpeakerEncoder(encoders)
        elif len(encoders) == 1:
            model = encoders[0]
            LOGGER.warning(
                "Only one encoder is available for '%s'; using %s only.",
                mtype,
                type(model).__name__,
            )
        else:
            model = _build_ecapa_fallback(
                model_path=model_path,
                device=device,
                reason="VITS/Tortoise checkpoints are unavailable",
            )
    elif mtype == "vits":
        try:
            model = VITSSpeakerEncoder(
                checkpoint_path=(
                    model_path if model_path and os.path.isfile(model_path) else None
                ),
                device=device,
                input_sr=input_sr,
            )
        except (ImportError, ModuleNotFoundError, FileNotFoundError) as exc:
            model = _build_ecapa_fallback(
                device=device,
                reason=f"VITS speaker encoder could not be loaded ({exc})",
            )
    elif mtype == "tortoise":
        try:
            model = TortoiseSpeakerEncoder(
                model_path=(
                    model_path if model_path and os.path.isfile(model_path) else None
                ),
                device=device,
                input_sr=input_sr,
            )
        except (ImportError, ModuleNotFoundError, FileNotFoundError) as exc:
            model = _build_ecapa_fallback(
                device=device,
                reason=f"Tortoise speaker encoder could not be loaded ({exc})",
            )
    elif mtype == "wavlm":
        try:
            model = WavLMSpeakerEncoder(
                model_path=model_path if model_path and os.path.isdir(model_path) else None,
                device=device,
                input_sr=input_sr,
            )
        except Exception as exc:
            model = _build_ecapa_fallback(
                device=device,
                reason=f"WavLM speaker encoder could not be loaded ({exc})",
            )
    elif mtype == "campp":
        try:
            model = CAMPlusSpeakerEncoder(
                model_path=campp_model_path,
                device=device,
                input_sr=input_sr,
            )
        except Exception as exc:
            model = _build_ecapa_fallback(
                device=device,
                reason=f"CAM++ speaker encoder could not be loaded ({exc})",
            )
    elif mtype == "xvector":
        model = XVectorSpeakerEncoder(model_path=model_path, device=device)
    else:
        model = ECAPASpeakerEncoder(model_path=model_path, device=device)
    model.eval()
    return model

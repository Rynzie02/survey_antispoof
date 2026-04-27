"""
Evaluation metrics for dual-target attack
"""

import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime

try:
    from pesq import pesq
except ImportError:
    pesq = None

try:
    from pystoi import stoi
except ImportError:
    stoi = None

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
except ImportError:
    VoiceEncoder = None
    preprocess_wav = None


class AttackMetrics:
    """Compute evaluation metrics for adversarial attacks"""

    def __init__(
        self,
        speaker_model,
        purification_model,
        device="cuda",
        sample_rate=16000,
        xvector_model=None,
        ecapa_sva_threshold=0.75,
        xvector_sva_threshold=0.75,
        resemblyzer_sva_threshold=0.75,
    ):
        self.speaker_model = speaker_model
        self.purification_model = purification_model
        self.device = device
        self.sample_rate = sample_rate
        self.xvector_model = xvector_model
        self.ecapa_sva_threshold = ecapa_sva_threshold
        self.xvector_sva_threshold = xvector_sva_threshold
        self.resemblyzer_sva_threshold = resemblyzer_sva_threshold
        self.resemblyzer = (
            VoiceEncoder(device=device, verbose=False) if VoiceEncoder is not None else None
        )

    @staticmethod
    def _sva_from_similarity(similarity, threshold):
        return float(np.mean(np.asarray(similarity) >= threshold))

    def compute_asr(self, x_clean, x_adv, threshold=None):
        """
        Compute Attack Success Rate (ASR): adv_source_sim < threshold (in [0,1])
        """
        if threshold is None:
            threshold = self.ecapa_sva_threshold
        with torch.no_grad():
            clean_embed = self.speaker_model.get_embedding(x_clean)
            adv_embed = self.speaker_model.get_embedding(x_adv)
            source_similarity = (
                F.cosine_similarity(adv_embed, clean_embed, dim=1) + 1
            ) / 2
            asr = (source_similarity < threshold).float().mean().item()
        return asr

    def compute_ppr(self, x_clean, x_adv, ppr_threshold=0.5):
        """
        Untargeted PPR: attack survives purification if purified_source_sim < ppr_threshold (in [0,1])
        """
        with torch.no_grad():
            clean_embed = self.speaker_model.get_embedding(x_clean)
            x_purified = self.purification_model(x_adv)
            purified_embed = self.speaker_model.get_embedding(x_purified)
            source_similarity = (
                F.cosine_similarity(purified_embed, clean_embed, dim=1) + 1
            ) / 2
            ppr = (source_similarity < ppr_threshold).float().mean().item()
        return ppr, source_similarity.mean().item()

    def compute_clean_purified_sim(self, x_clean):
        """Baseline: cos_sim between clean and purified-clean embeddings."""
        with torch.no_grad():
            clean_embed = self.speaker_model.get_embedding(x_clean)
            x_purified = self.purification_model(x_clean)
            purified_embed = self.speaker_model.get_embedding(x_purified)
            return (
                ((F.cosine_similarity(clean_embed, purified_embed, dim=1) + 1) / 2)
                .mean()
                .item()
            )

    def compute_snr(self, x_clean, x_adv):
        """
        Compute Signal-to-Noise Ratio (SNR)

        Args:
            x_clean: Clean audio (batch, samples)
            x_adv: Adversarial audio (batch, samples)
        Returns:
            SNR in dB
        """
        with torch.no_grad():
            signal_power = torch.mean(x_clean**2, dim=1)
            noise_power = torch.mean((x_adv - x_clean) ** 2, dim=1)
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
            return snr.mean().item()

    def compute_pesq(self, x_clean, x_adv, sample_rate=16000):
        """
        Compute PESQ (Perceptual Evaluation of Speech Quality)

        Args:
            x_clean: Clean audio (batch, samples)
            x_adv: Adversarial audio (batch, samples)
            sample_rate: Audio sample rate
        Returns:
            Average PESQ score
        """
        x_clean_np = x_clean.detach().cpu().numpy()
        x_adv_np = x_adv.detach().cpu().numpy()

        if pesq is None:
            return 0.0

        pesq_scores = []
        for i in range(len(x_clean_np)):
            try:
                score = pesq(sample_rate, x_clean_np[i], x_adv_np[i], "wb")
                pesq_scores.append(score)
            except:
                pass  # Skip if PESQ computation fails

        return np.mean(pesq_scores) if pesq_scores else 0.0

    def compute_stoi(self, x_clean, x_adv, sample_rate=16000):
        """
        Compute STOI (Short-Time Objective Intelligibility)

        Args:
            x_clean: Clean audio (batch, samples)
            x_adv: Adversarial audio (batch, samples)
            sample_rate: Audio sample rate
        Returns:
            Average STOI score
        """
        x_clean_np = x_clean.detach().cpu().numpy()
        x_adv_np = x_adv.detach().cpu().numpy()

        if stoi is None:
            return 0.0

        stoi_scores = []
        for i in range(len(x_clean_np)):
            try:
                score = stoi(x_clean_np[i], x_adv_np[i], sample_rate, extended=False)
                stoi_scores.append(score)
            except:
                pass

        return np.mean(stoi_scores) if stoi_scores else 0.0

    def _resemblyzer_embedding(self, waveform):
        if self.resemblyzer is None or preprocess_wav is None:
            return None

        wav = waveform.detach().cpu().numpy().astype(np.float32)
        wav = preprocess_wav(wav, source_sr=self.sample_rate)
        if wav.size == 0:
            return None
        return self.resemblyzer.embed_utterance(wav)

    def compute_resemblyzer_metrics(self, x_clean, x_adv):
        if self.resemblyzer is None:
            return {
                "resemblyzer_adv_source_sim": 0.0,
                "resemblyzer_purified_source_sim": 0.0,
                "resemblyzer_clean_purified_sim": 0.0,
                "resemblyzer_sva": 0.0,
            }

        with torch.no_grad():
            x_purified = self.purification_model(x_adv)
            x_clean_purified = self.purification_model(x_clean)

        adv_source_sims = []
        purified_source_sims = []
        clean_purified_sims = []

        for i in range(x_clean.shape[0]):
            clean_embed = self._resemblyzer_embedding(x_clean[i])
            adv_embed = self._resemblyzer_embedding(x_adv[i])
            purified_embed = self._resemblyzer_embedding(x_purified[i])
            clean_purified_embed = self._resemblyzer_embedding(x_clean_purified[i])

            if (
                clean_embed is None
                or adv_embed is None
                or purified_embed is None
                or clean_purified_embed is None
            ):
                continue

            adv_source_sims.append(float(np.dot(clean_embed, adv_embed)))
            purified_source_sims.append(float(np.dot(clean_embed, purified_embed)))
            clean_purified_sims.append(float(np.dot(clean_embed, clean_purified_embed)))

        if not adv_source_sims:
            return {
                "resemblyzer_adv_source_sim": 0.0,
                "resemblyzer_purified_source_sim": 0.0,
                "resemblyzer_clean_purified_sim": 0.0,
                "resemblyzer_sva": 0.0,
            }

        return {
            "resemblyzer_adv_source_sim": float(np.mean(adv_source_sims)),
            "resemblyzer_purified_source_sim": float(np.mean(purified_source_sims)),
            "resemblyzer_clean_purified_sim": float(np.mean(clean_purified_sims)),
            "resemblyzer_sva": self._sva_from_similarity(
                adv_source_sims, self.resemblyzer_sva_threshold
            ),
        }

    def compute_xvector_metrics(self, x_clean, x_adv):
        if self.xvector_model is None:
            return {
                "xvector_adv_source_sim": 0.0,
                "xvector_purified_source_sim": 0.0,
                "xvector_clean_purified_sim": 0.0,
                "xvector_sva": 0.0,
            }

        with torch.no_grad():
            clean_embed = self.xvector_model.get_embedding(x_clean)
            adv_embed = self.xvector_model.get_embedding(x_adv)
            x_purified = self.purification_model(x_adv)
            purified_embed = self.xvector_model.get_embedding(x_purified)
            x_clean_purified = self.purification_model(x_clean)
            clean_purified_embed = self.xvector_model.get_embedding(x_clean_purified)

        adv_source_sim = (
            (F.cosine_similarity(adv_embed, clean_embed, dim=1) + 1) / 2
        ).detach().cpu().numpy()
        purified_source_sim = (
            (F.cosine_similarity(purified_embed, clean_embed, dim=1) + 1) / 2
        ).detach().cpu().numpy()
        clean_purified_sim = (
            (F.cosine_similarity(clean_purified_embed, clean_embed, dim=1) + 1) / 2
        ).detach().cpu().numpy()

        return {
            "xvector_adv_source_sim": float(np.mean(adv_source_sim)),
            "xvector_purified_source_sim": float(np.mean(purified_source_sim)),
            "xvector_clean_purified_sim": float(np.mean(clean_purified_sim)),
            "xvector_sva": self._sva_from_similarity(
                adv_source_sim, self.xvector_sva_threshold
            ),
        }

    def compute_all_metrics(
        self, x_clean, x_adv, target_embed, sample_rate=16000, ppr_threshold=0.5
    ):
        """
        Compute all evaluation metrics

        Returns:
            Dictionary of metrics
        """
        ppr, purified_source_sim = self.compute_ppr(x_clean, x_adv, ppr_threshold)
        with torch.no_grad():
            adv_source_sim_vec = (
                (
                    F.cosine_similarity(
                        self.speaker_model.get_embedding(x_adv),
                        self.speaker_model.get_embedding(x_clean),
                        dim=1,
                    )
                    + 1
                )
                / 2
            ).detach().cpu().numpy()
        metrics = {
            "asr": self.compute_asr(x_clean, x_adv, threshold=self.ecapa_sva_threshold),
            "adv_source_sim": float(np.mean(adv_source_sim_vec)),
            "ppr": ppr,
            "purified_source_sim": purified_source_sim,
            "clean_purified_sim": self.compute_clean_purified_sim(x_clean),
            "ecapa_sva": self._sva_from_similarity(
                adv_source_sim_vec, self.ecapa_sva_threshold
            ),
            "snr": self.compute_snr(x_clean, x_adv),
            "pesq": self.compute_pesq(x_clean, x_adv, sample_rate),
            "stoi": self.compute_stoi(x_clean, x_adv, sample_rate),
        }
        metrics.update(self.compute_xvector_metrics(x_clean, x_adv))
        metrics.update(self.compute_resemblyzer_metrics(x_clean, x_adv))

        return metrics

    def print_metrics(self, metrics):
        """Print metrics in a formatted way"""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("\n" + "=" * 50)
        print(f"Evaluation Metrics  [{ts}]")
        print("=" * 50)
        print(f"Attack Success Rate (ASR):           {metrics['asr']:.2%}")
        print(f"Adv Source Similarity:               {metrics['adv_source_sim']:.4f}")
        print(f"Post-Purification Robustness (PPR):  {metrics['ppr']:.2%}")
        print(
            f"Purified Embed Sim (→ source):       {metrics['purified_source_sim']:.4f}"
        )
        print(
            f"Clean Purified Sim:                  {metrics['clean_purified_sim']:.4f}"
        )
        print(f"ECAPA SVA:                           {metrics['ecapa_sva']:.2%}")
        print(
            f"Resemblyzer Adv Sim:                 {metrics['resemblyzer_adv_source_sim']:.4f}"
        )
        print(
            f"Resemblyzer Purified Sim:            {metrics['resemblyzer_purified_source_sim']:.4f}"
        )
        print(
            f"Resemblyzer Clean Purified Sim:      {metrics['resemblyzer_clean_purified_sim']:.4f}"
        )
        print(
            f"Resemblyzer SVA:                     {metrics['resemblyzer_sva']:.2%}"
        )
        print(
            f"XVector Adv Sim:                     {metrics['xvector_adv_source_sim']:.4f}"
        )
        print(
            f"XVector Purified Sim:                {metrics['xvector_purified_source_sim']:.4f}"
        )
        print(
            f"XVector Clean Purified Sim:          {metrics['xvector_clean_purified_sim']:.4f}"
        )
        print(f"XVector SVA:                         {metrics['xvector_sva']:.2%}")
        print(f"Signal-to-Noise Ratio (SNR):         {metrics['snr']:.2f} dB")
        print(f"PESQ Score:                          {metrics['pesq']:.3f}")
        print(f"STOI Score:                          {metrics['stoi']:.3f}")
        print("=" * 50)

        print("\nSuccess Criteria:")
        print(f"ASR > 90%:  {'✓ PASS' if metrics['asr'] > 0.90 else '✗ FAIL'}")
        print(f"PPR > 70%:  {'✓ PASS' if metrics['ppr'] > 0.70 else '✗ FAIL'}")
        print("=" * 50 + "\n")

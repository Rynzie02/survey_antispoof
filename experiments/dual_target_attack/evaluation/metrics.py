"""
Evaluation metrics for dual-target attack
"""

import torch
import torch.nn.functional as F
import numpy as np
from pesq import pesq
from pystoi import stoi
from datetime import datetime


class AttackMetrics:
    """Compute evaluation metrics for adversarial attacks"""

    def __init__(self, speaker_model, purification_model, device="cuda"):
        self.speaker_model = speaker_model
        self.purification_model = purification_model
        self.device = device

    def compute_asr(self, x_clean, x_adv, target_embed):
        """
        Compute Attack Success Rate (ASR)

        Args:
            x_clean: Clean audio (batch, samples)
            x_adv: Adversarial audio (batch, samples)
            target_embed: Target speaker embeddings (batch, dim)
        Returns:
            ASR: Percentage of successful attacks
        """
        with torch.no_grad():
            clean_embed = self.speaker_model.get_embedding(x_clean)
            adv_embed = self.speaker_model.get_embedding(x_adv)
            target_similarity = F.cosine_similarity(adv_embed, target_embed, dim=1)
            source_similarity = F.cosine_similarity(adv_embed, clean_embed, dim=1)
            success = (target_similarity > source_similarity).float()
            asr = success.mean().item()

        return asr

    def compute_ppr(self, x_clean, x_adv, target_embed, ppr_threshold=0.5):
        """
        Untargeted PPR: attack survives purification if purified_source_sim < ppr_threshold
        """
        with torch.no_grad():
            clean_embed = self.speaker_model.get_embedding(x_clean)
            x_purified = self.purification_model(x_adv)
            purified_embed = self.speaker_model.get_embedding(x_purified)
            target_similarity = F.cosine_similarity(purified_embed, target_embed, dim=1)
            source_similarity = F.cosine_similarity(purified_embed, clean_embed, dim=1)
            success = (source_similarity < ppr_threshold).float()
            ppr = success.mean().item()

        return ppr, target_similarity.mean().item(), source_similarity.mean().item()

    def compute_clean_purified_sim(self, x_clean):
        """Baseline: cos_sim between clean and purified-clean embeddings."""
        with torch.no_grad():
            clean_embed = self.speaker_model.get_embedding(x_clean)
            x_purified = self.purification_model(x_clean)
            purified_embed = self.speaker_model.get_embedding(x_purified)
            return F.cosine_similarity(clean_embed, purified_embed, dim=1).mean().item()

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

        stoi_scores = []
        for i in range(len(x_clean_np)):
            try:
                score = stoi(x_clean_np[i], x_adv_np[i], sample_rate, extended=False)
                stoi_scores.append(score)
            except:
                pass

        return np.mean(stoi_scores) if stoi_scores else 0.0

    def compute_all_metrics(
        self, x_clean, x_adv, target_embed, sample_rate=16000, ppr_threshold=0.5
    ):
        """
        Compute all evaluation metrics

        Returns:
            Dictionary of metrics
        """
        ppr, purified_target_sim, purified_source_sim = self.compute_ppr(
            x_clean, x_adv, target_embed, ppr_threshold
        )
        with torch.no_grad():
            adv_source_sim = (
                F.cosine_similarity(
                    self.speaker_model.get_embedding(x_adv),
                    self.speaker_model.get_embedding(x_clean),
                    dim=1,
                )
                .mean()
                .item()
            )
        metrics = {
            "asr": self.compute_asr(x_clean, x_adv, target_embed),
            "adv_source_sim": adv_source_sim,
            "ppr": ppr,
            "purified_target_sim": purified_target_sim,
            "purified_source_sim": purified_source_sim,
            "clean_purified_sim": self.compute_clean_purified_sim(x_clean),
            "snr": self.compute_snr(x_clean, x_adv),
            "pesq": self.compute_pesq(x_clean, x_adv, sample_rate),
            "stoi": self.compute_stoi(x_clean, x_adv, sample_rate),
        }

        return metrics

    def print_metrics(self, metrics):
        """Print metrics in a formatted way"""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("\n" + "=" * 50)
        print(f"Evaluation Metrics  [{ts}]")
        print("=" * 50)
        print(f"Attack Success Rate (ASR):           {metrics['asr']:.2%}")
        print(f"Post-Purification Robustness (PPR):  {metrics['ppr']:.2%}")
        print(
            f"Purified Embed Sim (→ target):       {metrics['purified_target_sim']:.4f}"
        )
        print(
            f"Purified Embed Sim (→ source):       {metrics['purified_source_sim']:.4f}"
        )
        print(f"Signal-to-Noise Ratio (SNR):         {metrics['snr']:.2f} dB")
        print(f"PESQ Score:                          {metrics['pesq']:.3f}")
        print(f"STOI Score:                          {metrics['stoi']:.3f}")
        print("=" * 50)

        # Check success criteria
        print("\nSuccess Criteria:")
        print(f"ASR > 90%:  {'✓ PASS' if metrics['asr'] > 0.90 else '✗ FAIL'}")
        print(f"PPR > 70%:  {'✓ PASS' if metrics['ppr'] > 0.70 else '✗ FAIL'}")
        print("=" * 50 + "\n")

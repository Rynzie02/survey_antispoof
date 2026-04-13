"""
Evaluation metrics for dual-target attack
"""
import torch
import torch.nn.functional as F
import numpy as np
from pesq import pesq
from pystoi import stoi

class AttackMetrics:
    """Compute evaluation metrics for adversarial attacks"""

    def __init__(self, speaker_model, purification_model, device='cuda'):
        self.speaker_model = speaker_model
        self.purification_model = purification_model
        self.device = device

    def compute_asr(self, x_adv, target_labels):
        """
        Compute Attack Success Rate (ASR)

        Args:
            x_adv: Adversarial audio (batch, samples)
            target_labels: Target speaker labels (batch,)
        Returns:
            ASR: Percentage of successful attacks
        """
        with torch.no_grad():
            logits = self.speaker_model(x_adv)
            predictions = torch.argmax(logits, dim=1)
            success = (predictions == target_labels).float()
            asr = success.mean().item()

        return asr

    def compute_ppr(self, x_adv, target_labels):
        """
        Compute Post-Purification Robustness (PPR)

        Measures how many adversarial examples remain effective after purification

        Args:
            x_adv: Adversarial audio (batch, samples)
            target_labels: Target speaker labels (batch,)
        Returns:
            PPR: Percentage of attacks that survive purification
        """
        with torch.no_grad():
            # Purify adversarial audio
            x_purified = self.purification_model(x_adv)

            # Check if still adversarial after purification
            logits = self.speaker_model(x_purified)
            predictions = torch.argmax(logits, dim=1)
            success = (predictions == target_labels).float()
            ppr = success.mean().item()

        return ppr

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
            signal_power = torch.mean(x_clean ** 2, dim=1)
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
        x_clean_np = x_clean.cpu().numpy()
        x_adv_np = x_adv.cpu().numpy()

        pesq_scores = []
        for i in range(len(x_clean_np)):
            try:
                score = pesq(sample_rate, x_clean_np[i], x_adv_np[i], 'wb')
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
        x_clean_np = x_clean.cpu().numpy()
        x_adv_np = x_adv.cpu().numpy()

        stoi_scores = []
        for i in range(len(x_clean_np)):
            try:
                score = stoi(x_clean_np[i], x_adv_np[i], sample_rate, extended=False)
                stoi_scores.append(score)
            except:
                pass

        return np.mean(stoi_scores) if stoi_scores else 0.0

    def compute_all_metrics(self, x_clean, x_adv, target_labels, sample_rate=16000):
        """
        Compute all evaluation metrics

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'asr': self.compute_asr(x_adv, target_labels),
            'ppr': self.compute_ppr(x_adv, target_labels),
            'snr': self.compute_snr(x_clean, x_adv),
            'pesq': self.compute_pesq(x_clean, x_adv, sample_rate),
            'stoi': self.compute_stoi(x_clean, x_adv, sample_rate)
        }

        return metrics

    def print_metrics(self, metrics):
        """Print metrics in a formatted way"""
        print("\n" + "="*50)
        print("Evaluation Metrics")
        print("="*50)
        print(f"Attack Success Rate (ASR):           {metrics['asr']:.2%}")
        print(f"Post-Purification Robustness (PPR):  {metrics['ppr']:.2%}")
        print(f"Signal-to-Noise Ratio (SNR):         {metrics['snr']:.2f} dB")
        print(f"PESQ Score:                          {metrics['pesq']:.3f}")
        print(f"STOI Score:                          {metrics['stoi']:.3f}")
        print("="*50)

        # Check success criteria
        print("\nSuccess Criteria:")
        print(f"ASR > 90%:  {'✓ PASS' if metrics['asr'] > 0.90 else '✗ FAIL'}")
        print(f"PPR > 70%:  {'✓ PASS' if metrics['ppr'] > 0.70 else '✗ FAIL'}")
        print("="*50 + "\n")

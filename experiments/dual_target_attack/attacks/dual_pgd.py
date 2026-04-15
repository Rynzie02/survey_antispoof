"""
Dual-target PGD attack implementation
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm


class DualTargetPGD:
    """
    Dual-Target Projected Gradient Descent Attack

    Simultaneously optimizes two objectives:
    1. Speaker recognition attack (L_speaker)
    2. Purification robustness (L_purification)
    """

    def __init__(self, speaker_model, purification_model, config):
        self.speaker_model = speaker_model
        self.purification_model = purification_model
        self.config = config
        self.speaker_model.requires_grad_(False)
        self.purification_model.requires_grad_(False)

        self.epsilon = config.epsilon
        self.num_iterations = config.num_iterations
        self.step_size = config.step_size
        self.alpha = config.alpha
        self.beta = config.beta
        self.device = config.device

    def compute_speaker_loss(self, x_adv, target_embed):
        """
        Compute speaker cloning attack loss (embedding distance).

        Args:
            x_adv: Adversarial audio (batch, samples)
            target_embed: Target speaker embedding (batch, dim) — precomputed
        Returns:
            Loss value (cosine similarity; maximizing pulls adv toward target)
        """
        embed_adv = self.speaker_model.get_embedding(x_adv)
        return F.cosine_similarity(embed_adv, target_embed, dim=1).mean()

    def compute_purification_loss(self, x_adv, x_clean):
        """
        Purification robustness loss.
        Runs DiffWave in the forward pass with gradient enabled,
        so the perturbation is optimized to survive purification.
        """
        # Keep the pre-purification embedding as a fixed anchor so the second
        # loss only backpropagates through the purifier branch instead of
        # materializing two full speaker-encoder graphs.
        with torch.no_grad():
            embed_adv_before = self.speaker_model.get_embedding(x_adv)

        # Purify adversarial audio
        x_purified = self.purification_model(x_adv)
        embed_adv_after = self.speaker_model.get_embedding(x_purified)
        return F.cosine_similarity(embed_adv_before, embed_adv_after, dim=1).mean()

    def attack(self, x_clean, target_embed, return_trajectory=False):
        """
        Execute dual-target PGD attack

        Args:
            x_clean: Clean audio (batch, samples)
            target_embed: Target speaker embedding (batch, dim)
            return_trajectory: If True, return loss trajectory
        Returns:
            x_adv: Adversarial audio
            trajectory: (optional) Dictionary of loss values over iterations
        """
        x_adv = x_clean.clone().detach().requires_grad_(True)

        trajectory = (
            {"total_loss": [], "speaker_loss": [], "purification_loss": []}
            if return_trajectory
            else None
        )

        for i in range(self.num_iterations):
            # Zero gradients
            if x_adv.grad is not None:
                x_adv.grad.zero_()

            # Compute dual-target loss
            loss_speaker = self.compute_speaker_loss(x_adv, target_embed)
            loss_purification = self.compute_purification_loss(x_adv, x_clean)

            # Combined loss
            loss_total = self.alpha * loss_speaker + self.beta * loss_purification

            # Backward
            loss_total.backward()

            # Update adversarial example
            with torch.no_grad():
                # Gradient ascent (we're maximizing the loss)
                grad_sign = x_adv.grad.sign()
                x_adv = x_adv + self.step_size * grad_sign

                # Project back to epsilon ball
                perturbation = torch.clamp(x_adv - x_clean, -self.epsilon, self.epsilon)
                x_adv = x_clean + perturbation

                # Clamp to valid audio range [-1, 1]
                x_adv = torch.clamp(x_adv, -1.0, 1.0)

            x_adv = x_adv.detach().requires_grad_(True)

            # Log trajectory
            if return_trajectory and i % 10 == 0:
                trajectory["total_loss"].append(loss_total.item())
                trajectory["speaker_loss"].append(loss_speaker.item())
                trajectory["purification_loss"].append(loss_purification.item())

        return x_adv.detach(), trajectory


class SingleTargetPGD:
    """Baseline: Single-target PGD (only attacks speaker recognition)"""

    def __init__(self, speaker_model, config):
        self.speaker_model = speaker_model
        self.config = config

        self.epsilon = config.epsilon
        self.num_iterations = config.num_iterations
        self.step_size = config.step_size
        self.device = config.device

    def attack(self, x_clean, target_embed):
        """Execute single-target PGD attack"""
        x_adv = x_clean.clone().detach().requires_grad_(True)

        for i in range(self.num_iterations):
            if x_adv.grad is not None:
                x_adv.grad.zero_()

            embed_adv = self.speaker_model.get_embedding(x_adv)
            loss = F.cosine_similarity(embed_adv, target_embed, dim=1).mean()

            loss.backward()

            with torch.no_grad():
                grad_sign = x_adv.grad.sign()
                x_adv = x_adv + self.step_size * grad_sign

                perturbation = torch.clamp(x_adv - x_clean, -self.epsilon, self.epsilon)
                x_adv = x_clean + perturbation
                x_adv = torch.clamp(x_adv, -1.0, 1.0)

            x_adv = x_adv.detach().requires_grad_(True)

        return x_adv.detach()


class AdaptiveWeightPGD(DualTargetPGD):
    """Dual-target PGD with adaptive weight adjustment"""

    def attack(self, x_clean, target_embed, return_trajectory=False):
        """Execute attack with adaptive weights"""
        x_adv = x_clean.clone().detach().requires_grad_(True)

        trajectory = (
            {
                "total_loss": [],
                "speaker_loss": [],
                "purification_loss": [],
                "alpha": [],
                "beta": [],
            }
            if return_trajectory
            else None
        )

        for i in range(self.num_iterations):
            if x_adv.grad is not None:
                x_adv.grad.zero_()

            # Compute losses
            loss_speaker = self.compute_speaker_loss(x_adv, target_embed)
            loss_purification = self.compute_purification_loss(x_adv, x_clean)

            # Adaptive weight adjustment based on loss magnitudes
            with torch.no_grad():
                loss_speaker_mag = abs(loss_speaker.item())
                loss_purif_mag = abs(loss_purification.item())
                total_mag = loss_speaker_mag + loss_purif_mag + 1e-8

                # Normalize weights inversely proportional to loss magnitude
                # (give more weight to the objective that's lagging)
                alpha_adaptive = loss_purif_mag / total_mag
                beta_adaptive = loss_speaker_mag / total_mag

            # Combined loss with adaptive weights
            loss_total = (
                alpha_adaptive * loss_speaker + beta_adaptive * loss_purification
            )

            loss_total.backward()

            with torch.no_grad():
                grad_sign = x_adv.grad.sign()
                x_adv = x_adv + self.step_size * grad_sign

                perturbation = torch.clamp(x_adv - x_clean, -self.epsilon, self.epsilon)
                x_adv = x_clean + perturbation
                x_adv = torch.clamp(x_adv, -1.0, 1.0)

            x_adv = x_adv.detach().requires_grad_(True)

            if return_trajectory and i % 10 == 0:
                trajectory["total_loss"].append(loss_total.item())
                trajectory["speaker_loss"].append(loss_speaker.item())
                trajectory["purification_loss"].append(loss_purification.item())
                trajectory["alpha"].append(alpha_adaptive)
                trajectory["beta"].append(beta_adaptive)

        return x_adv.detach(), trajectory

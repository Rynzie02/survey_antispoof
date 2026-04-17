"""
Dual-target PGD attack implementation
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import time


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

    def compute_speaker_loss(self, x_adv, target_embed, source_embed=None):
        """
        Targeted: maximize cos(adv, target)
        Untargeted: minimize cos(adv, source)  i.e. -cos(adv, source)
        """
        embed_adv = self.speaker_model.get_embedding(x_adv)
        if source_embed is not None:
            # untargeted: push away from source
            return -F.cosine_similarity(embed_adv, source_embed, dim=1).mean()
        return F.cosine_similarity(embed_adv, target_embed, dim=1).mean()

    def compute_purification_loss(self, x_adv, x_clean):
        """
        Maximize embedding divergence between forward x_t and reverse x_t
        at the same timesteps, causing the denoising path to deviate.
        """
        if x_adv.ndim == 2:
            x_adv_3d = x_adv.unsqueeze(1)
        else:
            x_adv_3d = x_adv

        # Capture forward intermediates at evenly spaced timesteps
        T = self.purification_model.diffwave.reverse_timestep
        capture_stride = getattr(self.config, "purification_capture_stride", 5)
        capture_steps = list(range(capture_stride, T, capture_stride))
        forward_intermediates, _ = (
            self.purification_model.diffwave._diffusion_intermediates(
                x_adv_3d, capture_steps
            )
        )

        # Run full reverse, get paired (forward_x_t, reverse_x_t)
        x_noised = forward_intermediates[T - 1]
        _, pairs = self.purification_model.diffwave._reverse_with_intermediates(
            x_noised, forward_intermediates
        )

        # Loss: maximize embedding divergence at each paired timestep
        loss = torch.tensor(0.0, device=x_adv.device)
        for x_fwd, x_rev in pairs:
            embed_fwd = self.speaker_model.get_embedding(x_fwd.squeeze(1))
            embed_rev = self.speaker_model.get_embedding(x_rev.squeeze(1))
            loss = loss + (1 - F.cosine_similarity(embed_fwd, embed_rev, dim=1).mean())

        return loss / max(len(pairs), 1)

    def attack(self, x_clean, target_embed, return_trajectory=False):
        x_adv = x_clean.clone().detach().requires_grad_(True)

        trajectory = (
            {"total_loss": [], "speaker_loss": [], "purification_loss": []}
            if return_trajectory
            else None
        )

        t_speaker = 0.0
        t_purif = 0.0
        t_pgd = 0.0

        # precompute source embed for untargeted mode
        use_targeted = getattr(self.config, "use_targeted", True)
        with torch.no_grad():
            source_embed = (
                self.speaker_model.get_embedding(x_clean) if not use_targeted else None
            )

        for i in range(self.num_iterations):
            if x_adv.grad is not None:
                x_adv.grad.zero_()

            t0 = time.time()
            loss_speaker = self.compute_speaker_loss(x_adv, target_embed, source_embed)
            torch.cuda.synchronize() if x_adv.is_cuda else None
            t_speaker += time.time() - t0

            t0 = time.time()
            eot_samples = getattr(self.config, "eot_samples", 1)
            loss_purification = (
                sum(
                    self.compute_purification_loss(x_adv, x_clean)
                    for _ in range(eot_samples)
                )
                / eot_samples
            )
            torch.cuda.synchronize() if x_adv.is_cuda else None
            t_purif += time.time() - t0

            loss_total = self.alpha * loss_speaker + self.beta * loss_purification

            t0 = time.time()
            loss_total.backward()
            with torch.no_grad():
                grad_sign = x_adv.grad.sign()
                x_adv = x_adv + self.step_size * grad_sign
                perturbation = torch.clamp(x_adv - x_clean, -self.epsilon, self.epsilon)
                x_adv = x_clean + perturbation
                x_adv = torch.clamp(x_adv, -1.0, 1.0)
            torch.cuda.synchronize() if x_adv.is_cuda else None
            t_pgd += time.time() - t0

            x_adv = x_adv.detach().requires_grad_(True)

            print(
                f"  iter {i:02d}: total={loss_total.item():.4f} speaker={loss_speaker.item():.4f} purif={loss_purification.item():.4f}"
            )

            if return_trajectory and i % 10 == 0:
                trajectory["total_loss"].append(loss_total.item())
                trajectory["speaker_loss"].append(loss_speaker.item())
                trajectory["purification_loss"].append(loss_purification.item())

        print(
            f"[Timing] speaker_loss: {t_speaker:.2f}s | purification: {t_purif:.2f}s | pgd_step: {t_pgd:.2f}s"
        )
        return x_adv.detach(), trajectory


class SingleTargetPGD:
    """Baseline: Single-target PGD (only attacks speaker recognition)"""

    def __init__(self, speaker_model, config):
        self.speaker_model = speaker_model
        self.config = config

        self.epsilon = config.epsilon
        self.num_iterations = config.num_iterations
        self.step_size = config.step_size
        self.alpha = config.alpha
        self.beta = config.beta
        self.device = config.device

    def attack(self, x_clean, target_embed):
        """Execute single-target PGD attack"""
        x_adv = x_clean.clone().detach().requires_grad_(True)

        with torch.no_grad():
            source_embed = self.speaker_model.get_embedding(x_clean)

        # sanity check: verify gradients flow through speaker model
        _test = x_clean.clone().detach().requires_grad_(True)
        _emb = self.speaker_model.get_embedding(_test)
        _loss = -F.cosine_similarity(_emb, source_embed, dim=1).mean()
        _loss.backward()
        print(f"  [grad check] grad_max={_test.grad.abs().max().item():.6f}")

        # random init to escape zero-gradient at x_adv == x_clean
        with torch.no_grad():
            x_adv = x_clean + torch.empty_like(x_clean).uniform_(
                -self.epsilon, self.epsilon
            )
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
        x_adv = x_adv.detach().requires_grad_(True)

        for i in range(self.num_iterations):
            if x_adv.grad is not None:
                x_adv.grad.zero_()

            embed_adv = self.speaker_model.get_embedding(x_adv)
            loss = -F.cosine_similarity(embed_adv, source_embed, dim=1).mean()

            loss.backward()

            grad_norm = x_adv.grad.abs().max().item()
            print(
                f"  single iter {i:02d}: loss={loss.item():.4f} grad_max={grad_norm:.6f}"
            )

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

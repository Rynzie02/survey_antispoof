"""
FullTraceDiffAttackPGD: dual-objective PGD with end-to-end autograd through diffusion.

Loss = alpha * (-cos_sim(emb_clean, emb(x_adv)))           # break direct cloning
     + beta  * (-cos_sim(emb_clean, emb(purify(x_adv))))   # break post-purification cloning
"""

import math

import torch
import torch.nn.functional as F


class FullTraceDiffAttackPGD:
    """
    Suitable for small reverse_timestep (3-10 steps) where the full graph fits in memory.
    """

    def __init__(self, speaker_model, purification_model, config):
        self.speaker_model = speaker_model
        self.purification_model = purification_model
        self.speaker_model.requires_grad_(False)
        self.speaker_model.eval()
        self.purification_model.requires_grad_(False)
        self.purification_model.eval()
        self.epsilon = config.epsilon
        self.num_iterations = config.num_iterations
        self.step_size = config.step_size
        self.alpha = getattr(config, "alpha", 0.5)
        self.beta = getattr(config, "beta", 0.5)
        self.weight_strategy = getattr(config, "weight_strategy", "fixed")
        # staged: keep the first direct_ratio iterations cheap, then add diffusion.
        self.stage_direct_ratio = min(
            1.0,
            max(0.0, getattr(config, "staged_direct_ratio", 0.75)),
        )
        self.stage_switch = min(
            self.num_iterations,
            math.floor(self.num_iterations * self.stage_direct_ratio),
        )
        self.device = config.device

    def attack(self, x_clean, return_trajectory=False):
        diffwave = self.purification_model.diffwave
        Alpha_bar = diffwave.diffusion_hyperparams["Alpha_bar"]
        T = diffwave.reverse_timestep

        with torch.no_grad():
            if hasattr(self.speaker_model, 'get_clean_embeddings'):
                emb_clean = self.speaker_model.get_clean_embeddings(x_clean)
            else:
                emb_clean = self.speaker_model.get_embedding(x_clean)

        x_adv = (x_clean + torch.empty_like(x_clean).uniform_(-self.epsilon, self.epsilon)).clamp(-1, 1).detach()

        trajectory = {"total_loss": [], "direct_loss": [], "purif_loss": []} if return_trajectory else None

        for i in range(self.num_iterations):
            x_adv = x_adv.requires_grad_(True)

            # Loss 1: break direct cloning (x_adv without purification)
            if hasattr(self.speaker_model, 'compute_loss'):
                loss_direct = self.speaker_model.compute_loss(emb_clean, x_adv)
            else:
                emb_adv_direct = self.speaker_model.get_embedding(x_adv)
                loss_direct = -F.cosine_similarity(emb_clean, emb_adv_direct, dim=1).mean()

            if self.weight_strategy == "staged":
                a, b = (1.0, 0.0) if i < self.stage_switch else (self.alpha, self.beta)
            else:
                a, b = self.alpha, self.beta

            # Loss 2: break post-purification cloning. Skip it when its weight is zero
            # so staged stage-1 does not pay the expensive diffusion cost.
            loss_purif = None
            if b != 0.0:
                x_adv_3d = x_adv.unsqueeze(1)
                z = torch.randn_like(x_adv_3d)
                x_T = torch.sqrt(Alpha_bar[T - 1]) * x_adv_3d + torch.sqrt(1 - Alpha_bar[T - 1]) * z
                x_purified = diffwave._reverse(x_T)
                if hasattr(self.speaker_model, 'compute_loss'):
                    loss_purif = self.speaker_model.compute_loss(emb_clean, x_purified.squeeze(1))
                else:
                    emb_adv_purif = self.speaker_model.get_embedding(x_purified.squeeze(1))
                    loss_purif = -F.cosine_similarity(emb_clean, emb_adv_purif, dim=1).mean()

            if loss_purif is None:
                loss_purif_for_loss = torch.zeros((), device=x_adv.device)
                loss_purif_log = None
            else:
                loss_purif_for_loss = loss_purif
                loss_purif_log = loss_purif.item()

            loss = a * loss_direct + b * loss_purif_for_loss
            loss.backward()

            with torch.no_grad():
                x_adv = x_adv + self.step_size * x_adv.grad.sign()
                x_adv = torch.clamp(x_adv, x_clean - self.epsilon, x_clean + self.epsilon).clamp(-1, 1)

            purif_msg = "skipped" if loss_purif_log is None else f"{loss_purif_log:.4f}"
            print(f"  fulltrace iter {i:02d}: total={loss.item():.4f}  direct={loss_direct.item():.4f}  purif={purif_msg}")

            if return_trajectory and i % 10 == 0:
                trajectory["total_loss"].append(loss.item())
                trajectory["direct_loss"].append(loss_direct.item())
                trajectory["purif_loss"].append(loss_purif_log)

        return x_adv.detach(), trajectory

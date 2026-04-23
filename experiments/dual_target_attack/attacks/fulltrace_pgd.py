"""
FullTraceDiffAttackPGD: dual-objective PGD with end-to-end autograd through diffusion.

Loss = alpha * (-cos_sim(emb_clean, emb(x_adv)))           # break direct cloning
     + beta  * (-cos_sim(emb_clean, emb(purify(x_adv))))   # break post-purification cloning
"""

import torch
import torch.nn.functional as F


class FullTraceDiffAttackPGD:
    """
    Suitable for small reverse_timestep (3-10 steps) where the full graph fits in memory.
    """

    def __init__(self, speaker_model, purification_model, config):
        self.speaker_model = speaker_model
        self.purification_model = purification_model
        self.epsilon = config.epsilon
        self.num_iterations = config.num_iterations
        self.step_size = config.step_size
        self.alpha = getattr(config, "alpha", 0.5)
        self.beta = getattr(config, "beta", 0.5)
        self.device = config.device

    def attack(self, x_clean, return_trajectory=False):
        diffwave = self.purification_model.diffwave
        Alpha_bar = diffwave.diffusion_hyperparams["Alpha_bar"]
        T = diffwave.reverse_timestep

        with torch.no_grad():
            emb_clean = self.speaker_model.get_embedding(x_clean)

        x_adv = (x_clean + torch.empty_like(x_clean).uniform_(-self.epsilon, self.epsilon)).clamp(-1, 1).detach()

        trajectory = {"total_loss": [], "direct_loss": [], "purif_loss": []} if return_trajectory else None

        for i in range(self.num_iterations):
            x_adv = x_adv.requires_grad_(True)

            # Loss 1: break direct cloning (x_adv without purification)
            emb_adv_direct = self.speaker_model.get_embedding(x_adv)
            loss_direct = -F.cosine_similarity(emb_clean, emb_adv_direct, dim=1).mean()

            # Loss 2: break post-purification cloning
            x_adv_3d = x_adv.unsqueeze(1)
            z = torch.randn_like(x_adv_3d)
            x_T = torch.sqrt(Alpha_bar[T - 1]) * x_adv_3d + torch.sqrt(1 - Alpha_bar[T - 1]) * z
            x_purified = diffwave._reverse(x_T)
            emb_adv_purif = self.speaker_model.get_embedding(x_purified.squeeze(1))
            loss_purif = -F.cosine_similarity(emb_clean, emb_adv_purif, dim=1).mean()

            loss = self.alpha * loss_direct + self.beta * loss_purif
            loss.backward()

            with torch.no_grad():
                x_adv = x_adv + self.step_size * x_adv.grad.sign()
                x_adv = torch.clamp(x_adv, x_clean - self.epsilon, x_clean + self.epsilon).clamp(-1, 1)

            print(f"  fulltrace iter {i:02d}: total={loss.item():.4f}  direct={loss_direct.item():.4f}  purif={loss_purif.item():.4f}")

            if return_trajectory and i % 10 == 0:
                trajectory["total_loss"].append(loss.item())
                trajectory["direct_loss"].append(loss_direct.item())
                trajectory["purif_loss"].append(loss_purif.item())

        return x_adv.detach(), trajectory

"""
DiffAttack-style PGD for DiffWave (waveform domain).

Memory-efficient: denoising runs under no_grad, gradients are propagated
manually step-by-step (same strategy as diff_exp/quick_verify_asv_attack.py).
"""

import math

import torch
import torch.nn.functional as F


class DiffAttackPGD:
    """
    Untargeted DiffAttack-style PGD.

    Loss = alpha * (-cos_sim(emb_clean, emb_adv_purified))
           + beta  * MSE(denoise_traj_adv[t], forward_traj_adv[t])

    Gradient is computed manually through the reverse diffusion steps to avoid
    storing the full autograd graph (OOM-safe).
    """

    def __init__(self, speaker_model, purification_model, config):
        self.speaker_model = speaker_model
        self.diffwave = purification_model.diffwave

        self.epsilon = config.epsilon
        self.num_iterations = config.num_iterations
        self.step_size = config.step_size
        self.device = config.device
        self.alpha = getattr(config, "alpha", 0.5)
        self.beta = getattr(config, "beta", 0.5)
        self.t_interval = getattr(config, "diffattack_t_interval", 1)
        self.use_targeted = getattr(config, "use_targeted", False)
        self.weight_strategy = getattr(config, "weight_strategy", "fixed")
        stage_ratio = min(1.0, max(0.0, getattr(config, "staged_direct_ratio", 0.75)))
        self.stage_switch = math.floor(self.num_iterations * stage_ratio)

        if self.use_targeted:
            raise ValueError("DiffAttackPGD currently supports untargeted attack only.")

    # ------------------------------------------------------------------
    # Manual gradient helpers (mirrors diff_exp logic, adapted for DiffWave)
    # ------------------------------------------------------------------

    def _grad_through_reverse_step(
        self, grad_out, x_t_stored, current_t, prev_t, noise
    ):
        """
        Backprop grad_out through one _reverse_step, return grad w.r.t. x_t_stored.
        Only one step's graph is alive at a time → memory efficient.
        """
        diffwave = self.diffwave
        alpha_bar = diffwave.diffusion_hyperparams["Alpha_bar"]

        with torch.enable_grad():
            x_t = x_t_stored.clone().detach().requires_grad_(True)
            diffusion_steps = current_t * torch.ones(
                (x_t.shape[0], 1), device=x_t.device
            )
            epsilon_theta = diffwave.model((x_t, diffusion_steps))

            sqrt_ab = torch.sqrt(alpha_bar[current_t])
            sqrt_1mab = torch.sqrt(1 - alpha_bar[current_t])
            pred_x0 = (x_t - sqrt_1mab * epsilon_theta) / sqrt_ab

            if prev_t < 0:
                x_next = pred_x0
            else:
                ab_prev = alpha_bar[prev_t]
                sigma = torch.sqrt(
                    ((1 - ab_prev) / (1 - alpha_bar[current_t]))
                    * (1 - alpha_bar[current_t] / ab_prev)
                )
                coeff_eps = torch.sqrt(torch.clamp(1 - ab_prev - sigma**2, min=0.0))
                x_next = (
                    torch.sqrt(ab_prev) * pred_x0
                    + coeff_eps * epsilon_theta
                    + sigma * noise
                )

            loss = torch.sum(x_next * grad_out)
            grad_in = torch.autograd.grad(loss, x_t)[0].detach()

        return grad_in

    def _grad_through_forward_diffusion(self, x_adv, grad_xT, Alpha_bar, T, z):
        """
        Backprop through x_T = sqrt(ab[T-1]) * x_adv + sqrt(1-ab[T-1]) * z
        Returns grad w.r.t. x_adv.
        """
        return self._grad_through_forward_state(x_adv, grad_xT, Alpha_bar, T - 1, z)

    def _grad_through_forward_state(self, x_adv, grad_xt, Alpha_bar, t, z):
        """
        Backprop through x_t = sqrt(ab[t]) * x_adv + sqrt(1-ab[t]) * z
        Returns grad w.r.t. x_adv.
        """
        with torch.enable_grad():
            x = x_adv.clone().detach().requires_grad_(True)
            x_t = torch.sqrt(Alpha_bar[t]) * x + torch.sqrt(1 - Alpha_bar[t]) * z
            loss = torch.sum(x_t * grad_xt)
            return torch.autograd.grad(loss, x)[0].detach()

    # ------------------------------------------------------------------
    # Core attack
    # ------------------------------------------------------------------

    def attack(self, x_clean, return_trajectory=False):
        diffwave = self.diffwave
        T = diffwave.reverse_timestep
        schedule = diffwave._build_reverse_schedule()
        Alpha_bar = diffwave.diffusion_hyperparams["Alpha_bar"]
        mse_steps = set(range(0, T, self.t_interval))

        # Clean embedding (no purification)
        with torch.no_grad():
            if hasattr(self.speaker_model, 'get_clean_embeddings'):
                emb_clean = self.speaker_model.get_clean_embeddings(x_clean)
            else:
                emb_clean = self.speaker_model.get_embedding(x_clean)

        # Random init
        x_adv = x_clean.clone().detach()
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.epsilon, self.epsilon)
        x_adv = torch.clamp(x_adv, -1.0, 1.0)

        trajectory = (
            {"total_loss": [], "emb_loss": [], "mse_loss": []}
            if return_trajectory
            else None
        )

        for i in range(self.num_iterations):
            if self.weight_strategy == "staged":
                a = 1.0 if i < self.stage_switch else self.alpha
                b = 0.0 if i < self.stage_switch else self.beta
            else:
                a, b = self.alpha, self.beta

            x_adv_3d = x_adv.unsqueeze(1)  # (B,1,T)

            if b == 0.0:
                # Stage 1: skip diffusion entirely, attack direct embedding
                with torch.enable_grad():
                    x_in = x_adv.clone().detach().requires_grad_(True)
                    if hasattr(self.speaker_model, 'compute_loss'):
                        emb_loss = self.speaker_model.compute_loss(emb_clean, x_in)
                    else:
                        emb_adv = self.speaker_model.get_embedding(x_in)
                        emb_loss = -F.cosine_similarity(emb_clean, emb_adv, dim=1).mean()
                    grad_x_adv = torch.autograd.grad(emb_loss, x_in)[0].detach() * a
                mse_total, n_mse = 0.0, 0
            else:
                # Stage 2: full diffusion forward + reverse + MSE
                with torch.no_grad():
                    z = torch.randn_like(x_adv_3d)
                    x_T = (
                        torch.sqrt(Alpha_bar[T - 1]) * x_adv_3d
                        + torch.sqrt(1 - Alpha_bar[T - 1]) * z
                    )
                    fwd_adv = {
                        t: torch.sqrt(Alpha_bar[t]) * x_adv_3d + torch.sqrt(1 - Alpha_bar[t]) * z
                        for t in mse_steps
                    }
                    noises = {t: torch.randn_like(x_T) for t in schedule}
                    if schedule:
                        noises[schedule[-1]] = torch.zeros_like(x_T)
                    mid_x = {}
                    x_t = x_T
                    for idx, current_t in enumerate(schedule):
                        mid_x[current_t] = x_t.clone()
                        prev_t = schedule[idx + 1] if idx + 1 < len(schedule) else -1
                        x_t = diffwave._reverse_step(x_t, current_t, prev_t, noises[current_t])
                    x_purified = x_t

                with torch.enable_grad():
                    x_pur = x_purified.squeeze(1).clone().detach().requires_grad_(True)
                    if hasattr(self.speaker_model, 'compute_loss'):
                        emb_loss = self.speaker_model.compute_loss(emb_clean, x_pur)
                    else:
                        emb_adv = self.speaker_model.get_embedding(x_pur)
                        emb_loss = -F.cosine_similarity(emb_clean, emb_adv, dim=1).mean()
                    grad_emb = torch.autograd.grad(emb_loss, x_pur)[0].detach()

                grad = grad_emb.unsqueeze(1) * a
                mse_total, n_mse = 0.0, 0
                grad_x_adv_direct = torch.zeros_like(x_adv_3d)
                for idx in reversed(range(len(schedule))):
                    current_t = schedule[idx]
                    prev_t = schedule[idx + 1] if idx + 1 < len(schedule) else -1
                    if current_t in mse_steps and current_t in fwd_adv:
                        with torch.enable_grad():
                            xt_rev = mid_x[current_t].clone().detach().requires_grad_(True)
                            xt_fwd = fwd_adv[current_t].clone().detach().requires_grad_(True)
                            mse = F.mse_loss(xt_rev, xt_fwd)
                            mse_grad_rev, mse_grad_fwd = torch.autograd.grad(mse, (xt_rev, xt_fwd))
                        grad = grad + b * mse_grad_rev.detach()
                        grad_x_adv_direct = grad_x_adv_direct + self._grad_through_forward_state(
                            x_adv_3d, b * mse_grad_fwd.detach(), Alpha_bar, current_t, z,
                        )
                        mse_total += mse.item()
                        n_mse += 1
                    grad = self._grad_through_reverse_step(
                        grad, mid_x[current_t], current_t, prev_t, noises[current_t]
                    )

                grad_x_adv = self._grad_through_forward_diffusion(
                    x_adv_3d, grad, Alpha_bar, T, z
                ).squeeze(1)
                grad_x_adv = grad_x_adv + grad_x_adv_direct.squeeze(1)

            # ---- PGD update ----
            x_adv = x_adv + self.step_size * grad_x_adv.sign()
            x_adv = torch.clamp(x_adv, x_clean - self.epsilon, x_clean + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0).detach()

            mse_avg = mse_total / max(n_mse, 1)
            if i % 10 == 0 or i == self.num_iterations - 1:
                mse_str = "skipped" if b == 0.0 else f"{mse_avg:.4f}"
                grad_max = grad_x_adv.abs().max().item()
                print(
                    f"  diffattack iter {i:02d}: "
                    f"emb={emb_loss.item():.4f}  mse={mse_str}  "
                    f"grad_max={grad_max:.6f}",
                    flush=True,
                )

            if return_trajectory and i % 10 == 0:
                trajectory["emb_loss"].append(emb_loss.item())
                trajectory["mse_loss"].append(mse_avg)
                trajectory["total_loss"].append(
                    self.alpha * emb_loss.item() + self.beta * mse_avg
                )

        return x_adv, trajectory

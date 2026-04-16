import argparse
import json

import torch
from .DiffWave_Unconditional.WaveNet import WaveNet_Speech_Commands
from .DiffWave_Unconditional.util import calc_diffusion_hyperparams

import numpy as np
from typing import Union
import torchaudio
import csv
from torch.utils.checkpoint import checkpoint


class DiffWave(torch.nn.Module):
    def __init__(
        self,
        model: WaveNet_Speech_Commands,
        diffusion_hyperparams: dict,
        reverse_timestep: int = 200,
        grad_enable=True,
        device: Union[str, torch.device] = "cuda",
        step_stride: int = 1,
        use_checkpoint: bool = False,
        defense_method="None",
    ):
        super().__init__()

        """
            model: input (x_t, t), output epsilon_theta at timestep t
        """

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.diffusion_hyperparams = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in diffusion_hyperparams.items()
        }
        self.reverse_timestep = reverse_timestep
        self.step_stride = max(int(step_stride), 1)
        self.freeze = False
        self.grad_enable = grad_enable
        self.use_checkpoint = use_checkpoint
        self.defense_method = defense_method

    def forward(self, waveforms: Union[torch.Tensor, np.ndarray]):

        if isinstance(waveforms, np.ndarray):
            waveforms = torch.from_numpy(waveforms)
        waveforms = waveforms.to(self.device)

        output = waveforms

        if self.defense_method == "DualPure" or self.defense_method == "DiffNoise":
            output = self._diffusion(waveforms)
        elif (
            self.defense_method == "AudioPure"
            or self.defense_method == "DDPM"
            or self.defense_method == "PhonePuRe"
        ):
            output = self._diffusion(waveforms)
            output = self._reverse(output)
        elif self.defense_method == "DiffRev":
            output = self._reverse(output)
        elif self.defense_method == "OneShot":
            output = self._diffusion(waveforms)
            output = self.one_shot_denoise(output)

        return output

    def _diffusion(self, x_0: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """convert np.array to torch.tensor"""
        if isinstance(x_0, np.ndarray):
            x_0 = torch.from_numpy(x_0)
        x_0 = x_0.to(self.device)

        T, Alpha, Alpha_bar, Sigma = (
            self.diffusion_hyperparams["T"],
            self.diffusion_hyperparams["Alpha"],
            self.diffusion_hyperparams["Alpha_bar"],
            self.diffusion_hyperparams["Sigma"],
        )
        assert len(Alpha) == T
        assert len(Alpha_bar) == T
        assert len(Sigma) == T
        assert x_0.ndim == 3

        """noising"""
        z = torch.randn_like(x_0)
        x_t = (
            torch.sqrt(Alpha_bar[self.reverse_timestep - 1]) * x_0
            + torch.sqrt(1 - Alpha_bar[self.reverse_timestep - 1]) * z
        )
        return x_t

    def _diffusion_intermediates(self, x_0: torch.Tensor, capture_steps: list) -> dict:
        """
        Forward diffusion, capturing x_t at specified timesteps.
        capture_steps: list of t values to capture (e.g. [5, 10, 15, 20])
        Returns: dict {t: x_t}
        """
        x_0 = x_0.to(self.device)
        Alpha_bar = self.diffusion_hyperparams["Alpha_bar"]
        z = torch.randn_like(x_0)
        intermediates = {}
        for t in capture_steps:
            x_t = torch.sqrt(Alpha_bar[t]) * x_0 + torch.sqrt(1 - Alpha_bar[t]) * z
            intermediates[t] = x_t
        # also return the final noised x_T
        t_final = self.reverse_timestep - 1
        if t_final not in intermediates:
            intermediates[t_final] = (
                torch.sqrt(Alpha_bar[t_final]) * x_0
                + torch.sqrt(1 - Alpha_bar[t_final]) * z
            )
        return intermediates, z

    def _build_reverse_schedule(self):
        schedule = list(range(self.reverse_timestep - 1, -1, -self.step_stride))
        if not schedule or schedule[-1] != 0:
            schedule.append(0)
        return schedule

    def _reverse_step(
        self, x_t: torch.Tensor, current_t: int, prev_t: int, noise: torch.Tensor
    ) -> torch.Tensor:
        alpha_bar = self.diffusion_hyperparams["Alpha_bar"]

        diffusion_steps = current_t * torch.ones((x_t.shape[0], 1), device=x_t.device)
        epsilon_theta = self.model((x_t, diffusion_steps))

        alpha_bar_current = alpha_bar[current_t]
        sqrt_alpha_bar_current = torch.sqrt(alpha_bar_current)
        sqrt_one_minus_alpha_bar_current = torch.sqrt(1 - alpha_bar_current)

        pred_x0 = (
            x_t - sqrt_one_minus_alpha_bar_current * epsilon_theta
        ) / sqrt_alpha_bar_current

        if prev_t < 0:
            return pred_x0

        alpha_bar_prev = alpha_bar[prev_t]
        sigma = torch.sqrt(
            ((1 - alpha_bar_prev) / (1 - alpha_bar_current))
            * (1 - alpha_bar_current / alpha_bar_prev)
        )
        coeff_eps = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma**2, min=0.0))

        return (
            torch.sqrt(alpha_bar_prev) * pred_x0
            + coeff_eps * epsilon_theta
            + sigma * noise
        )

    def _reverse(self, x_t: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """convert np.array to torch.tensor"""
        if isinstance(x_t, np.ndarray):
            x_t = torch.from_numpy(x_t)
        x_t = x_t.to(self.device)

        T, Alpha, Alpha_bar, Sigma = (
            self.diffusion_hyperparams["T"],
            self.diffusion_hyperparams["Alpha"],
            self.diffusion_hyperparams["Alpha_bar"],
            self.diffusion_hyperparams["Sigma"],
        )
        assert len(Alpha) == T
        assert len(Alpha_bar) == T
        assert len(Sigma) == T
        assert x_t.ndim == 3

        """denoising"""
        x_t_rev = x_t.clone()
        schedule = self._build_reverse_schedule()
        noises = {current_t: torch.randn_like(x_t_rev) for current_t in schedule}
        if schedule:
            noises[schedule[-1]] = torch.zeros_like(x_t_rev)

        for idx, current_t in enumerate(schedule):
            prev_t = schedule[idx + 1] if idx + 1 < len(schedule) else -1
            noise = noises[current_t]
            if self.use_checkpoint and x_t_rev.requires_grad:
                x_t_rev = checkpoint(
                    lambda inp, step=current_t, next_step=prev_t, step_noise=noise: (
                        self._reverse_step(inp, step, next_step, step_noise)
                    ),
                    x_t_rev,
                    use_reentrant=False,
                )
            else:
                x_t_rev = self._reverse_step(x_t_rev, current_t, prev_t, noise)
        return x_t_rev

    def _reverse_with_intermediates(
        self, x_t: torch.Tensor, forward_intermediates: dict
    ):
        """
        Full reverse denoising (step_stride=1), capturing reverse x_t at the same
        timesteps as forward_intermediates for paired comparison.
        Returns: (x_0, list of (x_t_forward, x_t_reverse) pairs)
        """
        if isinstance(x_t, np.ndarray):
            x_t = torch.from_numpy(x_t)
        x_t = x_t.to(self.device)
        assert x_t.ndim == 3

        x_t_rev = x_t.clone()
        # Always use full step-by-step reverse (stride=1) for quality
        schedule = list(range(self.reverse_timestep - 1, -1, -1))
        noises = {t: torch.randn_like(x_t_rev) for t in schedule}
        noises[0] = torch.zeros_like(x_t_rev)

        capture_ts = set(forward_intermediates.keys())
        pairs = []  # (x_t_forward, x_t_reverse)

        for idx, current_t in enumerate(schedule):
            prev_t = schedule[idx + 1] if idx + 1 < len(schedule) else -1
            noise = noises[current_t]
            if self.use_checkpoint and x_t_rev.requires_grad:
                x_t_rev = checkpoint(
                    lambda inp, step=current_t, next_step=prev_t, n=noise: (
                        self._reverse_step(inp, step, next_step, n)
                    ),
                    x_t_rev,
                    use_reentrant=False,
                )
            else:
                x_t_rev = self._reverse_step(x_t_rev, current_t, prev_t, noise)

            if prev_t in capture_ts:
                pairs.append((forward_intermediates[prev_t], x_t_rev))

        return x_t_rev, pairs

    def fast_reverse(self, x_t: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """convert np.array to torch.tensor"""
        if isinstance(x_t, np.ndarray):
            x_t = torch.from_numpy(x_t)
        x_t = x_t.to(self.device)

        T, Alpha, Alpha_bar, Sigma = (
            self.diffusion_hyperparams["T"],
            self.diffusion_hyperparams["Alpha"],
            self.diffusion_hyperparams["Alpha_bar"],
            self.diffusion_hyperparams["Sigma"],
        )

        K = 3
        S = torch.linspace(1, self.reverse_timestep, K)
        S = torch.round(S).int() - 1
        Beta_new, Beta_tilde_new = torch.zeros(size=(K,)), torch.zeros(size=(K,))

        for i in range(K):
            if i > 0:
                Beta_new[i] = 1 - Alpha_bar[S[i]] / Alpha_bar[S[i - 1]]
                Beta_tilde_new[i] = (
                    (1 - Alpha_bar[S[i - 1]]) / (1 - Alpha_bar[S[i]]) * Beta_new[i]
                )
            else:
                Beta_new[i] = 1 - Alpha_bar[S[i]]
                Beta_tilde_new[i] = 0
        Alpha_new = 1 - Beta_new
        Alpha_bar_new = torch.cumprod(Alpha_new, dim=0)

        x_St = x_t
        for t in range(K - 1, -1, -1):
            real_t = S[t]
            eps_St = self.model(
                (x_St, real_t * torch.ones((x_St.shape[0], 1), device=self.device))
            )
            mu_St = (
                x_St - (1 - Alpha_new[t]) / torch.sqrt(1 - Alpha_bar_new[t]) * eps_St
            ) / torch.sqrt(Alpha_new[t])
            sigma_St = Beta_tilde_new[t]
            x_St = mu_St + sigma_St * torch.randn_like(x_St)

        return x_St

    def compute_coefficients(self, x_t: Union[torch.Tensor, np.ndarray], t: int):
        """
        a single reverse step
        compute coefficients at timestep t+1
        t: in [0, T-1]
        return: eps_theta(x_t+1, t+1), mu_theta(x_t+1, t+1) and sigma_theta(x_t+1, t+1)
        """

        Alpha, Alpha_bar, Sigma = (
            self.diffusion_hyperparams["Alpha"],
            self.diffusion_hyperparams["Alpha_bar"],
            self.diffusion_hyperparams["Sigma"],
        )

        diffusion_steps = t * torch.ones((x_t.shape[0], 1), device=x_t.device)
        epsilon_theta = self.model((x_t, diffusion_steps))
        mu_theta = (
            x_t - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta
        ) / torch.sqrt(Alpha[t])
        sigma_theta = Sigma[t]

        # sigma_theta = self.diffusion_hyperparams["Beta"][t].sqrt()

        return epsilon_theta, mu_theta, sigma_theta

    @torch.no_grad()
    def compute_eps_t(self, x_t: Union[torch.Tensor, np.ndarray], t):

        diffusion_steps = t * torch.ones((x_t.shape[0], 1), device=x_t.device)
        epsilon_theta = self.model((x_t, diffusion_steps))

        return epsilon_theta

    def one_shot_denoise(self, x_t: Union[torch.Tensor, np.ndarray]):

        if isinstance(x_t, np.ndarray):
            x_t = torch.from_numpy(x_t)
        x_t = x_t.to(self.device)
        t = self.reverse_timestep - 1
        diffusion_steps = t * torch.ones((x_t.shape[0], 1), device=x_t.device)
        epsilon_theta = self.model((x_t, diffusion_steps))

        pred_x_0 = self._predict_x0_from_eps(x_t, t, epsilon_theta)

        return pred_x_0

    def two_shot_denoise(self, x_t: Union[torch.Tensor, np.ndarray]):

        if isinstance(x_t, np.ndarray):
            x_t = torch.from_numpy(x_t)
        x_t = x_t.to(self.device)
        t = self.reverse_timestep - 1
        diffusion_steps = t * torch.ones((x_t.shape[0], 1), device=x_t.device)
        epsilon_theta = self.model((x_t, diffusion_steps))

        pred_x_1 = self._predict_x1_from_eps(x_t, t, epsilon_theta)
        pred_x_0 = self._predict_x0_from_x1(pred_x_1)

        return pred_x_0

    def _predict_x0_from_eps(self, x_t, t, eps):

        assert x_t.shape == eps.shape

        Alpha_bar = self.diffusion_hyperparams["Alpha_bar"]

        sqrt_recip_alphas_bar = (1 / Alpha_bar).sqrt()
        sqrt_recipm1_alphas_bar = (1 / Alpha_bar - 1).sqrt()
        pred_x_0 = (
            self._extract_into_tensor(sqrt_recip_alphas_bar, t, x_t.shape) * x_t
            - self._extract_into_tensor(sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

        return pred_x_0

    def _predict_x1_from_eps(self, x_t, t, eps):

        Alpha = self.diffusion_hyperparams["Alpha"]
        Alpha_bar = self.diffusion_hyperparams["Alpha_bar"]
        Beta = self.diffusion_hyperparams["Beta"]

        mu = (Alpha_bar[t] / Alpha[0]).sqrt()
        sigma = (1 - Alpha_bar[t] - (Alpha_bar[t] / Alpha[0]) * Beta[0] ** 2).sqrt()

        pred_x_1 = (x_t - sigma * eps) / mu

        return pred_x_1

    def _predict_x0_from_x1(self, x_1):

        _, mu_0, _ = self.compute_coefficients(x_1, 0)

        pred_x_0 = mu_0

        return pred_x_0

    def _extract_into_tensor(self, arr_or_func, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.
        :param arr: the 1-D numpy array or a func.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        if callable(arr_or_func):
            res = arr_or_func(timesteps).float()
        else:
            if isinstance(arr_or_func, torch.Tensor):
                res = arr_or_func.to(self.device)[timesteps].float()
            elif isinstance(arr_or_func, np.ndarray):
                res = torch.from_numpy(arr_or_func).to(self.device)[timesteps].float()
            else:
                raise TypeError(
                    "Unsupported data type {} in arr_or_func".format(type(arr_or_func))
                )

        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)


def create_diffwave_model(
    args,
    model_path,
    config_path,
    reverse_timestep=25,
    device: Union[str, torch.device] = "cuda",
    step_stride: int = 1,
    use_checkpoint: bool = False,
):

    with open(config_path) as f:
        data = f.read()
    cfg = json.loads(data)

    wavenet_config = cfg["wavenet_config"]  # to define wavenet
    diffusion_config = cfg["diffusion_config"]  # basic hyperparameters
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)

    device = torch.device(device)
    WaveNet_model = WaveNet_Speech_Commands(**wavenet_config).to(device)
    checkpoint_data = torch.load(model_path, map_location=device)
    WaveNet_model.load_state_dict(checkpoint_data["model_state_dict"])

    Denoiser = DiffWave(
        model=WaveNet_model,
        diffusion_hyperparams=diffusion_hyperparams,
        reverse_timestep=reverse_timestep,
        device=device,
        step_stride=step_stride,
        use_checkpoint=use_checkpoint,
        defense_method=args.defense,
    )

    return Denoiser

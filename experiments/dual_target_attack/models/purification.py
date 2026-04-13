"""
Diffusion-based audio purification model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleDiffusionPurifier(nn.Module):
    """
    Simplified diffusion-based audio purification model
    Based on DDPM/DDIM for audio denoising
    """

    def __init__(self, num_steps=50, noise_schedule='linear'):
        super().__init__()
        self.num_steps = num_steps
        self.noise_schedule = noise_schedule

        # Define noise schedule
        self.betas = self._get_noise_schedule(num_steps, noise_schedule)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Denoising network (simplified U-Net for audio)
        self.denoiser = AudioUNet()

    def _get_noise_schedule(self, num_steps, schedule_type):
        """Get noise schedule (beta values)"""
        if schedule_type == 'linear':
            return torch.linspace(0.0001, 0.02, num_steps)
        elif schedule_type == 'cosine':
            steps = torch.arange(num_steps + 1, dtype=torch.float32) / num_steps
            alphas_cumprod = torch.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

    def add_noise(self, x, t):
        """Add noise to audio at timestep t (forward diffusion)"""
        # x: (batch, samples)
        # t: (batch,) timestep indices

        noise = torch.randn_like(x)
        alpha_t = self.alphas_cumprod[t].view(-1, 1).to(x.device)

        noisy_x = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
        return noisy_x, noise

    def denoise_step(self, x_t, t):
        """Single denoising step (reverse diffusion)"""
        # Predict noise
        predicted_noise = self.denoiser(x_t, t)

        # Compute denoised audio
        alpha_t = self.alphas_cumprod[t].view(-1, 1).to(x_t.device)
        alpha_t_prev = self.alphas_cumprod[t - 1].view(-1, 1).to(x_t.device) if t[0] > 0 else torch.ones_like(alpha_t)

        # DDIM sampling (deterministic, faster)
        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        x_t_prev = torch.sqrt(alpha_t_prev) * x_0_pred + torch.sqrt(1 - alpha_t_prev) * predicted_noise

        return x_t_prev

    def purify(self, x_adv, num_steps=None):
        """
        Purify adversarial audio through diffusion process

        Args:
            x_adv: Adversarial audio (batch, samples)
            num_steps: Number of diffusion steps (default: self.num_steps)
        Returns:
            Purified audio
        """
        if num_steps is None:
            num_steps = self.num_steps

        batch_size = x_adv.shape[0]
        device = x_adv.device

        # Forward diffusion: add noise
        t_start = torch.full((batch_size,), num_steps - 1, dtype=torch.long, device=device)
        x_t, _ = self.add_noise(x_adv, t_start)

        # Reverse diffusion: denoise
        for t in reversed(range(num_steps)):
            t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)
            x_t = self.denoise_step(x_t, t_batch)

        return x_t

    def forward(self, x_adv, num_steps=None):
        """Forward pass = purification"""
        return self.purify(x_adv, num_steps)


class AudioUNet(nn.Module):
    """Simplified U-Net for audio denoising"""

    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()

        # Time embedding for diffusion timestep
        self.time_embed = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        # Encoder
        self.enc1 = nn.Conv1d(in_channels, base_channels, 3, padding=1)
        self.enc2 = nn.Conv1d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        self.enc3 = nn.Conv1d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = nn.Conv1d(base_channels * 4, base_channels * 4, 3, padding=1)

        # Decoder
        self.dec3 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose1d(base_channels * 4, base_channels, 4, stride=2, padding=1)
        self.dec1 = nn.Conv1d(base_channels * 2, in_channels, 3, padding=1)

    def get_time_embedding(self, t):
        """Sinusoidal time embedding"""
        half_dim = 64
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return self.time_embed(embeddings)

    def forward(self, x, t):
        """
        Args:
            x: Audio (batch, samples)
            t: Timestep (batch,)
        Returns:
            Predicted noise
        """
        # Reshape for 1D convolution
        x = x.unsqueeze(1)  # (batch, 1, samples)

        # Time embedding
        t_emb = self.get_time_embedding(t)

        # Encoder
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2))

        # Bottleneck
        b = F.relu(self.bottleneck(e3))

        # Decoder with skip connections
        d3 = F.relu(self.dec3(b))
        d3 = torch.cat([d3, e2], dim=1)

        d2 = F.relu(self.dec2(d3))
        d2 = torch.cat([d2, e1], dim=1)

        out = self.dec1(d2)
        out = out.squeeze(1)  # (batch, samples)

        return out


class SimpleDenoiser(nn.Module):
    """Simple denoising autoencoder as a fast alternative to diffusion"""

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        """Denoise audio"""
        x = x.unsqueeze(1)  # (batch, 1, samples)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.squeeze(1)


def load_purification_model(model_type='diffusion', model_path=None, device='cuda'):
    """Load purification model"""

    if model_type == 'diffusion':
        model = SimpleDiffusionPurifier(num_steps=50, noise_schedule='linear')
    elif model_type == 'denoising_ae':
        model = SimpleDenoiser()
    else:
        raise ValueError(f"Unknown purification type: {model_type}")

    if model_path:
        try:
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded purification model from {model_path}")
        except:
            print(f"Could not load model from {model_path}, using random initialization")
    else:
        print("Using randomly initialized purification model")

    model = model.to(device)
    model.eval()
    return model

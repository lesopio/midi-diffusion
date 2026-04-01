from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiscreteDiffusion:
    def __init__(self, num_classes: int, num_steps: int, device: torch.device, schedule: str = "cosine"):
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.device = device

        if schedule == "cosine":
            self.betas = cosine_beta_schedule(num_steps).to(device)
        elif schedule == "linear":
            self.betas = torch.linspace(1e-4, 0.1, num_steps, device=device)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.num_steps, (batch_size,), device=self.device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
        b = x0.size(0)
        keep_prob = self.alphas_cumprod[t].view(b, 1, 1)
        noise = torch.randint(0, self.num_classes, x0.shape, device=x0.device)
        mask = torch.rand_like(x0.float()) < keep_prob
        xt = torch.where(mask, x0, noise)
        if pad_token is not None:
            xt = torch.where(x0 == pad_token, x0, xt)
        return xt

    def loss(self, logits: torch.Tensor, x0: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, self.num_classes),
            x0.reshape(-1),
            ignore_index=pad_token,
        )

    @torch.no_grad()
    def p_sample_loop(self, model, shape: tuple, cond: torch.Tensor | None = None,
                      guidance_scale: float = 1.0, temperature: float = 1.0) -> torch.Tensor:
        b = shape[0]
        xt = torch.randint(0, self.num_classes, shape, device=self.device)
        for t_ in reversed(range(self.num_steps)):
            t = torch.full((b,), t_, device=self.device, dtype=torch.long)
            if guidance_scale != 1.0 and cond is not None:
                logits_uncond = model(xt, t, None)
                logits_cond = model(xt, t, cond)
                logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
            else:
                logits = model(xt, t, cond)
            if temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            probs_flat = probs.reshape(-1, self.num_classes)
            samples = torch.multinomial(probs_flat, num_samples=1)
            xt = samples.view(*probs.shape[:-1])
        return xt

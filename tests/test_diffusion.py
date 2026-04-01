import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from midigen.diffusion import DiscreteDiffusion


def test_diffusion_shapes():
    device = torch.device("cpu")
    diffusion = DiscreteDiffusion(num_classes=10, num_steps=5, device=device)
    x0 = torch.randint(0, 10, (2, 8, 3), device=device)
    t = diffusion.sample_timesteps(2)
    xt = diffusion.q_sample(x0, t, pad_token=0)
    assert xt.shape == x0.shape

    logits = torch.randn(2, 8, 3, 10, device=device)
    loss = diffusion.loss(logits, x0, pad_token=0)
    assert loss.dim() == 0

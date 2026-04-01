from __future__ import annotations

from pathlib import Path
import torch

from .config import get_config, TrainConfig
from .data import (
    PITCH_OFFSET,
    PITCH_RANGE,
    tokens_to_pretty_midi,
)
from .diffusion import DiscreteDiffusion
from .model import PolyphonicModel


def _config_from_checkpoint(checkpoint_path: str) -> TrainConfig:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        cfg_dict = checkpoint.get("config")
        if isinstance(cfg_dict, dict):
            return TrainConfig(**cfg_dict)
    except Exception:
        pass
    return get_config("4090")


def load_model(checkpoint_path: str, device: torch.device, cfg: TrainConfig | None = None) -> PolyphonicModel:
    if cfg is None:
        cfg = _config_from_checkpoint(checkpoint_path)
    model_max_len = cfg.max_seq_len or cfg.seq_len
    vocab_size = PITCH_RANGE + PITCH_OFFSET
    model = PolyphonicModel(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        max_len=model_max_len,
        max_voices=cfg.max_voices,
        diffusion_steps=cfg.diffusion_steps,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def generate_tokens(checkpoint_path: str,
                    num_samples: int,
                    seq_len: int,
                    steps: int,
                    device: torch.device,
                    max_voices: int,
                    steps_per_beat: int,
                    temperature: float = 1.0) -> torch.Tensor:
    cfg = _config_from_checkpoint(checkpoint_path)
    cfg = TrainConfig(**{**cfg.__dict__, "seq_len": seq_len, "max_voices": max_voices, "steps_per_beat": steps_per_beat})
    model_max_len = cfg.max_seq_len or cfg.seq_len
    if seq_len > model_max_len:
        raise ValueError(f"seq_len={seq_len} exceeds model max_len={model_max_len}")

    model = load_model(checkpoint_path, device, cfg)
    if steps > cfg.diffusion_steps:
        raise ValueError(f"steps={steps} exceeds model diffusion_steps={cfg.diffusion_steps}")
    diffusion = DiscreteDiffusion(
        num_classes=PITCH_RANGE + PITCH_OFFSET,
        num_steps=steps,
        device=device,
        schedule="cosine",
    )

    def model_fn(xt, t, cond=None):
        note_logits, _, _ = model(xt, t, cond)
        return note_logits

    tokens = diffusion.p_sample_loop(
        model_fn,
        shape=(num_samples, seq_len, max_voices),
        cond=None,
        guidance_scale=1.0,
        temperature=temperature,
    )
    return tokens


def save_midi(tokens: torch.Tensor, out_path: str, steps_per_beat: int, tempo: float = 120.0) -> None:
    arr = tokens.detach().cpu().numpy()
    pm = tokens_to_pretty_midi(arr, steps_per_beat=steps_per_beat, tempo=tempo)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pm.write(out_path)

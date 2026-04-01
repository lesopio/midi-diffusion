from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    seq_len: int
    d_model: int
    num_layers: int
    nhead: int
    diffusion_steps: int
    use_amp: bool
    grad_accum_steps: int
    num_workers: int
    preload_data: bool
    max_voices: int = 4
    steps_per_beat: int = 4
    max_seq_len: int = 0
    dynamic_length: bool = False


def _config_4090() -> TrainConfig:
    return TrainConfig(
        batch_size=64,
        seq_len=512,
        d_model=512,
        num_layers=8,
        nhead=8,
        diffusion_steps=50,
        use_amp=True,
        grad_accum_steps=1,
        num_workers=8,
        preload_data=True,
        max_voices=4,
        steps_per_beat=4,
    )


def _config_4050() -> TrainConfig:
    return TrainConfig(
        batch_size=2,
        seq_len=256,
        d_model=192,
        num_layers=4,
        nhead=4,
        diffusion_steps=30,
        use_amp=True,
        grad_accum_steps=8,
        num_workers=2,
        preload_data=False,
        max_voices=4,
        steps_per_beat=4,
    )


def get_config(device_type: str = "4090") -> TrainConfig:
    device_type = device_type.lower()
    if device_type == "4090":
        return _config_4090()
    if device_type == "4050":
        return _config_4050()
    raise ValueError(f"Unknown device type: {device_type}")


def auto_detect() -> TrainConfig:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available")
    name = torch.cuda.get_device_name(0).lower()
    if "4090" in name:
        return _config_4090()
    if "4050" in name:
        return _config_4050()
    raise ValueError(f"Unsupported device: {name}")

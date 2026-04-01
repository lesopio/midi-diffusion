from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .config import get_config
from .data import PolyphonicMidiDataset, collate_dynamic_length, LengthBucketBatchSampler, estimate_token_length
from .data import (
    PAD_TOKEN,
    PITCH_OFFSET,
    PITCH_RANGE,
    scan_midi_files,
)
from .diffusion import DiscreteDiffusion
from .model import PolyphonicModel
from .utils import setup_distributed, start_dashboard_server, broadcast_metrics


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_aux_losses(phrase_logits: torch.Tensor,
                       velocity: torch.Tensor,
                       tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    b, t_steps, _ = tokens.shape
    pad_mask = (tokens == PAD_TOKEN).all(dim=2)

    active_mask = tokens >= PITCH_OFFSET
    pitch_vals = torch.where(active_mask, tokens - PITCH_OFFSET, torch.full_like(tokens, -1))
    max_pitch, _ = pitch_vals.max(dim=2)

    phrase_target = torch.zeros((b, t_steps), dtype=torch.long, device=tokens.device)
    nonzero = max_pitch >= 0
    low_th = PITCH_RANGE // 3
    high_th = 2 * (PITCH_RANGE // 3)
    phrase_target[nonzero & (max_pitch <= low_th)] = 1
    phrase_target[nonzero & (max_pitch > low_th) & (max_pitch <= high_th)] = 2
    phrase_target[nonzero & (max_pitch > high_th)] = 3

    ignore_index = -100
    phrase_target[pad_mask] = ignore_index

    phrase_loss = F.cross_entropy(
        phrase_logits.reshape(-1, 4),
        phrase_target.reshape(-1),
        ignore_index=ignore_index,
    )

    velocity_target = (~pad_mask & active_mask.any(dim=2)).float()
    velocity_loss = F.mse_loss(velocity, velocity_target)

    return phrase_loss, velocity_loss


def train_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="4090", choices=["4090", "4050"])
    parser.add_argument("--data-dir", type=str, default="data/midi")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--steps-per-beat", type=int, default=0)
    parser.add_argument("--max-voices", type=int, default=0)
    parser.add_argument("--aux-weight", type=float, default=0.2)
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--save-every-steps", type=int, default=500)
    parser.add_argument("--dynamic-length", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=0)
    parser.add_argument("--bucket-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--val-every-epochs", type=int, default=1)
    parser.add_argument("--eval-data-dirs", type=str, default="")
    parser.add_argument("--eval-max-samples", type=int, default=0)
    parser.add_argument("--metrics-file", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = get_config(args.device)
    batch_size = args.batch_size or cfg.batch_size
    steps_per_beat = args.steps_per_beat or cfg.steps_per_beat
    max_voices = args.max_voices or cfg.max_voices
    max_seq_len = args.max_seq_len or cfg.max_seq_len or cfg.seq_len
    dynamic_length = args.dynamic_length or cfg.dynamic_length

    rank, world_size, local_rank, device, is_distributed = setup_distributed()
    use_amp = cfg.use_amp and device.type == "cuda"

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)

    if rank == 0 and args.dashboard:
        start_dashboard_server()

    set_seed(args.seed + rank)

    if rank == 0:
        valid_files, stats = scan_midi_files(args.data_dir)
        if len(valid_files) == 0:
            raise RuntimeError(f"No valid MIDI files found in {args.data_dir}")
        file_list_path = os.path.join(args.save_dir, "valid_files.txt")
        with open(file_list_path, "w", encoding="utf-8") as f:
            for fp in valid_files:
                f.write(fp + "\n")

    if is_distributed:
        torch.distributed.barrier()

    file_list_path = os.path.join(args.save_dir, "valid_files.txt")
    with open(file_list_path, "r", encoding="utf-8") as f:
        valid_files = [line.strip() for line in f if line.strip()]

    rng = random.Random(args.seed)
    shuffled_files = valid_files[:]
    rng.shuffle(shuffled_files)
    val_count = int(len(shuffled_files) * max(0.0, min(0.5, args.val_split)))
    val_files = shuffled_files[:val_count]
    train_files = shuffled_files[val_count:] if val_count > 0 else shuffled_files

    def _fingerprint_files(paths: list[str]) -> str:
        h = hashlib.sha1()
        for p in paths:
            try:
                st = os.stat(p)
                h.update(p.encode("utf-8", "ignore"))
                h.update(str(st.st_size).encode("utf-8"))
                h.update(str(int(st.st_mtime)).encode("utf-8"))
            except Exception:
                h.update(p.encode("utf-8", "ignore"))
        return h.hexdigest()

    def _load_lengths_cache(cache_path: str, fingerprint: str, steps_per_beat: int) -> tuple[list[int], int | None]:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("fingerprint") != fingerprint:
                return [], None
            if data.get("steps_per_beat") != steps_per_beat:
                return [], None
            lengths = data.get("lengths")
            if not isinstance(lengths, list) or not lengths:
                return [], None
            rec = data.get("recommended_max_seq_len")
            return [int(x) for x in lengths], int(rec) if rec is not None else None
        except Exception:
            return [], None

    def _save_lengths_cache(cache_path: str,
                            fingerprint: str,
                            steps_per_beat: int,
                            lengths: list[int],
                            recommended_max_seq_len: int) -> None:
        payload = {
            "fingerprint": fingerprint,
            "steps_per_beat": steps_per_beat,
            "lengths": lengths,
            "recommended_max_seq_len": recommended_max_seq_len,
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def _recommend_max_seq_len(lengths: list[int], base_len: int, bucket_size: int) -> int:
        if not lengths:
            return base_len
        p95 = float(np.percentile(lengths, 95))
        rec = max(base_len, int(p95))
        if bucket_size > 1:
            rec = int(math.ceil(rec / bucket_size) * bucket_size)
        return max(1, rec)

    lengths_full: list[int] = []
    if dynamic_length:
        fingerprint = _fingerprint_files(valid_files)
        cache_path = os.path.join(args.save_dir, f"lengths_cache_{fingerprint[:12]}_{steps_per_beat}.json")
        if rank == 0:
            lengths_full, cached_rec = _load_lengths_cache(cache_path, fingerprint, steps_per_beat)
            if not lengths_full:
                lengths_full = [
                    estimate_token_length(fp, steps_per_beat=steps_per_beat, max_len=None)
                    for fp in valid_files
                ]
                recommended = _recommend_max_seq_len(lengths_full, max_seq_len, args.bucket_size)
                _save_lengths_cache(cache_path, fingerprint, steps_per_beat, lengths_full, recommended)
            else:
                recommended = cached_rec or _recommend_max_seq_len(lengths_full, max_seq_len, args.bucket_size)
        if is_distributed:
            torch.distributed.barrier()
        if rank != 0:
            lengths_full, cached_rec = _load_lengths_cache(cache_path, fingerprint, steps_per_beat)
            if not lengths_full:
                lengths_full = [
                    estimate_token_length(fp, steps_per_beat=steps_per_beat, max_len=None)
                    for fp in valid_files
                ]
                cached_rec = _recommend_max_seq_len(lengths_full, max_seq_len, args.bucket_size)
            recommended = cached_rec or _recommend_max_seq_len(lengths_full, max_seq_len, args.bucket_size)

        if args.max_seq_len == 0:
            max_seq_len = recommended
            if rank == 0:
                print(f"[INFO] Auto max_seq_len set to {max_seq_len}")

    dataset = PolyphonicMidiDataset(
        midi_dir=args.data_dir,
        file_list=train_files,
        seq_len=cfg.seq_len,
        steps_per_beat=steps_per_beat,
        max_voices=max_voices,
        preload=cfg.preload_data,
        augmentation=True,
        dynamic_length=dynamic_length,
        max_len=max_seq_len,
    )

    if dynamic_length:
        if lengths_full:
            length_map = {fp: min(l, max_seq_len) for fp, l in zip(valid_files, lengths_full)}
            lengths = [length_map.get(fp, max_seq_len) for fp in train_files]
        else:
            lengths = [
                estimate_token_length(fp, steps_per_beat=steps_per_beat, max_len=max_seq_len)
                for fp in train_files
            ]
        bucket_sampler = LengthBucketBatchSampler(
            lengths=lengths,
            batch_size=batch_size,
            bucket_size=args.bucket_size,
            shuffle=True,
            drop_last=True,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=bucket_sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_dynamic_length,
        )
    elif is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            drop_last=True,
            pin_memory=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    vocab_size = PITCH_RANGE + PITCH_OFFSET
    model = PolyphonicModel(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        max_len=max_seq_len,
        max_voices=max_voices,
        diffusion_steps=cfg.diffusion_steps,
        num_conditions=0,
    ).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    diffusion = DiscreteDiffusion(
        num_classes=vocab_size,
        num_steps=cfg.diffusion_steps,
        device=device,
        schedule="cosine",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=2e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    global_step = 0
    latest_ckpt = os.path.join(args.save_dir, "checkpoint_latest.pt")

    config_snapshot = {
        **cfg.__dict__,
        "batch_size": batch_size,
        "max_voices": max_voices,
        "steps_per_beat": steps_per_beat,
        "max_seq_len": max_seq_len,
        "dynamic_length": dynamic_length,
    }

    if args.resume and os.path.exists(latest_ckpt):
        checkpoint = torch.load(latest_ckpt, map_location=device)
        if is_distributed:
            model.module.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if use_amp and "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = checkpoint.get("epoch", 1) + 1
        global_step = checkpoint.get("step", 0)
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        else:
            scheduler.last_epoch = start_epoch - 1

    total_steps = args.epochs * len(dataloader)
    start_time = time.time()
    total_batches = len(dataloader)
    last_group = total_batches % cfg.grad_accum_steps
    last_step_time = time.time()
    last_throughput = 0.0
    metrics_file = args.metrics_file if rank == 0 and args.metrics_file else ""

    def build_eval_loader(file_list: list[str]) -> DataLoader | None:
        if not file_list:
            return None
        eval_dataset = PolyphonicMidiDataset(
            midi_dir=args.data_dir,
            file_list=file_list,
            seq_len=cfg.seq_len,
            steps_per_beat=steps_per_beat,
            max_voices=max_voices,
            preload=False,
            augmentation=False,
            dynamic_length=dynamic_length,
            max_len=max_seq_len,
        )
        if dynamic_length:
            if lengths_full:
                length_map = {fp: min(l, max_seq_len) for fp, l in zip(valid_files, lengths_full)}
                eval_lengths = [length_map.get(fp, max_seq_len) for fp in file_list]
            else:
                eval_lengths = [
                    estimate_token_length(fp, steps_per_beat=steps_per_beat, max_len=max_seq_len)
                    for fp in file_list
                ]
            eval_sampler = LengthBucketBatchSampler(
                lengths=eval_lengths,
                batch_size=batch_size,
                bucket_size=args.bucket_size,
                shuffle=False,
                drop_last=False,
                seed=args.seed,
                rank=0,
                world_size=1,
            )
            return DataLoader(
                eval_dataset,
                batch_sampler=eval_sampler,
                num_workers=0,
                pin_memory=True,
                collate_fn=collate_dynamic_length,
            )
        return DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=True,
        )

    def eval_on_loader(eval_loader: DataLoader) -> dict[str, float]:
        model.eval()
        total_diff = 0.0
        total_phrase = 0.0
        total_vel = 0.0
        count = 0
        max_samples = args.eval_max_samples if args.eval_max_samples > 0 else None
        with torch.no_grad():
            for batch in eval_loader:
                if dynamic_length:
                    batch_tokens, _, pad_mask = batch
                    batch_tokens = batch_tokens.to(device)
                    pad_mask = pad_mask.to(device)
                else:
                    batch_tokens = batch.to(device)
                    pad_mask = None
                t = diffusion.sample_timesteps(batch_tokens.size(0))
                xt = diffusion.q_sample(batch_tokens, t, pad_token=PAD_TOKEN)
                note_logits, phrase_logits, velocity = model(xt, t, None, pad_mask=pad_mask)
                diff_loss = diffusion.loss(note_logits, batch_tokens, pad_token=PAD_TOKEN)
                phrase_loss, velocity_loss = compute_aux_losses(phrase_logits, velocity, batch_tokens)
                total_diff += float(diff_loss.item())
                total_phrase += float(phrase_loss.item())
                total_vel += float(velocity_loss.item())
                count += 1
                if max_samples is not None and count >= max_samples:
                    break
        if count == 0:
            return {"diff": 0.0, "phrase": 0.0, "vel": 0.0}
        return {
            "diff": total_diff / count,
            "phrase": total_phrase / count,
            "vel": total_vel / count,
        }

    val_loader = build_eval_loader(val_files) if rank == 0 else None
    eval_dirs = [d.strip() for d in args.eval_data_dirs.split(",") if d.strip()]
    eval_loaders: dict[str, DataLoader] = {}
    if rank == 0 and eval_dirs:
        for d in eval_dirs:
            try:
                eval_valid, _ = scan_midi_files(d)
                if eval_valid:
                    eval_loaders[d] = build_eval_loader(eval_valid)
            except Exception:
                pass

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        if dynamic_length:
            bucket_sampler.set_epoch(epoch)
        elif is_distributed:
            sampler.set_epoch(epoch)

        if rank == 0:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        else:
            pbar = dataloader

        epoch_losses = []
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(pbar):
            if dynamic_length:
                batch_tokens, _, pad_mask = batch
                batch_tokens = batch_tokens.to(device)
                pad_mask = pad_mask.to(device)
            else:
                batch_tokens = batch.to(device)
                pad_mask = None
            if last_group != 0 and batch_idx >= total_batches - last_group:
                accum_scale = last_group
            else:
                accum_scale = cfg.grad_accum_steps

            with torch.amp.autocast("cuda", enabled=use_amp):
                t = diffusion.sample_timesteps(batch_tokens.size(0))
                xt = diffusion.q_sample(batch_tokens, t, pad_token=PAD_TOKEN)

                note_logits, phrase_logits, velocity = model(xt, t, None, pad_mask=pad_mask)
                diff_loss = diffusion.loss(note_logits, batch_tokens, pad_token=PAD_TOKEN)
                phrase_loss, velocity_loss = compute_aux_losses(phrase_logits, velocity, batch_tokens)
                loss = diff_loss + args.aux_weight * (phrase_loss + velocity_loss)
                loss = loss / accum_scale

            scaler.scale(loss).backward()

            if (batch_idx + 1) % cfg.grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                now = time.time()
                step_time = now - last_step_time
                last_step_time = now
                samples_per_step = batch_tokens.size(0) * accum_scale * (world_size if is_distributed else 1)
                if step_time > 0:
                    last_throughput = samples_per_step / step_time

                loss_value = loss.item() * accum_scale
                epoch_losses.append(loss_value)

                if rank == 0:
                    pbar.set_postfix({
                        "loss": f"{loss_value:.4f}",
                        "diff": f"{diff_loss.item():.4f}",
                        "phrase": f"{phrase_loss.item():.4f}",
                        "vel": f"{velocity_loss.item():.4f}",
                    })

                    if (args.dashboard or metrics_file) and global_step % 5 == 0:
                        elapsed = time.time() - start_time
                        avg_per_step = elapsed / max(1, global_step)
                        remaining = avg_per_step * max(0, total_steps - global_step)
                        lr = optimizer.param_groups[0]["lr"]
                        if device.type == "cuda":
                            mem_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
                        else:
                            mem_gb = 0.0

                        def fmt(sec: float) -> str:
                            if sec >= 3600:
                                return f"{int(sec//3600)}h{int((sec%3600)//60)}m"
                            if sec >= 60:
                                return f"{int(sec//60)}m{int(sec%60)}s"
                            return f"{int(sec)}s"

                        payload = {
                            "epoch": epoch,
                            "elapsed": int(elapsed),
                            "elapsed_str": fmt(elapsed),
                            "remaining": int(remaining),
                            "remaining_str": fmt(remaining),
                            "loss": float(loss_value),
                            "diffusion_loss": float(diff_loss.item()),
                            "lr": float(lr),
                            "throughput": float(last_throughput),
                            "vram_gb": float(mem_gb),
                            "step": global_step,
                        }
                        if args.dashboard:
                            broadcast_metrics(payload)
                        if metrics_file:
                            try:
                                with open(metrics_file, "a", encoding="utf-8") as f:
                                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                            except Exception:
                                pass

                if rank == 0 and args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                    model_to_save = model.module if is_distributed else model
                    ckpt = {
                        "model_state": model_to_save.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "epoch": epoch,
                        "step": global_step,
                        "config": config_snapshot,
                    }
                    torch.save(ckpt, os.path.join(args.save_dir, f"checkpoint_step_{global_step}.pt"))
                    torch.save(ckpt, latest_ckpt)

        scheduler.step()

        if rank == 0:
            avg_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))
            print(f"Epoch {epoch} finished. Avg loss {avg_loss:.4f}")
            if args.val_every_epochs > 0 and val_loader and epoch % args.val_every_epochs == 0:
                val_metrics = eval_on_loader(val_loader)
                print(
                    f"[VAL] diff={val_metrics['diff']:.4f} "
                    f"phrase={val_metrics['phrase']:.4f} vel={val_metrics['vel']:.4f}"
                )
            for name, loader in eval_loaders.items():
                if loader and args.val_every_epochs > 0 and epoch % args.val_every_epochs == 0:
                    metrics = eval_on_loader(loader)
                    print(
                        f"[EVAL:{name}] diff={metrics['diff']:.4f} "
                        f"phrase={metrics['phrase']:.4f} vel={metrics['vel']:.4f}"
                    )
            model_to_save = model.module if is_distributed else model
            ckpt = {
                "model_state": model_to_save.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch,
                "step": global_step,
                "config": config_snapshot,
            }
            torch.save(ckpt, os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pt"))
            torch.save(ckpt, latest_ckpt)

        if is_distributed:
            torch.distributed.barrier()

    if rank == 0:
        print("Training finished.")

    if is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train_main()

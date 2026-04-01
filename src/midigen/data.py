from __future__ import annotations

import hashlib
import os
import random
from typing import List, Tuple, Optional, Iterable

import numpy as np
import pretty_midi
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

PITCH_MIN = 36
PITCH_MAX = 96
PITCH_RANGE = PITCH_MAX - PITCH_MIN

PAD_TOKEN = 0
REST_TOKEN = 1
PITCH_OFFSET = 2


class MusicAugmentation:
    """Simple augmentation for token sequences."""

    def __init__(self, transpose_range: int = 5, tempo_range: float = 0.2,
                 pad_token: int = PAD_TOKEN, rest_token: int = REST_TOKEN,
                 pitch_offset: int = PITCH_OFFSET, pitch_range: int = PITCH_RANGE):
        self.transpose_range = transpose_range
        self.tempo_range = tempo_range
        self.pad_token = pad_token
        self.rest_token = rest_token
        self.pitch_offset = pitch_offset
        self.pitch_range = pitch_range

    def __call__(self, tokens: torch.Tensor, aug_prob: float = 0.5) -> torch.Tensor:
        if random.random() > aug_prob:
            return tokens

        aug_type = random.choice(["transpose", "time_stretch"])
        if aug_type == "transpose":
            shift = random.randint(-self.transpose_range, self.transpose_range)
            augmented = tokens.clone()
            mask = augmented >= self.pitch_offset
            augmented[mask] = torch.clamp(
                augmented[mask] + shift,
                self.pitch_offset,
                self.pitch_offset + self.pitch_range - 1,
            )
            return augmented

        if aug_type == "time_stretch":
            ratio = random.uniform(1 - self.tempo_range, 1 + self.tempo_range)
            new_len = max(1, int(tokens.size(0) * ratio))
            augmented = F.interpolate(
                tokens.T.unsqueeze(0).float(),
                size=new_len,
                mode="nearest",
            )[0].T.long()

            if augmented.size(0) > tokens.size(0):
                return augmented[: tokens.size(0)]

            pad_len = tokens.size(0) - augmented.size(0)
            padding = torch.full(
                (pad_len, tokens.size(1)),
                self.pad_token,
                dtype=torch.long,
                device=augmented.device,
            )
            return torch.cat([augmented, padding], dim=0)

        return tokens


def validate_midi_header(path: str, max_size_mb: int = 10) -> bool:
    try:
        size = os.path.getsize(path)
        if size <= 0 or size > max_size_mb * 1024 * 1024:
            return False
        with open(path, "rb") as f:
            header = f.read(12)
        if len(header) < 4:
            return False
        if header[:4] == b"MThd":
            return True
        if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"RMID":
            return True
        return False
    except Exception:
        return False


def scan_midi_files(midi_dir: str) -> Tuple[List[str], dict]:
    files: List[str] = []
    for root, dirs, filenames in os.walk(midi_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in filenames:
            if fn.lower().endswith((".mid", ".midi")):
                files.append(os.path.join(root, fn))
    valid = [f for f in files if validate_midi_header(f)]
    return valid, {"total": len(files), "valid": len(valid)}


def load_midi(path: str) -> pretty_midi.PrettyMIDI:
    return pretty_midi.PrettyMIDI(path)


def build_time_grid(pm: pretty_midi.PrettyMIDI, steps_per_beat: int) -> np.ndarray:
    end_time = max(0.001, pm.get_end_time())
    beats = pm.get_beats()
    if beats is None or len(beats) < 2:
        tempo = pm.estimate_tempo() if pm is not None else 120.0
        tempo = tempo if tempo and tempo > 0 else 120.0
        beat_dur = 60.0 / tempo
        beats = np.arange(0, end_time + beat_dur, beat_dur, dtype=np.float32)
    else:
        beats = np.asarray(beats, dtype=np.float32)
        if beats[-1] < end_time:
            beats = np.concatenate([beats, np.array([end_time], dtype=np.float32)])

    grid = []
    for i in range(len(beats) - 1):
        start = float(beats[i])
        end = float(beats[i + 1])
        step = (end - start) / steps_per_beat
        for s in range(steps_per_beat):
            grid.append(start + s * step)
    grid.append(float(beats[-1]))
    return np.asarray(grid, dtype=np.float32)


def estimate_token_length(path: str, steps_per_beat: int, max_len: Optional[int] = None) -> int:
    try:
        pm = load_midi(path)
        grid = build_time_grid(pm, steps_per_beat)
        steps = max(1, len(grid) - 1)
        if max_len is None:
            return steps
        return min(steps, max_len)
    except Exception:
        return max_len if max_len is not None else 0


def midi_to_pianoroll(pm: pretty_midi.PrettyMIDI,
                      steps_per_beat: int,
                      pitch_min: int = PITCH_MIN,
                      pitch_max: int = PITCH_MAX) -> np.ndarray:
    grid = build_time_grid(pm, steps_per_beat)
    steps = max(1, len(grid) - 1)
    pr = np.zeros((steps, pitch_max - pitch_min), dtype=np.uint8)

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            if not (pitch_min <= note.pitch < pitch_max):
                continue
            start = np.searchsorted(grid, note.start, side="right") - 1
            end = np.searchsorted(grid, note.end, side="left")
            start = int(np.clip(start, 0, steps - 1))
            end = int(np.clip(end, start + 1, steps))
            pr[start:end, note.pitch - pitch_min] = 1
    return pr


def select_pitches(active: List[int], max_voices: int) -> List[int]:
    if len(active) <= max_voices:
        return active
    idxs = np.linspace(0, len(active) - 1, max_voices).astype(int)
    return [active[i] for i in idxs]


def assign_voices(active_pitches: List[int], prev_voice_pitches: List[int | None], max_voices: int) -> dict:
    if not active_pitches:
        return {}

    active_pitches = sorted(active_pitches)
    if len(active_pitches) > max_voices:
        active_pitches = select_pitches(active_pitches, max_voices)

    pitch_ranks = {p: i for i, p in enumerate(active_pitches)}

    pairs = []
    for v in range(max_voices):
        prev = prev_voice_pitches[v]
        for p in active_pitches:
            base = abs(prev - p) if prev is not None else 6.0
            order_penalty = 0.5 * abs(pitch_ranks[p] - v)
            cost = base + order_penalty
            pairs.append((cost, v, p))
    pairs.sort(key=lambda x: x[0])

    assigned = {}
    used_voices = set()
    used_pitches = set()
    for _, v, p in pairs:
        if v in used_voices or p in used_pitches:
            continue
        assigned[v] = p
        used_voices.add(v)
        used_pitches.add(p)
        if len(used_voices) >= min(max_voices, len(active_pitches)):
            break
    return assigned


def pianoroll_to_polyphonic_tokens(pr: np.ndarray,
                                   max_voices: int,
                                   max_len: int,
                                   pitch_min: int = PITCH_MIN,
                                   pitch_max: int = PITCH_MAX,
                                   pad_token: int = PAD_TOKEN,
                                   rest_token: int = REST_TOKEN,
                                   pitch_offset: int = PITCH_OFFSET) -> np.ndarray:
    t_steps, _ = pr.shape
    length = min(t_steps, max_len)
    tokens = np.full((length, max_voices), pad_token, dtype=np.int64)

    prev_voice_pitches: List[int | None] = [None] * max_voices

    for t in range(length):
        active_idx = np.where(pr[t] > 0)[0].tolist()
        if not active_idx:
            tokens[t, :] = rest_token
            prev_voice_pitches = [None] * max_voices
            continue

        active_midi = [pitch_min + i for i in active_idx]
        assignment = assign_voices(active_midi, prev_voice_pitches, max_voices)

        tokens[t, :] = rest_token
        for v, midi_pitch in assignment.items():
            pitch_idx = midi_pitch - pitch_min
            tokens[t, v] = pitch_offset + pitch_idx

        for v in range(max_voices):
            prev_voice_pitches[v] = assignment.get(v)

    return tokens


def tokens_to_pretty_midi(tokens: np.ndarray,
                           steps_per_beat: int,
                           tempo: float = 120.0,
                           pitch_min: int = PITCH_MIN,
                           pad_token: int = PAD_TOKEN,
                           rest_token: int = REST_TOKEN,
                           pitch_offset: int = PITCH_OFFSET) -> pretty_midi.PrettyMIDI:
    step_dur = (60.0 / tempo) / steps_per_beat
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    if tokens.ndim == 1:
        tokens = tokens[:, None]
    elif tokens.ndim != 2:
        raise ValueError("tokens must be 1D (time) or 2D (time, voices)")

    t_steps, voices = tokens.shape
    for v in range(voices):
        inst = pretty_midi.Instrument(program=0)
        current_pitch = None
        start_t = 0.0
        for t in range(t_steps + 1):
            token = None
            if t < t_steps:
                token = int(tokens[t, v])
            if t == t_steps or token in (pad_token, rest_token) or token is None:
                if current_pitch is not None:
                    end_t = t * step_dur
                    inst.notes.append(
                        pretty_midi.Note(
                            velocity=100,
                            pitch=current_pitch,
                            start=start_t,
                            end=end_t,
                        )
                    )
                    current_pitch = None
                continue
            pitch_idx = token - pitch_offset
            midi_pitch = pitch_min + pitch_idx
            if current_pitch is None:
                current_pitch = midi_pitch
                start_t = t * step_dur
            elif midi_pitch != current_pitch:
                end_t = t * step_dur
                inst.notes.append(
                    pretty_midi.Note(
                        velocity=100,
                        pitch=current_pitch,
                        start=start_t,
                        end=end_t,
                    )
                )
                current_pitch = midi_pitch
                start_t = t * step_dur

        pm.instruments.append(inst)

    return pm


def cache_key(path: str, seq_len: int, steps_per_beat: int, max_voices: int) -> str:
    token = f"{path}|{seq_len}|{steps_per_beat}|{max_voices}|{PITCH_MIN}|{PITCH_MAX}"
    return hashlib.sha1(token.encode("utf-8")).hexdigest()


def cache_key_v2(path: str, seq_len: int, steps_per_beat: int, max_voices: int,
                 dynamic_length: bool, max_len: int) -> str:
    token = (
        f"{path}|{seq_len}|{steps_per_beat}|{max_voices}|{PITCH_MIN}|{PITCH_MAX}|"
        f"dynamic={int(dynamic_length)}|max_len={max_len}"
    )
    return hashlib.sha1(token.encode("utf-8")).hexdigest()


def atomic_save_tensor(tensor: torch.Tensor, path: str) -> None:
    tmp_path = f"{path}.tmp_{os.getpid()}_{random.randint(0, 1_000_000)}"
    torch.save(tensor, tmp_path)
    os.replace(tmp_path, path)


def collate_dynamic_length(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens_list = [item[0] for item in batch]
    lengths = torch.tensor([item[1] for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    voices = tokens_list[0].size(1)
    padded = torch.full((len(tokens_list), max_len, voices), PAD_TOKEN, dtype=torch.long)
    for i, tokens in enumerate(tokens_list):
        padded[i, : tokens.size(0), :] = tokens
    pad_mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)
    return padded, lengths, pad_mask


class LengthBucketBatchSampler(Sampler[List[int]]):
    def __init__(self,
                 lengths: List[int],
                 batch_size: int,
                 bucket_size: int = 64,
                 shuffle: bool = True,
                 drop_last: bool = True,
                 seed: int = 42,
                 rank: int = 0,
                 world_size: int = 1):
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_size = max(1, bucket_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        self._batches: List[List[int]] = []

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._batches = []

    def _bucket_indices(self) -> Iterable[List[int]]:
        indices = list(range(len(self.lengths)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(indices)

        buckets: dict[int, List[int]] = {}
        for idx in indices:
            length = self.lengths[idx]
            bucket_id = int(length // self.bucket_size)
            buckets.setdefault(bucket_id, []).append(idx)

        bucket_keys = list(buckets.keys())
        if self.shuffle:
            rng.shuffle(bucket_keys)

        batches: List[List[int]] = []
        for key in bucket_keys:
            bucket = buckets[key]
            if self.shuffle:
                rng.shuffle(bucket)
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i: i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        if self.world_size > 1 and batches:
            remainder = len(batches) % self.world_size
            if remainder != 0:
                pad = self.world_size - remainder
                batches.extend(batches[:pad])

        self._batches = batches
        return self._batches

    def __iter__(self):
        self._bucket_indices()
        for i in range(self.rank, len(self._batches), self.world_size):
            yield self._batches[i]

    def __len__(self) -> int:
        if not self._batches:
            self._bucket_indices()
        if self.world_size <= 1:
            return len(self._batches)
        return len(self._batches) // self.world_size


class PolyphonicMidiDataset(Dataset):
    def __init__(self,
                 midi_dir: Optional[str] = None,
                 file_list: Optional[List[str]] = None,
                 seq_len: int = 512,
                 steps_per_beat: int = 4,
                 max_voices: int = 4,
                 preload: bool = False,
                 augmentation: bool = True,
                 dynamic_length: bool = False,
                 max_len: Optional[int] = None):
        if file_list is None:
            if midi_dir is None:
                raise ValueError("midi_dir or file_list must be provided")
            file_list = []
            for root, _, files in os.walk(midi_dir):
                for fn in files:
                    if fn.lower().endswith((".mid", ".midi")):
                        file_list.append(os.path.join(root, fn))
        self.files = list(file_list)
        self.seq_len = seq_len
        self.steps_per_beat = steps_per_beat
        self.max_voices = max_voices
        self.preload = preload
        self.dynamic_length = dynamic_length
        self.max_len = max_len or seq_len

        self.cache_dir = os.path.join(midi_dir or ".", ".poly_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.augmentation = None
        if augmentation:
            self.augmentation = MusicAugmentation(
                pad_token=PAD_TOKEN,
                rest_token=REST_TOKEN,
                pitch_offset=PITCH_OFFSET,
                pitch_range=PITCH_RANGE,
            )

    def __len__(self) -> int:
        return len(self.files)

    def _process_file(self, path: str) -> torch.Tensor:
        pm = load_midi(path)
        pr = midi_to_pianoroll(
            pm,
            steps_per_beat=self.steps_per_beat,
            pitch_min=PITCH_MIN,
            pitch_max=PITCH_MAX,
        )
        tokens = pianoroll_to_polyphonic_tokens(
            pr,
            max_voices=self.max_voices,
            max_len=self.max_len,
            pitch_min=PITCH_MIN,
            pitch_max=PITCH_MAX,
        )

        tokens = torch.from_numpy(tokens).long()

        if self.dynamic_length:
            return tokens

        if tokens.shape[0] < self.seq_len:
            pad_len = self.seq_len - tokens.shape[0]
            padding = torch.full(
                (pad_len, self.max_voices),
                PAD_TOKEN,
                dtype=torch.long,
            )
            tokens = torch.cat([tokens, padding], dim=0)
        return tokens

    def _empty_tokens(self) -> torch.Tensor:
        length = self.max_len if self.dynamic_length else self.seq_len
        return torch.full((length, self.max_voices), PAD_TOKEN, dtype=torch.long)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        key = cache_key_v2(
            path,
            self.seq_len,
            self.steps_per_beat,
            self.max_voices,
            self.dynamic_length,
            self.max_len,
        )
        cache_path = os.path.join(self.cache_dir, f"{key}.pt")
        processed_ok = True
        try:
            if self.preload and os.path.exists(cache_path):
                tokens = torch.load(cache_path)
            else:
                tokens = self._process_file(path)
        except Exception as exc:
            processed_ok = False
            print(f"[WARN] Failed to process MIDI: {path} ({exc})")
            tokens = self._empty_tokens()

        if self.preload and processed_ok and not os.path.exists(cache_path):
            try:
                atomic_save_tensor(tokens, cache_path)
            except Exception as exc:
                print(f"[WARN] Failed to write cache: {cache_path} ({exc})")

        if self.augmentation is not None:
            tokens = self.augmentation(tokens)
        if self.dynamic_length:
            return tokens, int(tokens.size(0))
        return tokens

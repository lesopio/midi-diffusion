from __future__ import annotations

import math
import torch
import torch.nn as nn


class PolyphonicTransformerEncoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 nhead: int,
                 num_layers: int,
                 max_len: int,
                 max_voices: int,
                 diffusion_steps: int,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.max_voices = max_voices

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.time_emb = nn.Embedding(max_len, d_model)
        self.voice_emb = nn.Embedding(max_voices, d_model)
        self.t_emb = nn.Embedding(diffusion_steps, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self,
                tokens: torch.Tensor,
                t: torch.Tensor | None = None,
                cond: torch.Tensor | None = None,
                pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t_steps, v = tokens.shape
        if v > self.max_voices:
            tokens = tokens[:, :, : self.max_voices]
            v = self.max_voices

        x = self.token_emb(tokens) * math.sqrt(self.d_model)

        time_ids = torch.arange(t_steps, device=tokens.device)
        voice_ids = torch.arange(v, device=tokens.device)
        x = x + self.time_emb(time_ids)[None, :, None, :]
        x = x + self.voice_emb(voice_ids)[None, None, :, :]

        if t is not None:
            x = x + self.t_emb(t)[:, None, None, :]

        if cond is not None:
            x = x + cond[:, None, None, :]

        x = x.view(b, t_steps * v, self.d_model)
        key_padding_mask = None
        if pad_mask is not None:
            if pad_mask.size(1) != t_steps:
                raise ValueError("pad_mask length mismatch with tokens")
            key_padding_mask = pad_mask[:, :, None].expand(b, t_steps, v).reshape(b, t_steps * v)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        x = x.view(b, t_steps, v, self.d_model)
        return x


class ConditionalHead(nn.Module):
    def __init__(self, num_conditions: int, d_model: int):
        super().__init__()
        self.cond_embed = nn.Embedding(num_conditions + 1, d_model)
        self.null_id = num_conditions

    def forward(self, cond_ids: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if cond_ids is None:
            cond_ids = torch.full((batch_size,), self.null_id, dtype=torch.long, device=device)
        return self.cond_embed(cond_ids)


class NoteHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PhraseHead(nn.Module):
    def __init__(self, d_model: int, n_classes: int = 4):
        super().__init__()
        self.proj = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class VelocityHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(x)).squeeze(-1)


class PolyphonicModel(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 nhead: int,
                 num_layers: int,
                 max_len: int,
                 max_voices: int,
                 diffusion_steps: int,
                 num_conditions: int = 0):
        super().__init__()
        self.encoder = PolyphonicTransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_len=max_len,
            max_voices=max_voices,
            diffusion_steps=diffusion_steps,
        )
        self.note_head = NoteHead(d_model, vocab_size)
        self.phrase_head = PhraseHead(d_model, n_classes=4)
        self.velocity_head = VelocityHead(d_model)

        self.cond_head = None
        if num_conditions > 0:
            self.cond_head = ConditionalHead(num_conditions, d_model)

    def forward(self,
                tokens: torch.Tensor,
                t: torch.Tensor | None = None,
                cond: torch.Tensor | None = None,
                pad_mask: torch.Tensor | None = None):
        b = tokens.size(0)
        cond_vec = None
        if self.cond_head is not None:
            cond_vec = self.cond_head(cond, batch_size=b, device=tokens.device)

        enc = self.encoder(tokens, t, cond_vec, pad_mask=pad_mask)
        note_logits = self.note_head(enc)
        phrase_logits = self.phrase_head(enc.mean(dim=2))
        velocity = self.velocity_head(enc.mean(dim=2))
        return note_logits, phrase_logits, velocity

"""Counterfactual perturbations for precomputed MusicInfuser audio features."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class FeatureCounterfactual:
    kind: str
    audio_embeddings: torch.Tensor
    original_curve: torch.Tensor
    counterfactual_curve: torch.Tensor
    silence_mask: torch.Tensor


def _as_btd(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if x.ndim == 2:
        return x.unsqueeze(0), True
    if x.ndim == 3:
        return x, False
    raise ValueError(f"Expected audio feature tensor with shape T x D or B x T x D, got {tuple(x.shape)}")


def _restore_shape(x: torch.Tensor, squeezed: bool) -> torch.Tensor:
    return x.squeeze(0) if squeezed else x


def feature_energy_curve(audio_embeddings: torch.Tensor) -> torch.Tensor:
    x, _ = _as_btd(audio_embeddings.float())
    curve = x.abs().mean(dim=-1)
    lo = curve.amin(dim=-1, keepdim=True)
    hi = curve.amax(dim=-1, keepdim=True)
    return (curve - lo) / (hi - lo).clamp_min(1e-6)


def _zero_pad_shift(x: torch.Tensor, shift: int) -> torch.Tensor:
    out = torch.zeros_like(x)
    if shift == 0:
        return x.clone()
    if abs(shift) >= x.shape[1]:
        return out
    if shift > 0:
        out[:, shift:] = x[:, :-shift]
    else:
        out[:, :shift] = x[:, -shift:]
    return out


def _tempo_resample(x: torch.Tensor, rate: float) -> torch.Tensor:
    b, t, d = x.shape
    new_t = max(2, int(round(t / rate)))
    y = x.transpose(1, 2)
    y = F.interpolate(y, size=new_t, mode="linear", align_corners=False)
    if new_t > t:
        start = (new_t - t) // 2
        y = y[:, :, start : start + t]
    elif new_t < t:
        y = F.pad(y, (0, t - new_t))
    return y.transpose(1, 2)


def make_feature_counterfactual(
    audio_embeddings: torch.Tensor,
    *,
    local_silence_prob: float = 0.35,
    shift_prob: float = 0.30,
    tempo_prob: float = 0.25,
    global_silence_prob: float = 0.10,
    local_silence_ratio: float = 0.20,
    max_shift_ratio: float = 0.25,
    tempo_rates: tuple[float, ...] = (0.8, 1.2),
) -> FeatureCounterfactual:
    x, squeezed = _as_btd(audio_embeddings)
    x = x.clone()
    b, t, _d = x.shape
    original_curve = feature_energy_curve(x)
    silence_mask = torch.zeros((b, t), dtype=x.dtype, device=x.device)

    probs = [
        ("local_silence", local_silence_prob),
        ("shift", shift_prob),
        ("tempo", tempo_prob),
        ("global_silence", global_silence_prob),
    ]
    total = sum(p for _k, p in probs)
    r = random.random() * total
    upto = 0.0
    kind = probs[-1][0]
    for name, prob in probs:
        upto += prob
        if r <= upto:
            kind = name
            break

    if kind == "local_silence":
        width = max(1, int(round(t * local_silence_ratio)))
        center = original_curve.argmax(dim=1)
        cf = x.clone()
        for i in range(b):
            start = int(max(0, center[i].item() - width // 2))
            end = int(min(t, start + width))
            cf[i, start:end] = 0
            silence_mask[i, start:end] = 1
    elif kind == "shift":
        max_shift = max(1, int(round(t * max_shift_ratio)))
        shift = random.choice([s for s in range(-max_shift, max_shift + 1) if s != 0])
        cf = _zero_pad_shift(x, shift)
    elif kind == "tempo":
        cf = _tempo_resample(x, random.choice(tempo_rates))
    else:
        cf = torch.zeros_like(x)
        silence_mask.fill_(1)

    counterfactual_curve = feature_energy_curve(cf)
    return FeatureCounterfactual(
        kind=kind,
        audio_embeddings=_restore_shape(cf, squeezed),
        original_curve=original_curve,
        counterfactual_curve=counterfactual_curve,
        silence_mask=silence_mask,
    )

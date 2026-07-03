#!/usr/bin/env python3
"""Torch losses for CAT-LoRA training.

These functions are intentionally backbone-agnostic. The MusicInfuser training
script should call them after obtaining predicted clean video latents from the
denoiser output.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


EPS = 1e-8


def latent_motion_curve(z: torch.Tensor) -> torch.Tensor:
    """Return per-frame latent motion.

    Expected shapes:
      B x T x C x H x W
      B x C x T x H x W

    The function treats the smaller of dim 1/2 as channels when ambiguous by
    preferring B x T x C x H x W if dim 1 is not 3/4.
    """
    if z.ndim != 5:
        raise ValueError(f"Expected 5D video latent tensor, got {tuple(z.shape)}")
    if z.shape[1] in (3, 4) and z.shape[2] > 4:
        z = z.permute(0, 2, 1, 3, 4)
    diff = (z[:, 1:] - z[:, :-1]).abs()
    return diff.flatten(2).mean(dim=2)


def normalize_curve(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=-1, keepdim=True)) / (
        x.std(dim=-1, keepdim=True, unbiased=False) + EPS
    )


def resample_curve(curve: torch.Tensor, target_len: int) -> torch.Tensor:
    if curve.shape[-1] == target_len:
        return curve
    x = curve.unsqueeze(1)
    return F.interpolate(x, size=target_len, mode="linear", align_corners=False).squeeze(1)


def corr_loss(audio_curve: torch.Tensor, motion_curve: torch.Tensor) -> torch.Tensor:
    n = min(audio_curve.shape[-1], motion_curve.shape[-1])
    a = normalize_curve(audio_curve[..., :n])
    m = normalize_curve(motion_curve[..., :n])
    corr = (a * m).mean(dim=-1)
    return (1.0 - corr).mean()


def peak_kl_loss(audio_curve: torch.Tensor, motion_curve: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    n = min(audio_curve.shape[-1], motion_curve.shape[-1])
    a = audio_curve[..., :n]
    m = motion_curve[..., :n]
    p_a = F.softmax(a / temperature, dim=-1)
    log_p_m = F.log_softmax(m / temperature, dim=-1)
    p_m = F.softmax(m / temperature, dim=-1)
    log_p_a = F.log_softmax(a / temperature, dim=-1)
    return 0.5 * (
        F.kl_div(log_p_m, p_a, reduction="batchmean")
        + F.kl_div(log_p_a, p_m, reduction="batchmean")
    )


def delta_response_loss(
    original_audio_curve: torch.Tensor,
    cf_audio_curve: torch.Tensor,
    original_motion_curve: torch.Tensor,
    cf_motion_curve: torch.Tensor,
) -> torch.Tensor:
    n = min(
        original_audio_curve.shape[-1],
        cf_audio_curve.shape[-1],
        original_motion_curve.shape[-1],
        cf_motion_curve.shape[-1],
    )
    delta_audio = cf_audio_curve[..., :n] - original_audio_curve[..., :n]
    delta_motion = cf_motion_curve[..., :n] - original_motion_curve[..., :n]
    return corr_loss(delta_audio, delta_motion)


def silence_suppression_loss(
    original_motion_curve: torch.Tensor,
    cf_motion_curve: torch.Tensor,
    silence_mask: torch.Tensor,
    ratio: float = 0.5,
) -> torch.Tensor:
    n = min(original_motion_curve.shape[-1], cf_motion_curve.shape[-1], silence_mask.shape[-1])
    orig = original_motion_curve[..., :n]
    cf = cf_motion_curve[..., :n]
    mask = (silence_mask[..., :n] > 0.5).float()
    denom = mask.sum(dim=-1).clamp_min(1.0)
    penalty = F.relu(cf - ratio * orig) * mask
    return (penalty.sum(dim=-1) / denom).mean()


def temporal_mean_preserve_loss(student_z: torch.Tensor, teacher_z: torch.Tensor) -> torch.Tensor:
    if student_z.shape[1] in (3, 4) and student_z.shape[2] > 4:
        student_z = student_z.permute(0, 2, 1, 3, 4)
    if teacher_z.shape[1] in (3, 4) and teacher_z.shape[2] > 4:
        teacher_z = teacher_z.permute(0, 2, 1, 3, 4)
    return F.l1_loss(student_z.mean(dim=1), teacher_z.detach().mean(dim=1))


def motion_smoothness_loss(motion_curve: torch.Tensor) -> torch.Tensor:
    if motion_curve.shape[-1] < 2:
        return motion_curve.mean() * 0.0
    return (motion_curve[..., 1:] - motion_curve[..., :-1]).abs().mean()

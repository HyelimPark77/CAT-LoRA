#!/usr/bin/env python3
"""Counterfactual audio transforms used by CAT-LoRA and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass(frozen=True)
class PerturbationSpec:
    name: str
    kind: str
    shift_sec: float = 0.0
    tempo_rate: float = 1.0
    local_silence_sec: float = 0.75
    local_center_sec: float | None = None


def crop_or_pad(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) >= n:
        return y[:n].astype(np.float32)
    return np.pad(y, (0, n - len(y))).astype(np.float32)


def normalize_peak(y: np.ndarray, peak: float = 0.98) -> np.ndarray:
    m = float(np.max(np.abs(y))) if y.size else 0.0
    if m <= 1e-8:
        return y.astype(np.float32)
    return (y * min(1.0, peak / m)).astype(np.float32)


def global_silence(y: np.ndarray) -> np.ndarray:
    return np.zeros_like(y, dtype=np.float32)


def zero_pad_shift(y: np.ndarray, sr: int, shift_sec: float) -> np.ndarray:
    shift = int(round(abs(shift_sec) * sr))
    if shift == 0:
        return y.copy().astype(np.float32)
    if shift_sec > 0:
        out = np.concatenate([np.zeros(shift, dtype=np.float32), y.astype(np.float32)])
        return out[: len(y)]
    out = np.concatenate([y.astype(np.float32)[shift:], np.zeros(shift, dtype=np.float32)])
    return out[: len(y)]


def strongest_onset_center(y: np.ndarray, sr: int, hop_length: int = 512) -> float:
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    if onset.size == 0 or float(np.max(onset)) <= 0:
        return len(y) / (2.0 * sr)
    frame = int(np.argmax(onset))
    return float(librosa.frames_to_time(frame, sr=sr, hop_length=hop_length))


def local_silence(
    y: np.ndarray,
    sr: int,
    duration_sec: float = 0.75,
    center_sec: float | None = None,
    fade_sec: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Mute a short segment with fade ramps.

    Returns the perturbed signal and a boolean mask over samples indicating the
    muted region. The mask supports silence-suppression training losses.
    """
    center = strongest_onset_center(y, sr) if center_sec is None else center_sec
    half = duration_sec / 2.0
    start = max(0, int(round((center - half) * sr)))
    end = min(len(y), int(round((center + half) * sr)))
    out = y.copy().astype(np.float32)
    mask = np.zeros(len(y), dtype=bool)
    if end <= start:
        return out, mask
    mask[start:end] = True
    out[start:end] = 0.0

    fade = int(round(fade_sec * sr))
    if fade > 0:
        left0 = max(0, start - fade)
        if start > left0:
            ramp = np.linspace(1.0, 0.0, start - left0, endpoint=False)
            out[left0:start] *= ramp.astype(np.float32)
        right1 = min(len(y), end + fade)
        if right1 > end:
            ramp = np.linspace(0.0, 1.0, right1 - end, endpoint=False)
            out[end:right1] *= ramp.astype(np.float32)
    return out, mask


def tempo_change(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    stretched = librosa.effects.time_stretch(y.astype(np.float32), rate=rate)
    return normalize_peak(crop_or_pad(stretched, len(y)))


def apply_perturbation(
    y: np.ndarray,
    sr: int,
    spec: PerturbationSpec,
) -> tuple[np.ndarray, np.ndarray | None]:
    if spec.kind == "original":
        return y.copy().astype(np.float32), None
    if spec.kind == "silence":
        return global_silence(y), np.ones(len(y), dtype=bool)
    if spec.kind == "local_silence":
        return local_silence(
            y,
            sr,
            duration_sec=spec.local_silence_sec,
            center_sec=spec.local_center_sec,
        )
    if spec.kind == "shift":
        return zero_pad_shift(y, sr, spec.shift_sec), None
    if spec.kind == "tempo":
        return tempo_change(y, sr, spec.tempo_rate), None
    raise ValueError(f"Unsupported perturbation kind: {spec.kind}")


DEFAULT_TRAIN_SPECS = [
    PerturbationSpec(name="local_silence", kind="local_silence", local_silence_sec=0.75),
    PerturbationSpec(name="shift_pos_0p5", kind="shift", shift_sec=0.5),
    PerturbationSpec(name="shift_neg_0p5", kind="shift", shift_sec=-0.5),
    PerturbationSpec(name="tempo_0p8", kind="tempo", tempo_rate=0.8),
    PerturbationSpec(name="tempo_1p2", kind="tempo", tempo_rate=1.2),
]


DEFAULT_EVAL_SPECS = [
    PerturbationSpec(name="original", kind="original"),
    PerturbationSpec(name="silence", kind="silence"),
    PerturbationSpec(name="local_silence", kind="local_silence", local_silence_sec=1.0),
    PerturbationSpec(name="shift_pos_1p0", kind="shift", shift_sec=1.0),
    PerturbationSpec(name="shift_neg_1p0", kind="shift", shift_sec=-1.0),
    PerturbationSpec(name="tempo_0p8", kind="tempo", tempo_rate=0.8),
    PerturbationSpec(name="tempo_1p2", kind="tempo", tempo_rate=1.2),
]

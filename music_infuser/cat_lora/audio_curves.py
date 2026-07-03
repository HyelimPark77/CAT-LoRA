#!/usr/bin/env python3
"""Audio temporal control curves for audio-conditioned video diagnostics.

The curves here intentionally represent *when* audio events happen, not
high-level audio semantics. They are used for counterfactual faithfulness tests
and CAT-LoRA training signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


EPS = 1e-8


@dataclass(frozen=True)
class AudioCurveConfig:
    sample_rate: int = 16000
    hop_length: int = 512
    frame_length: int = 2048
    energy_weight: float = 0.4
    onset_weight: float = 0.4
    flux_weight: float = 0.2


def load_audio(path: str | Path, sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(str(path), sr=sample_rate, mono=True)
    return y.astype(np.float32), sr


def minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.min(x)) if x.size else 0.0
    hi = float(np.max(x)) if x.size else 0.0
    if hi - lo < EPS:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo + EPS)).astype(np.float32)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return ((x - float(np.mean(x))) / (float(np.std(x)) + EPS)).astype(np.float32)


def resample_curve(curve: np.ndarray, target_len: int) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float32)
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if curve.size == target_len:
        return curve
    if curve.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    src = np.linspace(0.0, 1.0, num=curve.size)
    dst = np.linspace(0.0, 1.0, num=target_len)
    return np.interp(dst, src, curve).astype(np.float32)


def rms_energy(y: np.ndarray, cfg: AudioCurveConfig) -> np.ndarray:
    return librosa.feature.rms(
        y=y,
        frame_length=cfg.frame_length,
        hop_length=cfg.hop_length,
    )[0].astype(np.float32)


def onset_strength(y: np.ndarray, sr: int, cfg: AudioCurveConfig) -> np.ndarray:
    return librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=cfg.hop_length,
    ).astype(np.float32)


def spectral_flux(y: np.ndarray, cfg: AudioCurveConfig) -> np.ndarray:
    stft = np.abs(
        librosa.stft(
            y,
            n_fft=cfg.frame_length,
            hop_length=cfg.hop_length,
            center=True,
        )
    )
    if stft.shape[1] <= 1:
        return np.zeros(stft.shape[1], dtype=np.float32)
    diff = np.diff(stft, axis=1)
    flux = np.sqrt(np.sum(np.maximum(diff, 0.0) ** 2, axis=0))
    return np.pad(flux, (1, 0), mode="constant").astype(np.float32)


def beat_activation(y: np.ndarray, sr: int, cfg: AudioCurveConfig) -> np.ndarray:
    """Return a sparse beat impulse curve.

    Beat tracking is unstable for very short clips, so this should be treated as
    an evaluation feature rather than the main training signal.
    """
    try:
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=cfg.hop_length)
    except Exception:
        beat_frames = np.array([], dtype=np.int64)
    n = len(rms_energy(y, cfg))
    beats = np.zeros(n, dtype=np.float32)
    beat_frames = beat_frames[(beat_frames >= 0) & (beat_frames < n)]
    beats[beat_frames] = 1.0
    return beats


def audio_control_curve(
    y: np.ndarray,
    sr: int,
    cfg: AudioCurveConfig | None = None,
    target_len: int | None = None,
) -> dict[str, np.ndarray]:
    cfg = cfg or AudioCurveConfig(sample_rate=sr)
    energy = rms_energy(y, cfg)
    onset = onset_strength(y, sr, cfg)
    flux = spectral_flux(y, cfg)

    n = max(len(energy), len(onset), len(flux))
    energy = resample_curve(energy, n)
    onset = resample_curve(onset, n)
    flux = resample_curve(flux, n)
    beat = resample_curve(beat_activation(y, sr, cfg), n)

    energy01 = minmax01(energy)
    onset01 = minmax01(onset)
    flux01 = minmax01(flux)
    control = minmax01(
        cfg.energy_weight * energy01
        + cfg.onset_weight * onset01
        + cfg.flux_weight * flux01
    )

    times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=cfg.hop_length)
    out = {
        "time": times.astype(np.float32),
        "energy": energy01,
        "onset": onset01,
        "flux": flux01,
        "beat": beat.astype(np.float32),
        "control": control,
    }
    if target_len is not None:
        out = {k: (resample_curve(v, target_len) if k != "time" else np.linspace(0, times[-1] if len(times) else 0, target_len, dtype=np.float32)) for k, v in out.items()}
    return out


def audio_control_curve_from_file(
    path: str | Path,
    cfg: AudioCurveConfig | None = None,
    target_len: int | None = None,
) -> dict[str, np.ndarray]:
    cfg = cfg or AudioCurveConfig()
    y, sr = load_audio(path, cfg.sample_rate)
    return audio_control_curve(y, sr, cfg, target_len=target_len)

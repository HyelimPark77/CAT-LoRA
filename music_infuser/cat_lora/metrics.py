#!/usr/bin/env python3
"""Metrics for counterfactual audio-temporal faithfulness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from music_infuser.cat_lora.audio_curves import resample_curve


EPS = 1e-8


@dataclass(frozen=True)
class FaithfulnessScores:
    audio_motion_corr: float
    peak_alignment_sec: float
    peak_f1: float
    silence_suppression: float | None = None
    shift_lag_sec: float | None = None
    tempo_corr: float | None = None


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    x = np.asarray(a[:n], dtype=np.float32)
    y = np.asarray(b[:n], dtype=np.float32)
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)) + EPS)
    return float(np.sum(x * y) / denom)


def top_peaks(x: np.ndarray, top_frac: float = 0.2) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return np.asarray([], dtype=np.int64)
    k = max(1, int(round(x.size * top_frac)))
    idx = np.argpartition(x, -k)[-k:]
    return np.sort(idx.astype(np.int64))


def peak_alignment(
    audio_curve: np.ndarray,
    motion_curve: np.ndarray,
    fps: float,
    tolerance_sec: float = 0.25,
    top_frac: float = 0.2,
) -> tuple[float, float]:
    n = min(len(audio_curve), len(motion_curve))
    if n == 0 or fps <= 0:
        return 0.0, 0.0
    a_peaks = top_peaks(audio_curve[:n], top_frac=top_frac)
    m_peaks = top_peaks(motion_curve[:n], top_frac=top_frac)
    if len(a_peaks) == 0 or len(m_peaks) == 0:
        return 0.0, 0.0

    distances = []
    hits = 0
    tol_frames = max(1, int(round(tolerance_sec * fps)))
    for ap in a_peaks:
        d = int(np.min(np.abs(m_peaks - ap)))
        distances.append(d / fps)
        if d <= tol_frames:
            hits += 1
    precision = hits / max(1, len(m_peaks))
    recall = hits / max(1, len(a_peaks))
    f1 = 2 * precision * recall / (precision + recall + EPS)
    return float(np.mean(distances)), float(f1)


def silence_suppression(
    original_motion: np.ndarray,
    perturbed_motion: np.ndarray,
    silence_mask: np.ndarray | None = None,
) -> float:
    n = min(len(original_motion), len(perturbed_motion))
    if n == 0:
        return 0.0
    orig = np.asarray(original_motion[:n], dtype=np.float32)
    pert = np.asarray(perturbed_motion[:n], dtype=np.float32)
    if silence_mask is not None:
        mask = resample_curve(silence_mask.astype(np.float32), n) > 0.5
        if np.any(mask):
            orig = orig[mask]
            pert = pert[mask]
    return float(1.0 - (np.mean(pert) / (np.mean(orig) + EPS)))


def estimate_lag_sec(audio_curve: np.ndarray, motion_curve: np.ndarray, fps: float) -> float:
    n = min(len(audio_curve), len(motion_curve))
    if n == 0 or fps <= 0:
        return 0.0
    a = np.asarray(audio_curve[:n], dtype=np.float32)
    m = np.asarray(motion_curve[:n], dtype=np.float32)
    a = a - float(np.mean(a))
    m = m - float(np.mean(m))
    corr = np.correlate(m, a, mode="full")
    lag = int(np.argmax(corr) - (n - 1))
    return float(lag / fps)


def faithfulness_scores(
    audio_control: np.ndarray,
    motion: np.ndarray,
    fps: float,
    original_motion: np.ndarray | None = None,
    silence_mask: np.ndarray | None = None,
) -> FaithfulnessScores:
    audio = resample_curve(audio_control, len(motion))
    mean_peak_dist, peak_f1 = peak_alignment(audio, motion, fps=fps)
    suppression = None
    if original_motion is not None:
        suppression = silence_suppression(original_motion, motion, silence_mask=silence_mask)
    return FaithfulnessScores(
        audio_motion_corr=pearson_corr(audio, motion),
        peak_alignment_sec=mean_peak_dist,
        peak_f1=peak_f1,
        silence_suppression=suppression,
        shift_lag_sec=estimate_lag_sec(audio, motion, fps=fps),
        tempo_corr=pearson_corr(audio, motion),
    )


def prompt_dominance_ratio(
    audio_change_video_distance: float,
    prompt_change_video_distance: float,
) -> float:
    return float(prompt_change_video_distance / (audio_change_video_distance + EPS))

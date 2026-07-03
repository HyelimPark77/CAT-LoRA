#!/usr/bin/env python3
"""Visual temporal response curves for generated videos."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


EPS = 1e-8


def minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < EPS:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo + EPS)).astype(np.float32)


def read_video_frames(
    path: str | Path,
    max_frames: int | None = None,
    resize_short_side: int | None = 256,
) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize_short_side is not None:
            h, w = frame.shape[:2]
            short = min(h, w)
            if short > 0 and short != resize_short_side:
                scale = resize_short_side / short
                frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))))
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from video: {path}")
    return np.stack(frames, axis=0), fps


def frame_difference_curve(frames: np.ndarray) -> np.ndarray:
    frames_f = frames.astype(np.float32) / 255.0
    diffs = np.abs(frames_f[1:] - frames_f[:-1])
    return diffs.mean(axis=(1, 2, 3)).astype(np.float32)


def optical_flow_curve(frames: np.ndarray) -> np.ndarray:
    if len(frames) < 2:
        return np.zeros(0, dtype=np.float32)
    prev = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    vals = []
    for frame in frames[1:]:
        cur = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev,
            cur,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        vals.append(float(np.mean(mag)))
        prev = cur
    return np.asarray(vals, dtype=np.float32)


def visual_response_curve(
    path: str | Path,
    mode: str = "hybrid",
    max_frames: int | None = None,
    resize_short_side: int | None = 256,
) -> dict[str, np.ndarray | float]:
    frames, fps = read_video_frames(path, max_frames=max_frames, resize_short_side=resize_short_side)
    diff = frame_difference_curve(frames)
    flow = optical_flow_curve(frames)
    if mode == "frame_diff":
        motion = diff
    elif mode == "flow":
        motion = flow
    elif mode == "hybrid":
        n = min(len(diff), len(flow))
        motion = 0.5 * minmax01(diff[:n]) + 0.5 * minmax01(flow[:n])
    else:
        raise ValueError(f"Unsupported video curve mode: {mode}")
    return {
        "motion": minmax01(motion),
        "frame_diff": minmax01(diff),
        "flow": minmax01(flow),
        "fps": fps,
        "num_frames": float(len(frames)),
    }


def latent_motion_curve(latents: "np.ndarray") -> np.ndarray:
    """Differentiable version is implemented in torch losses.

    This NumPy helper is useful for stored latent arrays with shape T x ...
    """
    z = np.asarray(latents, dtype=np.float32)
    if z.shape[0] < 2:
        return np.zeros(0, dtype=np.float32)
    return minmax01(np.mean(np.abs(z[1:] - z[:-1]), axis=tuple(range(1, z.ndim))))

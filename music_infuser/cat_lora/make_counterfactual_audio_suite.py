#!/usr/bin/env python3
"""Create paper-grade counterfactual audio suites for CAT-LoRA evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

try:
    from .perturbations import (
        PerturbationSpec,
        apply_perturbation,
        crop_or_pad,
        normalize_peak,
    )
except ImportError:
    from perturbations import PerturbationSpec, apply_perturbation, crop_or_pad, normalize_peak


def load_mono(path: Path, sr: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return normalize_peak(y.astype(np.float32))


def crop_duration(y: np.ndarray, sr: int, duration_sec: float) -> np.ndarray:
    return crop_or_pad(y, int(round(duration_sec * sr)))


def specs_for_duration(duration_sec: float) -> list[PerturbationSpec]:
    if duration_sec < 7.0:
        shift = 1.0
        local_center = duration_sec * 0.5
        local_dur = min(1.0, duration_sec * 0.25)
    else:
        shift = 2.0
        local_center = duration_sec * 0.5
        local_dur = min(2.0, duration_sec * 0.25)
    return [
        PerturbationSpec(name="original", kind="original"),
        PerturbationSpec(name="silence", kind="silence"),
        PerturbationSpec(
            name=f"local_silence_{local_center - local_dur / 2:.1f}_{local_center + local_dur / 2:.1f}",
            kind="local_silence",
            local_silence_sec=local_dur,
            local_center_sec=local_center,
        ),
        PerturbationSpec(name=f"shift{shift:g}s", kind="shift", shift_sec=shift),
        PerturbationSpec(name="tempo08", kind="tempo", tempo_rate=0.8),
        PerturbationSpec(name="tempo12", kind="tempo", tempo_rate=1.2),
    ]


def write_wav(path: Path, y: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y.astype(np.float32), sr)


def add_mismatch(
    *,
    entries: list[dict],
    output_dir: Path,
    source_path: Path | None,
    name: str,
    prefix: str,
    duration_sec: float,
    sr: int,
) -> None:
    if source_path is None:
        return
    y = crop_duration(load_mono(source_path, sr), sr, duration_sec)
    out = output_dir / f"{prefix}_{name}_{duration_sec:g}s.wav"
    write_wav(out, y, sr)
    entries.append(
        {
            "condition": name,
            "kind": "mismatch",
            "audio": str(out),
            "source_audio": str(source_path),
            "duration_sec": duration_sec,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="moon")
    parser.add_argument("--duration-sec", type=float, required=True)
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--mismatch-same-audio", type=Path)
    parser.add_argument("--mismatch-diff-audio", type=Path)
    args = parser.parse_args()

    y = crop_duration(load_mono(args.input_audio, args.sr), args.sr, args.duration_sec)
    entries: list[dict] = []

    for spec in specs_for_duration(args.duration_sec):
        y_cf, mask = apply_perturbation(y, args.sr, spec)
        out = args.output_dir / f"{args.prefix}_{spec.name}_{args.duration_sec:g}s.wav"
        write_wav(out, y_cf, args.sr)
        row = {
            "condition": spec.name,
            "kind": spec.kind,
            "audio": str(out),
            "duration_sec": args.duration_sec,
            "shift_sec": spec.shift_sec,
            "tempo_rate": spec.tempo_rate,
            "local_silence_sec": spec.local_silence_sec,
            "local_center_sec": spec.local_center_sec,
        }
        if mask is not None and mask.any():
            idx = np.flatnonzero(mask)
            row["mask_start_sec"] = float(idx[0] / args.sr)
            row["mask_end_sec"] = float((idx[-1] + 1) / args.sr)
        entries.append(row)

    add_mismatch(
        entries=entries,
        output_dir=args.output_dir,
        source_path=args.mismatch_same_audio,
        name="mismatch_same",
        prefix=args.prefix,
        duration_sec=args.duration_sec,
        sr=args.sr,
    )
    add_mismatch(
        entries=entries,
        output_dir=args.output_dir,
        source_path=args.mismatch_diff_audio,
        name="mismatch_diff",
        prefix=args.prefix,
        duration_sec=args.duration_sec,
        sr=args.sr,
    )

    meta = {
        "source_audio": str(args.input_audio),
        "prefix": args.prefix,
        "duration_sec": args.duration_sec,
        "sample_rate": args.sr,
        "entries": entries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.output_dir / "suite_manifest.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {len(entries)} audio conditions to {args.output_dir}")
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()

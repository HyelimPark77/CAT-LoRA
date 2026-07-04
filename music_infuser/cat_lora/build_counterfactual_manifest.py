#!/usr/bin/env python3
"""Build evaluation manifests from generated counterfactual outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_model_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Model entries must be NAME=OUTPUT_ROOT")
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-suite", type=Path, required=True)
    parser.add_argument("--model", action="append", type=parse_model_arg, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-id", default="dancer_rhythm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-curve", default="hybrid")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    suite = json.loads((args.audio_suite / "suite_manifest.json").read_text(encoding="utf-8"))
    runs = []
    missing = []
    for model_name, out_root in args.model:
        for entry in suite["entries"]:
            audio = Path(entry["audio"])
            stem = audio.stem
            video = out_root / stem / f"{stem}_generated.mp4"
            if not audio.exists():
                missing.append(str(audio))
            if not video.exists():
                missing.append(str(video))
            runs.append(
                {
                    "model": model_name,
                    "condition": entry["condition"],
                    "condition_kind": entry.get("kind"),
                    "prompt_id": args.prompt_id,
                    "seed": args.seed,
                    "audio": str(audio),
                    "video": str(video),
                    "duration_sec": entry.get("duration_sec"),
                    "shift_sec": entry.get("shift_sec"),
                    "tempo_rate": entry.get("tempo_rate"),
                    "mask_start_sec": entry.get("mask_start_sec"),
                    "mask_end_sec": entry.get("mask_end_sec"),
                }
            )

    if missing and not args.allow_missing:
        print("Missing files:")
        for path in missing:
            print(path)
        raise SystemExit(f"Refusing to write manifest with {len(missing)} missing files")

    manifest = {
        "video_curve": args.video_curve,
        "audio_suite": str(args.audio_suite),
        "source_audio": suite.get("source_audio"),
        "duration_sec": suite.get("duration_sec"),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(runs)} runs to {args.output}")


if __name__ == "__main__":
    main()

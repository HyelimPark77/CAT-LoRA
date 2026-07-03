#!/usr/bin/env python3
"""Evaluate counterfactual audio-temporal faithfulness for generated videos."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from music_infuser.cat_lora.audio_curves import audio_control_curve_from_file
from music_infuser.cat_lora.metrics import (
    faithfulness_scores,
    pearson_corr,
    resample_curve,
    silence_suppression,
)
from music_infuser.cat_lora.video_curves import visual_response_curve


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_eval(manifest_path: Path, output_csv: Path) -> None:
    manifest = load_json(manifest_path)
    rows = []
    cache: dict[str, dict] = {}

    runs = manifest["runs"]
    by_group: dict[tuple[str, str, int], list[dict]] = {}
    for run in runs:
        key = (run["model"], run.get("prompt_id", "default"), int(run.get("seed", 0)))
        by_group.setdefault(key, []).append(run)

    for group_key, group_runs in by_group.items():
        original = next((r for r in group_runs if r["condition"] == "original"), None)
        original_motion = None
        if original is not None:
            v = visual_response_curve(original["video"], mode=manifest.get("video_curve", "hybrid"))
            original_motion = v["motion"]
            cache[original["video"]] = v

        for run in group_runs:
            video_path = run["video"]
            audio_path = run["audio"]
            if video_path in cache:
                vcurve = cache[video_path]
            else:
                vcurve = visual_response_curve(video_path, mode=manifest.get("video_curve", "hybrid"))
                cache[video_path] = vcurve

            motion = vcurve["motion"]
            fps = float(vcurve["fps"])
            acurve = audio_control_curve_from_file(audio_path, target_len=len(motion))
            scores = faithfulness_scores(
                acurve["control"],
                motion,
                fps=fps,
                original_motion=original_motion,
            )
            row = {
                "model": run["model"],
                "condition": run["condition"],
                "prompt_id": run.get("prompt_id", "default"),
                "seed": run.get("seed", 0),
                "video": video_path,
                "audio": audio_path,
                "fps": fps,
                "num_motion_steps": len(motion),
                "audio_motion_corr": scores.audio_motion_corr,
                "peak_alignment_sec": scores.peak_alignment_sec,
                "peak_f1": scores.peak_f1,
                "silence_suppression": scores.silence_suppression,
                "estimated_lag_sec": scores.shift_lag_sec,
                "tempo_corr": scores.tempo_corr,
                "energy_corr": pearson_corr(resample_curve(acurve["energy"], len(motion)), motion),
                "onset_corr": pearson_corr(resample_curve(acurve["onset"], len(motion)), motion),
                "flux_corr": pearson_corr(resample_curve(acurve["flux"], len(motion)), motion),
            }
            if run["condition"] == "silence" and original_motion is not None:
                row["silence_suppression"] = silence_suppression(original_motion, motion)
            rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    run_eval(args.manifest, args.output_csv)


if __name__ == "__main__":
    main()

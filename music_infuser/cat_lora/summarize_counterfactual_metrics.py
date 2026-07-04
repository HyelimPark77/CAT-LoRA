#!/usr/bin/env python3
"""Summarize counterfactual metric CSVs into table-ready files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = [
    "audio_motion_corr",
    "peak_alignment_sec",
    "peak_f1",
    "silence_suppression",
    "estimated_lag_sec",
    "energy_corr",
    "onset_corr",
    "flux_corr",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frames = []
    for path in args.csv:
        df = pd.read_csv(path)
        df["source_csv"] = str(path)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_model = all_df.groupby(["source_csv", "model"])[METRICS].mean(numeric_only=True).reset_index()
    by_condition = (
        all_df.groupby(["source_csv", "model", "condition"])[METRICS]
        .mean(numeric_only=True)
        .reset_index()
    )
    all_df.to_csv(args.output_dir / "all_metrics.csv", index=False)
    by_model.to_csv(args.output_dir / "summary_by_model.csv", index=False)
    by_condition.to_csv(args.output_dir / "summary_by_condition.csv", index=False)

    print("\n=== Summary by model ===")
    print(by_model.to_string(index=False))
    print("\n=== Summary by condition ===")
    print(by_condition.to_string(index=False))


if __name__ == "__main__":
    main()

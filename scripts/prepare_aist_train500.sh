#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/342/hyellim/projects/CAT-LoRA}"
SRC_ROOT="${SRC_ROOT:-/home/ubuntu/342/hyellim/datasets/mm-diffusion/AIST++_crop/AIST++_crop}"
RAW_OUT="${RAW_OUT:-/home/ubuntu/342/hyellim/datasets/musicinfuser/aist_raw_train500}"
PREPROCESSED_OUT="${PREPROCESSED_OUT:-/home/ubuntu/342/hyellim/datasets/musicinfuser/videos_preprocessed_aist_73_train500}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$PROJECT_ROOT/weights}"
NUM_SOURCES="${NUM_SOURCES:-500}"
NUM_FRAMES="${NUM_FRAMES:-73}"
RUN_PREPROCESS="${RUN_PREPROCESS:-1}"
CAPTION="${CAPTION:-a person dancing to the rhythm of music}"

cd "$PROJECT_ROOT"
mkdir -p "$RAW_OUT"

python - <<'PY'
import os
from pathlib import Path

src_root = Path(os.environ["SRC_ROOT"])
raw_out = Path(os.environ["RAW_OUT"])
num_sources = int(os.environ["NUM_SOURCES"])
caption = os.environ["CAPTION"]
train_cameras = {"c01", "c10"}
test_tracks = {"mLH4", "mKR2", "mBR0", "mLO2", "mJB5", "mWA0", "mJS3", "mMH3", "mHO5", "mPO1"}

videos = []
for p in sorted(src_root.rglob("*.mp4")):
    parts = p.stem.split("_")
    if len(parts) < 4:
        continue
    camera = parts[2]
    track = parts[-2] if len(parts) >= 2 else ""
    if camera in train_cameras and track not in test_tracks:
        videos.append(p)

if len(videos) < num_sources:
    raise SystemExit(f"Only found {len(videos)} eligible AIST videos, requested {num_sources}")

raw_out.mkdir(parents=True, exist_ok=True)
selected = videos[:num_sources]
for src in selected:
    dst = raw_out / src.name
    if not dst.exists():
        dst.symlink_to(src)
    txt = raw_out / f"{src.stem}.txt"
    txt.write_text(caption + "\n", encoding="utf-8")

print(f"selected_sources={len(selected)}")
print(f"raw_out={raw_out}")
PY

if [[ "$RUN_PREPROCESS" == "1" ]]; then
  bash "$PROJECT_ROOT/music_infuser/preprocess.bash" \
    -v "$RAW_OUT" \
    -o "$PREPROCESSED_OUT" \
    -w "$WEIGHTS_DIR" \
    -n "$NUM_FRAMES"

  python - <<'PY'
import os
from pathlib import Path
import shutil

out = Path(os.environ["PREPROCESSED_OUT"])
fixed = 0
for mp4 in sorted(out.glob("*.seg*.mp4")):
    seg_stem = mp4.stem
    base_stem = seg_stem.split(".seg")[0]
    for suffix in [".embed.pt", ".embed_.pt"]:
        src = out / f"{base_stem}{suffix}"
        dst = out / f"{seg_stem}{suffix}"
        if not dst.exists() and src.exists():
            shutil.copy2(src, dst)
            fixed += 1
print(f"copied_segment_embeddings={fixed}")
PY
fi

PYTHONPATH=music_infuser python - <<'PY'
import os
from glob import glob
from dataset import MusicInfuserDataset

data_dir = os.environ["PREPROCESSED_OUT"]
vids = sorted(v for v in glob(f"{data_dir}/*.mp4") if not v.endswith(".recon.mp4"))
d = MusicInfuserDataset(vids, [], yt_ratio=0.0, basic_prompt_ratio=0.2, repeat=1)
print(f"input_mp4={len(vids)}")
print(f"valid_pairs={len(d)}")
print(f"specific_prompt_pairs={len(d.items)}")
print(f"basic_prompt_pairs={len(d.items_basic)}")
PY

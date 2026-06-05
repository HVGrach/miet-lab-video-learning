#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: scripts/run_frames.sh /path/to/8_frame_folder" >&2
  exit 2
fi

docker run --rm \
  -v "$1:/app/input:ro" \
  timesformer-infer \
  python -m src.predict_frames --input /app/input

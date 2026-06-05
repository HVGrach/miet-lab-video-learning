#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: scripts/run_embeddings.sh /path/to/object_embedding.npy" >&2
  exit 2
fi

docker run --rm \
  --entrypoint python \
  -v "$1:/app/input/object_embedding.npy:ro" \
  timesformer-infer \
  -m src.predict_embeddings --input /app/input/object_embedding.npy

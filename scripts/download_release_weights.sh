#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-lab6-timesformer-artifacts}"
REPO="${GITHUB_REPOSITORY:-HVGrach/miet-lab-video-learning}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

if ! command -v gh >/dev/null 2>&1; then
  echo "error: GitHub CLI 'gh' is required to download release weights" >&2
  exit 2
fi

mkdir -p "$ROOT_DIR/weights"

gh release download "$TAG" \
  --repo "$REPO" \
  --pattern timesformer_checkpoint.pt \
  --pattern timesformer_checkpoint.pt.sha256 \
  --pattern timesformer_hf.tar.gz \
  --pattern timesformer_hf.tar.gz.sha256 \
  --dir "$TMP_DIR"

(cd "$TMP_DIR" && shasum -a 256 -c timesformer_checkpoint.pt.sha256)
(cd "$TMP_DIR" && shasum -a 256 -c timesformer_hf.tar.gz.sha256)

cp "$TMP_DIR/timesformer_checkpoint.pt" "$ROOT_DIR/weights/timesformer_checkpoint.pt"
tar -xzf "$TMP_DIR/timesformer_hf.tar.gz" -C "$ROOT_DIR/weights"

echo "restored weights/timesformer_checkpoint.pt and weights/timesformer_hf/"

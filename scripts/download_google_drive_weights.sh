#!/usr/bin/env bash
set -euo pipefail

DRIVE_FILE_ID="${LAB6_DRIVE_FILE_ID:-1q4rjJALqwzE-Hb3QRtvxand6QDoSSZJh}"
EXPECTED_SHA256="${LAB6_WEIGHTS_SHA256:-967d087056729d2828d6aa48e65a17d1ec3f38cc62d694ad3f0d4a23537199d3}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_DIR="$ROOT_DIR/release_assets"
BUNDLE_PATH="${1:-$ASSET_DIR/lab6_weights_bundle.tar.gz}"

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    python3 - "$1" <<'PY'
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
  fi
}

mkdir -p "$ASSET_DIR" "$ROOT_DIR/weights" "$ROOT_DIR/outputs/models"

if [[ ! -f "$BUNDLE_PATH" ]]; then
  echo "downloading Google Drive weights bundle to $BUNDLE_PATH" >&2
  python3 - "$DRIVE_FILE_ID" "$BUNDLE_PATH" <<'PY'
from __future__ import annotations

import html
import os
import re
import shutil
import sys
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar


file_id, output_path = sys.argv[1], sys.argv[2]
base_url = f"https://drive.google.com/uc?export=download&id={file_id}"
cookie_jar = CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))


def open_url(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return opener.open(request, timeout=120)


def write_stream(response, first_chunk: bytes) -> None:
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "wb") as fh:
        fh.write(first_chunk)
        shutil.copyfileobj(response, fh, length=1024 * 1024)
    os.replace(tmp_path, output_path)


def parse_confirm_url(page: str) -> str | None:
    form_match = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"', page)
    if form_match:
        action = html.unescape(form_match.group(1))
        params: dict[str, str] = {}
        for tag in re.findall(r"<input[^>]+>", page):
            name_match = re.search(r'name="([^"]+)"', tag)
            value_match = re.search(r'value="([^"]*)"', tag)
            if name_match and value_match:
                params[html.unescape(name_match.group(1))] = html.unescape(value_match.group(1))
        if params:
            return f"{action}?{urllib.parse.urlencode(params)}"

    for match in re.finditer(r'href="([^"]+)"', page):
        href = html.unescape(match.group(1))
        if "confirm=" in href and ("download" in href or "uc?" in href):
            return urllib.parse.urljoin("https://drive.google.com", href)
    return None


response = open_url(base_url)
first = response.read(64 * 1024)
if first.startswith(b"\x1f\x8b"):
    write_stream(response, first)
    raise SystemExit(0)

body = first + response.read()
page = body.decode("utf-8", errors="ignore")
confirm_url = parse_confirm_url(page)
if confirm_url is None:
    debug_path = f"{output_path}.response.html"
    with open(debug_path, "wb") as fh:
        fh.write(body)
    raise SystemExit(f"could not find Google Drive confirm URL; saved response to {debug_path}")

response = open_url(confirm_url)
first = response.read(64 * 1024)
if not first.startswith(b"\x1f\x8b"):
    debug_path = f"{output_path}.response.html"
    with open(debug_path, "wb") as fh:
        fh.write(first + response.read())
    raise SystemExit(f"Google Drive did not return a gzip archive; saved response to {debug_path}")

write_stream(response, first)
PY
fi

actual_sha256="$(sha256_file "$BUNDLE_PATH")"
if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
  echo "error: checksum mismatch for $BUNDLE_PATH" >&2
  echo "expected: $EXPECTED_SHA256" >&2
  echo "actual:   $actual_sha256" >&2
  exit 2
fi

tar -xzf "$BUNDLE_PATH" -C "$ROOT_DIR"

for required in \
  "$ROOT_DIR/weights/timesformer_checkpoint.pt" \
  "$ROOT_DIR/weights/timesformer_hf/config.json" \
  "$ROOT_DIR/weights/bagging_best.joblib" \
  "$ROOT_DIR/weights/labels.json" \
  "$ROOT_DIR/outputs/models/lab6_action_classifier.pt"; do
  if [[ ! -e "$required" ]]; then
    echo "error: restored bundle is missing $required" >&2
    exit 2
  fi
done

echo "restored weights from $BUNDLE_PATH"

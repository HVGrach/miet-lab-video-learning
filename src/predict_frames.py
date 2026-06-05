from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import (
    DEFAULT_HF_MODEL_DIR,
    DEFAULT_LABELS,
    DEFAULT_TIMESFORMER_CHECKPOINT,
    DEFAULT_WEIGHTS_DIR,
    load_labels,
)
from .data_io import collect_frame_paths
from .model_timesformer import load_timesformer_model, predict_timesformer_label
from .preprocess_frames import preprocess_frame_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify one folder with exactly 8 frames.")
    parser.add_argument("--input", required=True, help="Folder containing exactly 8 image files.")
    parser.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS_DIR), help="Directory with deployment weights.")
    parser.add_argument("--checkpoint", default=None, help="TimeSFormer checkpoint .pt path.")
    parser.add_argument("--labels", default=None, help="labels.json path.")
    parser.add_argument("--hf-model-path", default=None, help="Local HuggingFace TimeSFormer directory.")
    parser.add_argument("--device", default="cpu", help="Torch device. Use cpu for Docker portability.")
    parser.add_argument(
        "--allow-runtime-downloads",
        action="store_true",
        help="Allow HuggingFace downloads at runtime. Disabled by default for deployment.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    weights_dir = Path(args.weights_dir)
    checkpoint = Path(args.checkpoint) if args.checkpoint else weights_dir / DEFAULT_TIMESFORMER_CHECKPOINT.name
    labels_path = Path(args.labels) if args.labels else weights_dir / DEFAULT_LABELS.name
    hf_model_path = Path(args.hf_model_path) if args.hf_model_path else weights_dir / DEFAULT_HF_MODEL_DIR.name
    if not hf_model_path.is_dir():
        hf_model_path = None

    try:
        labels = load_labels(labels_path)
        frame_paths = collect_frame_paths(args.input)
        pixel_values = preprocess_frame_paths(frame_paths)
        model = load_timesformer_model(
            checkpoint_path=checkpoint,
            hf_model_path=hf_model_path,
            device=args.device,
            allow_runtime_downloads=args.allow_runtime_downloads,
        )
        label = predict_timesformer_label(model, pixel_values, labels, device=args.device)
    except Exception as error:  # noqa: BLE001
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

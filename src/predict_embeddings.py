from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DEFAULT_BAGGING_MODEL, DEFAULT_LABELS, DEFAULT_WEIGHTS_DIR, load_labels
from .data_io import format_batch_csv, format_batch_json, load_embedding_inputs, stack_embedding_rows
from .model_bagging import load_bagging_model, predict_bagging_labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify object embeddings with bagging_best.joblib.")
    parser.add_argument("--input", required=True, help=".npy embedding file or folder with .npy files.")
    parser.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS_DIR), help="Directory with deployment weights.")
    parser.add_argument("--model", default=None, help="bagging_best.joblib path.")
    parser.add_argument("--labels", default=None, help="labels.json path.")
    parser.add_argument(
        "--format",
        choices=("plain", "csv", "json"),
        default="plain",
        help="Output format. plain is allowed only for one embedding row.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    weights_dir = Path(args.weights_dir)
    model_path = Path(args.model) if args.model else weights_dir / DEFAULT_BAGGING_MODEL.name
    labels_path = Path(args.labels) if args.labels else weights_dir / DEFAULT_LABELS.name

    try:
        labels = load_labels(labels_path)
        items = load_embedding_inputs(args.input)
        names, features = stack_embedding_rows(items)
        model = load_bagging_model(model_path)
        predicted = predict_bagging_labels(model, features, labels)
        output = _format_output(args.format, names, predicted)
    except Exception as error:  # noqa: BLE001
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(output)
    return 0


def _format_output(fmt: str, names: list[str], labels: list[str]) -> str:
    if fmt == "plain":
        if len(labels) != 1:
            raise ValueError("plain output is only valid for one embedding row; use --format csv or --format json")
        return labels[0]
    if fmt == "csv":
        return format_batch_csv(names, labels)
    if fmt == "json":
        return format_batch_json(names, labels)
    raise ValueError(f"unknown output format: {fmt}")


if __name__ == "__main__":
    raise SystemExit(main())

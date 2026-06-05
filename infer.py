from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lab6_analysis.inference import DEFAULT_CHECKPOINT, load_lab6_predictor, predict_folder


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"python", "python3"}:
        os.execvp(sys.executable, [sys.executable, *argv[1:]])

    parser = argparse.ArgumentParser(
        description="Classify one lab6 8-frame action folder.",
    )
    parser.add_argument("folder", help="Folder containing exactly 8 images.")
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / DEFAULT_CHECKPOINT),
        help="Path to a lab6 checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for inference. Use cpu for Docker portability.",
    )
    args = parser.parse_args(argv)

    try:
        predictor = load_lab6_predictor(args.checkpoint, device=args.device)
        result = predict_folder(args.folder, predictor)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(result.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import DEPLOYMENT


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_frame_paths(folder: str | Path, *, allow_hidden_junk: bool = True) -> list[Path]:
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"input must be a folder: {folder_path}")

    files = [path for path in folder_path.iterdir() if path.is_file()]
    unsupported = [
        path
        for path in files
        if path.suffix.lower() not in IMAGE_SUFFIXES
        and not (allow_hidden_junk and path.name.startswith("."))
    ]
    if unsupported:
        names = ", ".join(path.name for path in sorted(unsupported, key=lambda p: p.name))
        raise ValueError(f"input folder contains non-image files: {names}")

    image_paths = sorted(
        [path for path in files if path.suffix.lower() in IMAGE_SUFFIXES],
        key=lambda path: path.name,
    )
    if len(image_paths) != DEPLOYMENT.num_frames:
        raise ValueError(
            f"input folder must contain exactly {DEPLOYMENT.num_frames} images; "
            f"found {len(image_paths)}"
        )
    return image_paths


def load_embedding_inputs(input_path: str | Path) -> list[tuple[str, np.ndarray]]:
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() != ".npy":
            raise ValueError(f"embedding input must be a .npy file: {path}")
        return [(path.name, _load_embedding_array(path))]
    if path.is_dir():
        files = sorted(path.glob("*.npy"), key=lambda item: item.name)
        if not files:
            raise ValueError(f"embedding folder does not contain .npy files: {path}")
        return [(item.name, _load_embedding_array(item)) for item in files]
    raise ValueError(f"embedding input does not exist: {path}")


def stack_embedding_rows(items: Iterable[tuple[str, np.ndarray]]) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    rows: list[np.ndarray] = []
    for name, array in items:
        matrix = _as_2d(array)
        if matrix.shape[0] == 1:
            names.append(name)
        else:
            names.extend(f"{name}#{index}" for index in range(matrix.shape[0]))
        rows.append(matrix)
    if not rows:
        raise ValueError("no embeddings to predict")
    return names, np.concatenate(rows, axis=0).astype(np.float32, copy=False)


def format_batch_json(names: list[str], labels: list[str]) -> str:
    return json.dumps(
        [{"input": name, "label": label} for name, label in zip(names, labels)],
        ensure_ascii=False,
    )


def format_batch_csv(names: list[str], labels: list[str]) -> str:
    rows = ["input,label"]
    rows.extend(f"{name},{label}" for name, label in zip(names, labels))
    return "\n".join(rows)


def _load_embedding_array(path: Path) -> np.ndarray:
    try:
        array = np.load(path, allow_pickle=False)
    except Exception as error:  # noqa: BLE001
        raise ValueError(f"could not read embedding .npy file {path}: {error}") from error
    return _as_2d(array)


def _as_2d(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"embedding array must be 1D or 2D; got shape={arr.shape}")

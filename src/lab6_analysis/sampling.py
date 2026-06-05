from __future__ import annotations

from pathlib import Path
import hashlib
import json
import random
import re
from typing import Iterable

import pandas as pd

EXPECTED_FRAME_COUNT = 8
MOVE_WORK_LABELS = {"move", "work"}

IM_NUMBER_RE = re.compile(r"(?:^|_)im0*(?P<number>\d{1,8})(?:_|$)", re.IGNORECASE)
TRAILING_NUMBER_RE = re.compile(r"_(?P<number>\d+)$")
NUMERIC_STEM_RE = re.compile(r"^(?P<number>\d+)$")
NATURAL_TOKEN_RE = re.compile(r"(\d+)")


def sort_frame_paths_temporally(paths: Iterable[str]) -> list[str]:
    """Sort lab6 frame paths by the best available frame number."""
    return sorted([str(path) for path in paths], key=_temporal_sort_key)


def make_dilated_window(
    paths: list[str],
    *,
    start: int,
    dilation: int,
    frame_count: int = EXPECTED_FRAME_COUNT,
) -> list[str]:
    """Return one fixed-length window with a temporal dilation step."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if dilation <= 0:
        raise ValueError("dilation must be positive")
    if start < 0:
        raise ValueError("start must be non-negative")

    last_index = start + (frame_count - 1) * dilation
    if last_index >= len(paths):
        raise ValueError("window does not fit into paths")
    return [paths[start + index * dilation] for index in range(frame_count)]


def make_strided_dilated_windows(
    paths: list[str],
    *,
    label: str,
    folder_rel: str,
    dilation: int,
    stride: int,
    frame_count: int = EXPECTED_FRAME_COUNT,
    max_windows: int | None = None,
) -> list[dict[str, object]]:
    """Create stride/dilation windows for move/work long tracks."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    ordered = sort_frame_paths_temporally(paths)
    span = 1 + (frame_count - 1) * dilation
    if len(ordered) < span:
        return []

    starts = list(range(0, len(ordered) - span + 1, stride))
    starts = _limit_evenly(starts, max_windows)
    rows: list[dict[str, object]] = []
    for start in starts:
        frame_indices = [start + index * dilation for index in range(frame_count)]
        window = make_dilated_window(
            ordered,
            start=start,
            dilation=dilation,
            frame_count=frame_count,
        )
        rows.append(
            _candidate_row(
                sample_id=f"{folder_rel}::stride_dilation:d{dilation}:s{stride}:start{start}",
                label=label,
                folder_rel=folder_rel,
                group_key=f"start_{start}",
                sample_kind="stride_dilation",
                frame_paths=window,
                frame_indices=frame_indices,
                source_track_id=folder_rel,
                source_frame_count=len(ordered),
                start_index=start,
                stride=stride,
                dilation=dilation,
                temporal_order="ordered",
                generation_seed=None,
            )
        )
    return rows


def make_inaction_skip_shuffle_samples(
    paths: list[str],
    *,
    folder_rel: str,
    samples_per_folder: int,
    seed: int,
    frame_count: int = EXPECTED_FRAME_COUNT,
    shuffle: bool = False,
) -> list[dict[str, object]]:
    """Create deterministic inaction samples by skipping arbitrary frames."""
    ordered = sort_frame_paths_temporally(paths)
    if len(ordered) < frame_count or samples_per_folder <= 0:
        return []

    label = folder_rel.split("/", 1)[0]
    rng = random.Random(_stable_seed(seed, folder_rel))
    rows: list[dict[str, object]] = []
    for sample_index in range(samples_per_folder):
        indices = sorted(rng.sample(range(len(ordered)), frame_count))
        if shuffle:
            rng.shuffle(indices)
            if indices == sorted(indices) and len(indices) > 1:
                indices[0], indices[1] = indices[1], indices[0]
        window = [ordered[index] for index in indices]
        sample_kind = "inaction_skip_shuffle" if shuffle else "inaction_skip_ordered"
        rows.append(
            _candidate_row(
                sample_id=f"{folder_rel}::{sample_kind}:seed{seed}:sample{sample_index}",
                label=label,
                folder_rel=folder_rel,
                group_key=f"{sample_kind}_{sample_index}",
                sample_kind=sample_kind,
                frame_paths=window,
                frame_indices=indices,
                source_track_id=folder_rel,
                source_frame_count=len(ordered),
                start_index=None,
                stride=None,
                dilation=None,
                temporal_order="shuffled" if shuffle else "ordered_skip",
                generation_seed=seed,
            )
        )
    return rows


def build_training_sample_candidates(
    frame_manifest: pd.DataFrame,
    *,
    explicit_sample_manifest: pd.DataFrame | None = None,
    move_work_strategies: tuple[tuple[int, int], ...] | None = None,
    move_work_dilations: tuple[int, ...] = (1, 2, 4, 8),
    move_work_stride: int = 8,
    max_windows_per_folder_per_dilation: int | None = None,
    max_windows_per_folder_per_dilation_by_label: dict[str, int] | None = None,
    inaction_samples_per_folder: int = 24,
    inaction_shuffle: bool = False,
    seed: int = 20260602,
    frame_count: int = EXPECTED_FRAME_COUNT,
    skip_prefixed_explicit_folders: bool = True,
) -> pd.DataFrame:
    """Build explicit and generated 8-frame candidates for training."""
    _require_columns(frame_manifest, {"label", "folder_rel", "rel_path"})
    rows: list[dict[str, object]] = []
    prefixed_folders: set[str] = set()
    if move_work_strategies is None:
        if move_work_dilations == (1, 2, 4, 8) and move_work_stride == 8:
            move_work_strategies = ((1, 8), (2, 8), (4, 16), (8, 32))
        else:
            move_work_strategies = tuple(
                (dilation, move_work_stride) for dilation in move_work_dilations
            )
    if (
        max_windows_per_folder_per_dilation_by_label is None
        and max_windows_per_folder_per_dilation is None
    ):
        max_windows_per_folder_per_dilation_by_label = {"move": 2, "work": 1}

    if explicit_sample_manifest is not None and not explicit_sample_manifest.empty:
        _require_columns(
            explicit_sample_manifest,
            {
                "sample_id",
                "label",
                "folder_rel",
                "group_key",
                "sample_kind",
                "frame_count",
                "frame_paths",
            },
        )
        for _, sample in explicit_sample_manifest.iterrows():
            sample_kind = str(sample["sample_kind"])
            if sample_kind in {"prefixed_8", "remaining_8"}:
                prefixed_folders.add(str(sample["folder_rel"]))
            rows.append(
                _candidate_row(
                    sample_id=f"explicit::{sample['sample_id']}",
                    label=str(sample["label"]),
                    folder_rel=str(sample["folder_rel"]),
                    group_key=str(sample["group_key"]),
                    sample_kind=f"explicit_{sample_kind}",
                    frame_paths=json.loads(sample["frame_paths"]),
                    frame_indices=list(range(int(sample["frame_count"]))),
                    source_track_id=str(sample["folder_rel"]),
                    source_frame_count=int(sample["frame_count"]),
                    start_index=0,
                    stride=0,
                    dilation=1,
                    temporal_order="as_manifest",
                    generation_seed=None,
                )
            )

    for (label, folder_rel), group in frame_manifest.groupby(
        ["label", "folder_rel"], sort=True
    ):
        paths = sort_frame_paths_temporally(group["rel_path"].tolist())
        if len(paths) <= frame_count:
            continue
        if skip_prefixed_explicit_folders and folder_rel in prefixed_folders:
            continue

        if label == "inaction":
            rows.extend(
                make_inaction_skip_shuffle_samples(
                    paths,
                    folder_rel=folder_rel,
                    samples_per_folder=inaction_samples_per_folder,
                    seed=seed,
                    frame_count=frame_count,
                    shuffle=inaction_shuffle,
                )
            )
            continue

        if label in MOVE_WORK_LABELS:
            for dilation, stride in move_work_strategies:
                max_windows = max_windows_per_folder_per_dilation
                if max_windows_per_folder_per_dilation_by_label is not None:
                    max_windows = max_windows_per_folder_per_dilation_by_label.get(
                        label, max_windows
                    )
                rows.extend(
                    make_strided_dilated_windows(
                        paths,
                        label=label,
                        folder_rel=folder_rel,
                        dilation=dilation,
                        stride=stride,
                        frame_count=frame_count,
                        max_windows=max_windows,
                    )
                )

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return pd.DataFrame(columns=_candidate_columns())
    return candidates[_candidate_columns()]


def summarize_training_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Summarize generated candidate volume by class and strategy."""
    if candidates.empty:
        return pd.DataFrame()
    _require_columns(candidates, {"label", "sample_kind", "folder_rel", "sample_id"})
    summary = (
        candidates.groupby(["label", "sample_kind"], sort=True)
        .agg(
            samples=("sample_id", "count"),
            folders=("folder_rel", "nunique"),
            mean_source_frames=("source_frame_count", "mean"),
        )
        .reset_index()
    )
    summary["mean_source_frames"] = summary["mean_source_frames"].round(2)
    return summary


def _candidate_row(
    *,
    sample_id: str,
    label: str,
    folder_rel: str,
    group_key: str,
    sample_kind: str,
    frame_paths: list[str],
    frame_indices: list[int],
    source_track_id: str,
    source_frame_count: int,
    start_index: int | None,
    stride: int | None,
    dilation: int | None,
    temporal_order: str,
    generation_seed: int | None,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "label": label,
        "folder_rel": folder_rel,
        "group_key": group_key,
        "sample_kind": sample_kind,
        "frame_count": len(frame_paths),
        "frame_paths": json.dumps(frame_paths, ensure_ascii=False),
        "frame_indices": json.dumps(frame_indices, ensure_ascii=False),
        "source_track_id": source_track_id,
        "source_frame_count": source_frame_count,
        "start_index": start_index,
        "stride": stride,
        "dilation": dilation,
        "temporal_order": temporal_order,
        "generation_seed": generation_seed,
    }


def _candidate_columns() -> list[str]:
    return [
        "sample_id",
        "label",
        "folder_rel",
        "group_key",
        "sample_kind",
        "frame_count",
        "frame_paths",
        "frame_indices",
        "source_track_id",
        "source_frame_count",
        "start_index",
        "stride",
        "dilation",
        "temporal_order",
        "generation_seed",
    ]


def _temporal_sort_key(path: str):
    stem = Path(path).stem
    frame_number = _extract_frame_number(stem)
    if frame_number is not None:
        return (0, frame_number, _natural_key(stem), path)
    return (1, _natural_key(stem), path)


def _extract_frame_number(stem: str) -> int | None:
    for pattern in (IM_NUMBER_RE, TRAILING_NUMBER_RE, NUMERIC_STEM_RE):
        match = pattern.search(stem)
        if match:
            return int(match.group("number"))
    return None


def _natural_key(text: str):
    return [
        int(token) if token.isdigit() else token.lower()
        for token in NATURAL_TOKEN_RE.split(text)
    ]


def _limit_evenly(values: list[int], limit: int | None) -> list[int]:
    if limit is None or len(values) <= limit:
        return values
    if limit <= 0:
        return []
    if limit == 1:
        return [values[0]]

    selected_indexes = {
        round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)
    }
    return [values[index] for index in sorted(selected_indexes)]


def _stable_seed(seed: int, folder_rel: str) -> int:
    digest = hashlib.sha256(f"{seed}:{folder_rel}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing columns: {sorted(missing)}")

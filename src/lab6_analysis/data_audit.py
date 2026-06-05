from __future__ import annotations

from pathlib import Path
import json
import re

import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EXPECTED_LABELS = {"inaction", "move", "work"}
PREFIXED_FRAME_RE = re.compile(r"^(?P<prefix>.+)_im(?P<index>[1-8])$", re.IGNORECASE)
PLAIN_FRAME_RE = re.compile(r"^(?:im)?(?P<index>[1-8])$", re.IGNORECASE)


def build_frame_manifest(dataset_root: str | Path) -> pd.DataFrame:
    """Build a frame-level manifest for lab6-style image folders."""
    root = Path(dataset_root)
    rows: list[dict[str, object]] = []

    for image_path in sorted(_iter_image_paths(root)):
        rel_path = image_path.relative_to(root)
        label = rel_path.parts[0]
        if label not in EXPECTED_LABELS:
            continue

        with Image.open(image_path) as image:
            width, height = image.size
            mode = image.mode

        folder_rel = str(image_path.parent.relative_to(root))
        rows.append(
            {
                "label": label,
                "folder_rel": folder_rel,
                "file_name": image_path.name,
                "rel_path": str(rel_path),
                "width": width,
                "height": height,
                "mode": mode,
                "aspect_ratio": width / height,
                "area": width * height,
            }
        )

    return pd.DataFrame(rows)


def build_sample_manifest(dataset_root: str | Path) -> pd.DataFrame:
    """Build a logical 8-frame sample manifest from lab6 folders."""
    root = Path(dataset_root)
    rows: list[dict[str, object]] = []

    for folder in sorted(path for path in root.rglob("*") if path.is_dir()):
        image_paths = sorted(_direct_images(folder))
        if not image_paths:
            continue
        rel_folder = folder.relative_to(root)
        label = rel_folder.parts[0]
        if label not in EXPECTED_LABELS:
            continue

        matched_paths: set[Path] = set()
        prefixed = _group_prefixed_frames(image_paths)
        for group_key, frames in sorted(prefixed.items()):
            if set(frames) == set(range(1, 9)):
                ordered = [frames[index] for index in range(1, 9)]
                matched_paths.update(ordered)
                rows.append(
                    _sample_row(
                        root=root,
                        label=label,
                        folder=folder,
                        group_key=group_key,
                        sample_kind="prefixed_8",
                        frames=ordered,
                    )
                )

        remaining = [path for path in image_paths if path not in matched_paths]
        if len(remaining) == 8:
            ordered_remaining = _sort_plain_or_name(remaining)
            has_prefixed_groups = bool(matched_paths)
            rows.append(
                _sample_row(
                    root=root,
                    label=label,
                    folder=folder,
                    group_key="remaining" if has_prefixed_groups else "folder",
                    sample_kind="remaining_8" if has_prefixed_groups else "folder_8",
                    frames=ordered_remaining,
                )
            )

    return pd.DataFrame(rows)


def summarize_dimension_stats(frame_manifest: pd.DataFrame) -> pd.DataFrame:
    """Summarize frame geometry by class label."""
    if frame_manifest.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for label, group in frame_manifest.groupby("label", sort=True):
        row: dict[str, object] = {
            "label": label,
            "frame_count": int(len(group)),
            "folder_count": int(group["folder_rel"].nunique()),
            "unique_sizes": int(group[["width", "height"]].drop_duplicates().shape[0]),
        }
        for column in ("width", "height", "aspect_ratio", "area"):
            values = group[column]
            row[f"{column}_min"] = float(values.min())
            row[f"{column}_p25"] = float(values.quantile(0.25))
            row[f"{column}_median"] = float(values.median())
            row[f"{column}_p75"] = float(values.quantile(0.75))
            row[f"{column}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_folder_geometry(frame_manifest: pd.DataFrame) -> pd.DataFrame:
    """Summarize geometry variability inside each image folder/track."""
    if frame_manifest.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (label, folder_rel), group in frame_manifest.groupby(
        ["label", "folder_rel"], sort=True
    ):
        unique_sizes = int(group[["width", "height"]].drop_duplicates().shape[0])
        rows.append(
            {
                "label": label,
                "folder_rel": folder_rel,
                "frame_count": int(len(group)),
                "unique_sizes": unique_sizes,
                "has_size_variability": bool(unique_sizes > 1),
                "width_min": float(group["width"].min()),
                "width_median": float(group["width"].median()),
                "width_max": float(group["width"].max()),
                "height_min": float(group["height"].min()),
                "height_median": float(group["height"].median()),
                "height_max": float(group["height"].max()),
                "aspect_ratio_min": float(group["aspect_ratio"].min()),
                "aspect_ratio_median": float(group["aspect_ratio"].median()),
                "aspect_ratio_max": float(group["aspect_ratio"].max()),
            }
        )
    return pd.DataFrame(rows)


def summarize_sample_geometry(
    frame_manifest: pd.DataFrame, sample_manifest: pd.DataFrame
) -> pd.DataFrame:
    """Summarize geometry for logical 8-frame samples."""
    if frame_manifest.empty or sample_manifest.empty:
        return pd.DataFrame()

    sample_frames = explode_sample_frames(sample_manifest)
    merged = sample_frames.merge(
        frame_manifest[
            ["rel_path", "width", "height", "mode", "aspect_ratio", "area"]
        ],
        on="rel_path",
        how="left",
        validate="many_to_one",
    )

    rows: list[dict[str, object]] = []
    for sample_id, group in merged.groupby("sample_id", sort=True):
        first = group.iloc[0]
        unique_sizes = int(group[["width", "height"]].drop_duplicates().shape[0])
        rows.append(
            {
                "sample_id": sample_id,
                "label": first["label"],
                "folder_rel": first["folder_rel"],
                "sample_kind": first["sample_kind"],
                "group_key": first["group_key"],
                "frame_count": int(len(group)),
                "unique_sizes": unique_sizes,
                "has_size_variability": bool(unique_sizes > 1),
                "width_min": float(group["width"].min()),
                "width_median": float(group["width"].median()),
                "width_max": float(group["width"].max()),
                "height_min": float(group["height"].min()),
                "height_median": float(group["height"].median()),
                "height_max": float(group["height"].max()),
                "aspect_ratio_min": float(group["aspect_ratio"].min()),
                "aspect_ratio_median": float(group["aspect_ratio"].median()),
                "aspect_ratio_max": float(group["aspect_ratio"].max()),
            }
        )
    return pd.DataFrame(rows)


def find_geometry_outliers(
    frame_manifest: pd.DataFrame,
    *,
    min_height: int = 64,
    min_width: int = 64,
    aspect_low: float = 0.25,
    aspect_high: float = 2.5,
) -> pd.DataFrame:
    """Return frames with suspicious geometry and an explicit reason."""
    rows: list[dict[str, object]] = []
    for _, row in frame_manifest.iterrows():
        reasons: list[str] = []
        if row["width"] < min_width:
            reasons.append("small_width")
        if row["height"] < min_height:
            reasons.append("small_height")
        if row["aspect_ratio"] < aspect_low:
            reasons.append("low_aspect")
        if row["aspect_ratio"] > aspect_high:
            reasons.append("high_aspect")
        if not reasons:
            continue
        outlier = row.to_dict()
        outlier["outlier_reason"] = ",".join(reasons)
        rows.append(outlier)
    return pd.DataFrame(rows)


def explode_sample_frames(sample_manifest: pd.DataFrame) -> pd.DataFrame:
    """Convert sample manifest JSON frame lists to one row per frame."""
    rows: list[dict[str, object]] = []
    for _, sample in sample_manifest.iterrows():
        frame_paths = json.loads(sample["frame_paths"])
        for frame_index, rel_path in enumerate(frame_paths, start=1):
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "label": sample["label"],
                    "folder_rel": sample["folder_rel"],
                    "sample_kind": sample["sample_kind"],
                    "group_key": sample["group_key"],
                    "frame_index": frame_index,
                    "rel_path": rel_path,
                }
            )
    return pd.DataFrame(rows)


def _iter_image_paths(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _direct_images(folder: Path) -> list[Path]:
    return [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _group_prefixed_frames(paths: list[Path]) -> dict[str, dict[int, Path]]:
    groups: dict[str, dict[int, Path]] = {}
    for path in paths:
        match = PREFIXED_FRAME_RE.match(path.stem)
        if not match:
            continue
        group = groups.setdefault(match.group("prefix"), {})
        group[int(match.group("index"))] = path
    return groups


def _sort_plain_or_name(paths: list[Path]) -> list[Path]:
    indexed: list[tuple[int, Path]] = []
    for path in paths:
        match = PLAIN_FRAME_RE.match(path.stem)
        if not match:
            return sorted(paths)
        indexed.append((int(match.group("index")), path))
    return [path for _, path in sorted(indexed)]


def _sample_row(
    *,
    root: Path,
    label: str,
    folder: Path,
    group_key: str,
    sample_kind: str,
    frames: list[Path],
) -> dict[str, object]:
    rel_folder = str(folder.relative_to(root))
    rel_frames = [str(path.relative_to(root)) for path in frames]
    return {
        "sample_id": f"{rel_folder}::{group_key}",
        "label": label,
        "folder_rel": rel_folder,
        "group_key": group_key,
        "sample_kind": sample_kind,
        "frame_count": len(frames),
        "frame_paths": json.dumps(rel_frames, ensure_ascii=False),
    }

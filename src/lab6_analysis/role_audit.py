from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageStat


FORBIDDEN_MODEL_FEATURE_COLUMNS = {
    "sample_id",
    "folder_rel",
    "source_track_id",
    "source_track_group_id",
    "hash_guard_group_id",
    "file_name",
    "rel_path",
    "path",
    "md5",
    "filename_time_token",
    "dominant_time_token",
    "file_size",
    "mtime",
    "ctime",
    "birthtime",
    "folder_mtime",
    "width",
    "height",
    "aspect_ratio",
    "area",
    "role_label",
    "role_label_source",
    "sample_kind",
    "group_key",
    "frame_count",
    "frame_paths",
    "frame_indices",
    "source_frame_count",
    "start_index",
    "stride",
    "dilation",
    "temporal_order",
    "generation_seed",
}

ROLE_LABELS = {"employee", "customer", "unknown"}


def validate_model_feature_columns(columns: list[str] | pd.Index) -> None:
    """Reject non-pixel-derived metadata before fitting any classifier."""
    forbidden = sorted(FORBIDDEN_MODEL_FEATURE_COLUMNS & set(columns))
    if forbidden:
        raise ValueError(f"model features contain forbidden metadata: {forbidden}")


def validate_role_labels_are_manual(role_labels: pd.DataFrame) -> None:
    """Ensure role labels came from manual review, not from action labels."""
    _require_columns(role_labels, {"sample_id", "role_label", "role_label_source"})
    invalid_labels = sorted(set(role_labels["role_label"]) - ROLE_LABELS)
    if invalid_labels:
        raise ValueError(f"unknown role labels: {invalid_labels}")
    bad_sources = sorted(set(role_labels["role_label_source"]) - {"manual"})
    if bad_sources:
        raise ValueError(
            "role classifier requires manual role labels; "
            f"found sources: {bad_sources}"
        )


def build_manual_role_seed_manifest(sample_manifest: pd.DataFrame) -> pd.DataFrame:
    """Create a manual-review manifest without deriving role from action label."""
    _require_columns(sample_manifest, {"sample_id", "label", "folder_rel", "frame_paths"})
    rows = []
    for _, sample in sample_manifest.iterrows():
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "label": sample["label"],
                "folder_rel": sample["folder_rel"],
                "frame_paths": sample["frame_paths"],
                "role_label": "unknown",
                "role_label_source": "manual_review_needed",
                "review_note": "",
            }
        )
    return pd.DataFrame(rows)


def compute_green_proxy_for_frames(
    dataset_root: str | Path,
    frame_manifest: pd.DataFrame,
    *,
    max_frames_per_folder: int | None = 16,
    image_size: tuple[int, int] = (32, 32),
) -> pd.DataFrame:
    """Compute a crude uniform-color proxy for audit ranking, not modeling."""
    _require_columns(frame_manifest, {"label", "folder_rel", "rel_path"})
    root = Path(dataset_root)
    rows: list[dict[str, object]] = []

    for (_, folder_rel), group in frame_manifest.groupby(["label", "folder_rel"], sort=True):
        selected = group
        if max_frames_per_folder is not None and len(group) > max_frames_per_folder:
            selected = group.iloc[_evenly_spaced_indexes(len(group), max_frames_per_folder)]
        for _, frame in selected.iterrows():
            with Image.open(root / frame["rel_path"]) as image:
                rgb = image.convert("RGB").resize(image_size)
                r, g, b = ImageStat.Stat(rgb).mean
            rows.append(
                {
                    "label": frame["label"],
                    "folder_rel": frame["folder_rel"],
                    "rel_path": frame["rel_path"],
                    "red_mean": r,
                    "green_mean": g,
                    "blue_mean": b,
                    "green_excess": g - (r + b) / 2,
                    "green_ratio": g / (r + g + b + 1e-9),
                }
            )
    return pd.DataFrame(rows)


def summarize_green_proxy_by_folder(frame_proxy: pd.DataFrame) -> pd.DataFrame:
    """Summarize the proxy by folder for manual role-audit prioritization."""
    _require_columns(
        frame_proxy,
        {"label", "folder_rel", "green_excess", "green_ratio"},
    )
    summary = (
        frame_proxy.groupby(["label", "folder_rel"], sort=True)
        .agg(
            sampled_frames=("green_excess", "count"),
            green_excess_mean=("green_excess", "mean"),
            green_excess_median=("green_excess", "median"),
            green_ratio_mean=("green_ratio", "mean"),
        )
        .reset_index()
    )
    summary["green_uniform_proxy"] = summary["green_excess_median"]
    return summary


def rank_move_role_review_candidates(folder_proxy: pd.DataFrame) -> pd.DataFrame:
    """Rank move folders that visually look more employee-uniform-like."""
    _require_columns(folder_proxy, {"label", "folder_rel", "green_uniform_proxy"})
    move = folder_proxy[folder_proxy["label"] == "move"].copy()
    if move.empty:
        return move.assign(review_priority=pd.Series(dtype=float))
    move["review_priority"] = move["green_uniform_proxy"].rank(
        method="first", ascending=False
    )
    return move.sort_values(["review_priority", "folder_rel"]).reset_index(drop=True)


def _evenly_spaced_indexes(length: int, count: int) -> list[int]:
    if count <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [0]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing columns: {sorted(missing)}")

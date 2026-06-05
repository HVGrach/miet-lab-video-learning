from __future__ import annotations

import hashlib
import random

import pandas as pd


def assign_grouped_split(
    candidates: pd.DataFrame,
    *,
    val_fraction: float = 0.2,
    seed: int = 20260602,
    group_column: str = "source_track_id",
) -> pd.DataFrame:
    """Assign train/val split without splitting a source track across sets."""
    _require_columns(candidates, {"sample_id", "label", group_column})
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")

    result = candidates.copy()
    result["split"] = "train"
    if result.empty or val_fraction == 0:
        return result

    val_groups: set[str] = set()
    for label, group in result.groupby("label", sort=True):
        groups = sorted(str(value) for value in group[group_column].dropna().unique())
        if len(groups) <= 1:
            continue
        rng = random.Random(_stable_seed(seed, str(label)))
        rng.shuffle(groups)
        val_count = max(1, round(len(groups) * val_fraction))
        val_count = min(val_count, len(groups) - 1)
        val_groups.update(groups[:val_count])

    result.loc[result[group_column].astype(str).isin(val_groups), "split"] = "val"
    return result


def validate_grouped_split(
    split_manifest: pd.DataFrame,
    *,
    group_column: str = "source_track_id",
) -> dict[str, object]:
    """Report whether any source track leaks across train and validation."""
    _require_columns(split_manifest, {"sample_id", "label", group_column, "split"})
    overlapping: list[str] = []
    for group_id, group in split_manifest.groupby(group_column, sort=True):
        splits = set(group["split"])
        if len(splits) > 1:
            overlapping.append(str(group_id))

    split_counts = (
        split_manifest.groupby(["label", "split"], sort=True)
        .size()
        .reset_index(name="samples")
    )
    return {
        "is_valid": not overlapping,
        "overlapping_tracks": overlapping,
        "sample_count": int(len(split_manifest)),
        "track_count": int(split_manifest[group_column].nunique()),
        "split_counts": split_counts,
    }


def _stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing columns: {sorted(missing)}")

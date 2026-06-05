from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import re

import pandas as pd


FILENAME_TIME_RE = re.compile(r"(?P<time>\d{2}-\d{2}-\d{2})")


def extract_filename_time_token(file_name: str) -> str | None:
    """Extract HH-MM-SS-like tokens from lab6 frame names."""
    match = FILENAME_TIME_RE.search(Path(file_name).name)
    if not match:
        return None
    return match.group("time")


def build_file_metadata_manifest(
    dataset_root: str | Path, frame_manifest: pd.DataFrame
) -> pd.DataFrame:
    """Build file metadata and content hashes for every frame."""
    _require_columns(frame_manifest, {"label", "folder_rel", "file_name", "rel_path"})
    root = Path(dataset_root)
    rows: list[dict[str, object]] = []
    for _, frame in frame_manifest.iterrows():
        rel_path = str(frame["rel_path"])
        path = root / rel_path
        stat = path.stat()
        rows.append(
            {
                "label": frame["label"],
                "folder_rel": frame["folder_rel"],
                "file_name": frame["file_name"],
                "rel_path": rel_path,
                "file_size": int(stat.st_size),
                "md5": _md5_file(path),
                "filename_time_token": extract_filename_time_token(str(frame["file_name"])),
                "mtime": float(stat.st_mtime),
                "ctime": float(stat.st_ctime),
                "birthtime": float(getattr(stat, "st_birthtime", stat.st_mtime)),
            }
        )
    return pd.DataFrame(rows)


def build_duplicate_frame_audit(file_metadata: pd.DataFrame) -> pd.DataFrame:
    """Return one row per frame whose content hash appears more than once."""
    _require_columns(file_metadata, {"label", "folder_rel", "rel_path", "md5"})
    rows: list[dict[str, object]] = []
    for md5, group in file_metadata.groupby("md5", sort=True):
        if len(group) <= 1:
            continue
        labels = sorted(str(value) for value in group["label"].unique())
        folders = sorted(str(value) for value in group["folder_rel"].unique())
        for _, row in group.sort_values("rel_path").iterrows():
            duplicate = row.to_dict()
            duplicate["duplicate_cluster_id"] = f"md5::{md5}"
            duplicate["duplicate_count"] = int(len(group))
            duplicate["duplicate_folder_count"] = int(len(folders))
            duplicate["duplicate_label_count"] = int(len(labels))
            duplicate["duplicate_labels"] = ",".join(labels)
            duplicate["cross_label_duplicate"] = bool(len(labels) > 1)
            rows.append(duplicate)
    return pd.DataFrame(rows)


def explode_split_frame_hashes(
    split_manifest: pd.DataFrame, file_metadata: pd.DataFrame
) -> pd.DataFrame:
    """Attach frame hashes to every frame referenced by split samples."""
    _require_columns(
        split_manifest,
        {"sample_id", "label", "folder_rel", "split", "frame_paths"},
    )
    _require_columns(file_metadata, {"rel_path", "md5"})
    hash_by_path = dict(zip(file_metadata["rel_path"], file_metadata["md5"]))
    rows: list[dict[str, object]] = []
    for _, sample in split_manifest.iterrows():
        frame_paths = json.loads(sample["frame_paths"])
        for frame_index, rel_path in enumerate(frame_paths, start=1):
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "label": sample["label"],
                    "folder_rel": sample["folder_rel"],
                    "split": sample["split"],
                    "frame_index": frame_index,
                    "rel_path": rel_path,
                    "md5": hash_by_path[rel_path],
                }
            )
    return pd.DataFrame(rows)


def build_sample_overlap_audit(split_frame_hashes: pd.DataFrame) -> pd.DataFrame:
    """Find sample pairs that share exact frame content hashes."""
    _require_columns(
        split_frame_hashes,
        {"sample_id", "label", "folder_rel", "split", "md5"},
    )
    sample_meta = (
        split_frame_hashes.groupby("sample_id", sort=True)
        .agg(
            label=("label", "first"),
            folder_rel=("folder_rel", "first"),
            split=("split", "first"),
            hash_count=("md5", "nunique"),
        )
        .to_dict("index")
    )
    pair_hashes: dict[tuple[str, str], set[str]] = {}
    for md5, group in split_frame_hashes.groupby("md5", sort=True):
        sample_ids = sorted(str(value) for value in group["sample_id"].unique())
        if len(sample_ids) <= 1:
            continue
        for left, right in itertools.combinations(sample_ids, 2):
            pair_hashes.setdefault((left, right), set()).add(str(md5))

    rows: list[dict[str, object]] = []
    for (left, right), hashes in sorted(pair_hashes.items()):
        left_meta = sample_meta[left]
        right_meta = sample_meta[right]
        union_count = (
            int(left_meta["hash_count"]) + int(right_meta["hash_count"]) - len(hashes)
        )
        rows.append(
            {
                "sample_id_a": left,
                "sample_id_b": right,
                "label_a": left_meta["label"],
                "label_b": right_meta["label"],
                "folder_rel_a": left_meta["folder_rel"],
                "folder_rel_b": right_meta["folder_rel"],
                "split_a": left_meta["split"],
                "split_b": right_meta["split"],
                "overlap_hash_count": int(len(hashes)),
                "hash_jaccard": len(hashes) / union_count,
                "overlap_hashes": json.dumps(sorted(hashes)),
                "cross_label_overlap": bool(left_meta["label"] != right_meta["label"]),
                "cross_split_overlap": bool(left_meta["split"] != right_meta["split"]),
            }
        )
    return pd.DataFrame(rows)


def build_hash_linkage_candidates(duplicate_audit: pd.DataFrame) -> pd.DataFrame:
    """Create exact-evidence source group candidates from duplicate hashes."""
    _require_columns(
        duplicate_audit,
        {
            "duplicate_cluster_id",
            "md5",
            "label",
            "folder_rel",
            "rel_path",
            "duplicate_count",
            "cross_label_duplicate",
        },
    )
    rows: list[dict[str, object]] = []
    for cluster_id, group in duplicate_audit.groupby("duplicate_cluster_id", sort=True):
        folders = sorted(str(value) for value in group["folder_rel"].unique())
        if len(folders) <= 1:
            continue
        labels = sorted(str(value) for value in group["label"].unique())
        md5 = str(group["md5"].iloc[0])
        for folder_rel in folders:
            folder_rows = group[group["folder_rel"] == folder_rel]
            rows.append(
                {
                    "source_track_group_id": f"hash::{md5[:12]}",
                    "evidence_type": "hash_overlap",
                    "confidence": "exact",
                    "label": ",".join(labels),
                    "folder_rel": folder_rel,
                    "evidence_hash": md5,
                    "evidence_frame_count": int(len(folder_rows)),
                    "duplicate_cluster_id": cluster_id,
                    "cross_label_duplicate": bool(group["cross_label_duplicate"].any()),
                    "uses_mtime_for_link": False,
                }
            )
    return pd.DataFrame(rows)


def add_hash_guard_source_groups(
    manifest: pd.DataFrame,
    split_frame_hashes: pd.DataFrame,
    *,
    base_group_column: str = "source_track_id",
    virtual_group_column: str | None = None,
) -> pd.DataFrame:
    """Add stricter source groups by unioning base groups that share frame hashes."""
    _require_columns(manifest, {"sample_id", base_group_column})
    if virtual_group_column is not None:
        _require_columns(manifest, {virtual_group_column})
    _require_columns(split_frame_hashes, {"sample_id", "md5"})

    sample_to_group = dict(zip(manifest["sample_id"], manifest[base_group_column]))
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        parent[max(left_root, right_root)] = min(left_root, right_root)

    for group_value in manifest[base_group_column].astype(str).unique():
        find(group_value)

    if virtual_group_column is not None:
        for _, group in manifest.groupby(virtual_group_column, sort=True):
            base_groups = sorted(
                str(value)
                for value in group[base_group_column].dropna().unique()
            )
            if len(base_groups) <= 1:
                continue
            first = base_groups[0]
            for other in base_groups[1:]:
                union(first, other)

    frame_hashes = split_frame_hashes.copy()
    if base_group_column not in frame_hashes.columns:
        frame_hashes[base_group_column] = frame_hashes["sample_id"].map(sample_to_group)

    for _, group in frame_hashes.groupby("md5", sort=True):
        base_groups = sorted(
            str(value)
            for value in group[base_group_column].dropna().unique()
        )
        if len(base_groups) <= 1:
            continue
        first = base_groups[0]
        for other in base_groups[1:]:
            union(first, other)

    components: dict[str, list[str]] = {}
    for group_value in manifest[base_group_column].astype(str).unique():
        components.setdefault(find(group_value), []).append(group_value)

    group_to_guard: dict[str, str] = {}
    for component_groups in components.values():
        component_groups = sorted(component_groups)
        if len(component_groups) == 1:
            guard_id = component_groups[0]
        else:
            digest = hashlib.md5("|".join(component_groups).encode("utf-8")).hexdigest()
            guard_id = f"hash_guard::{digest[:12]}"
        for group_value in component_groups:
            group_to_guard[group_value] = guard_id

    result = manifest.copy()
    result["hash_guard_group_id"] = (
        result[base_group_column].astype(str).map(group_to_guard)
    )
    return result


def summarize_folder_time_tokens(frame_manifest: pd.DataFrame) -> pd.DataFrame:
    """Summarize filename-derived time tokens per folder."""
    _require_columns(frame_manifest, {"label", "folder_rel", "file_name"})
    rows: list[dict[str, object]] = []
    for (label, folder_rel), group in frame_manifest.groupby(
        ["label", "folder_rel"], sort=True
    ):
        tokens = [
            token
            for token in (extract_filename_time_token(name) for name in group["file_name"])
            if token is not None
        ]
        dominant = None
        dominant_count = 0
        if tokens:
            counts = pd.Series(tokens).value_counts()
            dominant = str(counts.index[0])
            dominant_count = int(counts.iloc[0])
        rows.append(
            {
                "label": label,
                "folder_rel": folder_rel,
                "frame_count": int(len(group)),
                "dominant_time_token": dominant,
                "time_token_count": dominant_count,
                "has_filename_time": bool(tokens),
                "unique_time_tokens": int(len(set(tokens))),
            }
        )
    return pd.DataFrame(rows)


def add_folder_mtime(
    dataset_root: str | Path, folder_summary: pd.DataFrame
) -> pd.DataFrame:
    """Attach folder mtimes for audit only; mtimes are not used for grouping."""
    _require_columns(folder_summary, {"folder_rel"})
    root = Path(dataset_root)
    result = folder_summary.copy()
    result["folder_mtime"] = [
        (root / folder_rel).stat().st_mtime if (root / folder_rel).exists() else None
        for folder_rel in result["folder_rel"]
    ]
    return result


def build_track_link_candidates(folder_summary: pd.DataFrame) -> pd.DataFrame:
    """Find virtual source-group candidates using conservative filename evidence."""
    _require_columns(
        folder_summary,
        {
            "label",
            "folder_rel",
            "frame_count",
            "dominant_time_token",
            "time_token_count",
            "has_filename_time",
        },
    )
    rows: list[dict[str, object]] = []
    eligible = folder_summary[
        folder_summary["has_filename_time"]
        & folder_summary["dominant_time_token"].notna()
    ].copy()
    if eligible.empty:
        return pd.DataFrame(columns=_candidate_columns())

    for (label, token), group in eligible.groupby(
        ["label", "dominant_time_token"], sort=True
    ):
        if len(group) < 2:
            continue
        group_id = f"{label}::filename_time::{token}"
        for _, row in group.sort_values("folder_rel").iterrows():
            rows.append(
                {
                    "source_track_group_id": group_id,
                    "label": label,
                    "folder_rel": row["folder_rel"],
                    "frame_count": int(row["frame_count"]),
                    "dominant_time_token": token,
                    "time_token_count": int(row["time_token_count"]),
                    "link_reason": "same_label_filename_time",
                    "confidence": "review",
                    "uses_mtime_for_link": False,
                }
            )
    return pd.DataFrame(rows, columns=_candidate_columns())


def apply_virtual_source_groups(
    manifest: pd.DataFrame, track_link_candidates: pd.DataFrame
) -> pd.DataFrame:
    """Add source_track_group_id to a manifest without changing physical folders."""
    _require_columns(manifest, {"folder_rel"})
    result = manifest.copy()
    result["source_track_group_id"] = result.get(
        "source_track_group_id", result["folder_rel"]
    )
    if track_link_candidates.empty:
        return result
    _require_columns(track_link_candidates, {"folder_rel", "source_track_group_id"})
    mapping = dict(
        zip(
            track_link_candidates["folder_rel"],
            track_link_candidates["source_track_group_id"],
        )
    )
    result["source_track_group_id"] = result["folder_rel"].map(mapping).fillna(
        result["source_track_group_id"]
    )
    return result


def _candidate_columns() -> list[str]:
    return [
        "source_track_group_id",
        "label",
        "folder_rel",
        "frame_count",
        "dominant_time_token",
        "time_token_count",
        "link_reason",
        "confidence",
        "uses_mtime_for_link",
    ]


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing columns: {sorted(missing)}")

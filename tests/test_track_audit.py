import os
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from lab6_analysis.track_audit import (
    add_hash_guard_source_groups,
    build_duplicate_frame_audit,
    build_file_metadata_manifest,
    build_hash_linkage_candidates,
    build_sample_overlap_audit,
    build_track_link_candidates,
    extract_filename_time_token,
    explode_split_frame_hashes,
    summarize_folder_time_tokens,
)
from lab6_analysis.splitting import assign_grouped_split
from lab6_analysis.sampling import sort_frame_paths_temporally


class TrackAuditTests(unittest.TestCase):
    def test_extract_filename_time_token(self):
        self.assertEqual(
            extract_filename_time_token("im000001_00-37-05.jpg"),
            "00-37-05",
        )
        self.assertEqual(
            extract_filename_time_token("im_01-32-52_10.jpg"),
            "01-32-52",
        )
        self.assertIsNone(extract_filename_time_token("im1.jpg"))

    def test_summarize_folder_time_tokens_uses_filename_not_mtime(self):
        frame_manifest = pd.DataFrame(
            {
                "label": ["move", "move", "work"],
                "folder_rel": ["move/a", "move/a", "work/b"],
                "file_name": [
                    "im000001_00-37-05.jpg",
                    "im000002_00-37-05.jpg",
                    "im1.jpg",
                ],
                "rel_path": [
                    "move/a/im000001_00-37-05.jpg",
                    "move/a/im000002_00-37-05.jpg",
                    "work/b/im1.jpg",
                ],
            }
        )

        summary = summarize_folder_time_tokens(frame_manifest)

        row = summary.set_index("folder_rel").loc["move/a"]
        self.assertEqual(row["dominant_time_token"], "00-37-05")
        self.assertEqual(row["time_token_count"], 2)
        self.assertTrue(row["has_filename_time"])

    def test_build_track_link_candidates_does_not_merge_by_mtime_only(self):
        folder_summary = pd.DataFrame(
            {
                "label": ["move", "move"],
                "folder_rel": ["move/a", "move/b"],
                "frame_count": [8, 8],
                "dominant_time_token": [None, None],
                "time_token_count": [0, 0],
                "has_filename_time": [False, False],
                "folder_mtime": [1000.0, 1000.0],
            }
        )

        candidates = build_track_link_candidates(folder_summary)

        self.assertTrue(candidates.empty)

    def test_build_track_link_candidates_groups_same_label_time_token(self):
        folder_summary = pd.DataFrame(
            {
                "label": ["move", "move", "work"],
                "folder_rel": ["move/a", "move/b", "work/c"],
                "frame_count": [8, 10, 12],
                "dominant_time_token": ["00-37-05", "00-37-05", "00-37-05"],
                "time_token_count": [8, 10, 12],
                "has_filename_time": [True, True, True],
            }
        )

        candidates = build_track_link_candidates(folder_summary)

        self.assertEqual(len(candidates), 2)
        self.assertEqual(set(candidates["folder_rel"]), {"move/a", "move/b"})
        self.assertEqual(set(candidates["link_reason"]), {"same_label_filename_time"})
        self.assertEqual(candidates["source_track_group_id"].nunique(), 1)

    def test_temporal_sort_ignores_filesystem_mtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "move" / "track"
            folder.mkdir(parents=True)
            paths = []
            for index in [2, 1, 3]:
                path = folder / f"im_00-00-01_{index}.jpg"
                Image.new("RGB", (10, 10), color="white").save(path)
                os.utime(path, (1000 + (10 - index), 1000 + (10 - index)))
                paths.append(str(path.relative_to(root)))

            self.assertEqual(
                sort_frame_paths_temporally(paths),
                [
                    "move/track/im_00-00-01_1.jpg",
                    "move/track/im_00-00-01_2.jpg",
                    "move/track/im_00-00-01_3.jpg",
                ],
            )

    def test_build_file_metadata_manifest_hashes_frames_without_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            move = root / "move" / "a"
            move.mkdir(parents=True)
            Image.new("RGB", (8, 8), color="green").save(
                move / "im000001_00-37-05.jpg"
            )
            frame_manifest = pd.DataFrame(
                {
                    "label": ["move"],
                    "folder_rel": ["move/a"],
                    "file_name": ["im000001_00-37-05.jpg"],
                    "rel_path": ["move/a/im000001_00-37-05.jpg"],
                }
            )

            metadata = build_file_metadata_manifest(root, frame_manifest)

            self.assertEqual(len(metadata), 1)
            self.assertNotIn("path", metadata.columns)
            self.assertEqual(metadata.iloc[0]["filename_time_token"], "00-37-05")
            self.assertEqual(len(metadata.iloc[0]["md5"]), 32)
            self.assertGreater(metadata.iloc[0]["file_size"], 0)

    def test_duplicate_frame_audit_flags_cross_label_duplicates(self):
        metadata = pd.DataFrame(
            {
                "label": ["move", "work", "move"],
                "folder_rel": ["move/a", "work/b", "move/c"],
                "rel_path": ["move/a/1.jpg", "work/b/1.jpg", "move/c/2.jpg"],
                "md5": ["same", "same", "unique"],
            }
        )

        audit = build_duplicate_frame_audit(metadata)

        self.assertEqual(len(audit), 2)
        self.assertEqual(set(audit["duplicate_cluster_id"]), {"md5::same"})
        self.assertTrue(audit["cross_label_duplicate"].all())

    def test_sample_overlap_audit_flags_train_val_hash_overlap(self):
        metadata = pd.DataFrame(
            {
                "rel_path": [
                    "move/a/1.jpg",
                    "move/a/2.jpg",
                    "move/b/1.jpg",
                    "move/b/3.jpg",
                ],
                "md5": ["shared", "a_only", "shared", "b_only"],
            }
        )
        split = pd.DataFrame(
            {
                "sample_id": ["a", "b"],
                "label": ["move", "move"],
                "folder_rel": ["move/a", "move/b"],
                "split": ["train", "val"],
                "frame_paths": [
                    json.dumps(["move/a/1.jpg", "move/a/2.jpg"]),
                    json.dumps(["move/b/1.jpg", "move/b/3.jpg"]),
                ],
            }
        )

        frame_hashes = explode_split_frame_hashes(split, metadata)
        overlap = build_sample_overlap_audit(frame_hashes)

        self.assertEqual(len(overlap), 1)
        self.assertEqual(overlap.iloc[0]["overlap_hash_count"], 1)
        self.assertTrue(overlap.iloc[0]["cross_split_overlap"])

    def test_hash_linkage_candidates_use_exact_hash_evidence(self):
        duplicate_audit = pd.DataFrame(
            {
                "duplicate_cluster_id": ["md5::abc", "md5::abc"],
                "md5": ["abc", "abc"],
                "label": ["move", "work"],
                "folder_rel": ["move/a", "work/b"],
                "rel_path": ["move/a/1.jpg", "work/b/1.jpg"],
                "duplicate_count": [2, 2],
                "cross_label_duplicate": [True, True],
            }
        )

        candidates = build_hash_linkage_candidates(duplicate_audit)

        self.assertEqual(len(candidates), 2)
        self.assertEqual(set(candidates["evidence_type"]), {"hash_overlap"})
        self.assertEqual(set(candidates["confidence"]), {"exact"})
        self.assertFalse(candidates["uses_mtime_for_link"].any())

    def test_hash_guard_source_groups_connect_samples_with_shared_hash(self):
        split = pd.DataFrame(
            {
                "sample_id": ["a", "b", "c"],
                "label": ["move", "move", "work"],
                "folder_rel": ["move/a", "move/b", "work/c"],
                "source_track_id": ["move/a", "move/b", "work/c"],
                "frame_paths": ["[]", "[]", "[]"],
            }
        )
        frame_hashes = pd.DataFrame(
            {
                "sample_id": ["a", "b", "c"],
                "source_track_id": ["move/a", "move/b", "work/c"],
                "md5": ["shared", "shared", "unique"],
            }
        )

        guarded = add_hash_guard_source_groups(split, frame_hashes)

        by_sample = guarded.set_index("sample_id")
        self.assertEqual(
            by_sample.loc["a", "hash_guard_group_id"],
            by_sample.loc["b", "hash_guard_group_id"],
        )
        self.assertNotEqual(
            by_sample.loc["a", "hash_guard_group_id"],
            by_sample.loc["c", "hash_guard_group_id"],
        )

    def test_hash_guard_source_groups_can_include_virtual_groups(self):
        split = pd.DataFrame(
            {
                "sample_id": ["a", "b", "c"],
                "label": ["move", "move", "work"],
                "folder_rel": ["move/a", "move/b", "work/c"],
                "source_track_id": ["move/a", "move/b", "work/c"],
                "source_track_group_id": ["move::t", "move::t", "work/c"],
                "frame_paths": ["[]", "[]", "[]"],
            }
        )
        frame_hashes = pd.DataFrame(
            {
                "sample_id": ["a", "b", "c"],
                "source_track_id": ["move/a", "move/b", "work/c"],
                "md5": ["a_hash", "b_hash", "c_hash"],
            }
        )

        guarded = add_hash_guard_source_groups(
            split,
            frame_hashes,
            virtual_group_column="source_track_group_id",
        )

        by_sample = guarded.set_index("sample_id")
        self.assertEqual(
            by_sample.loc["a", "hash_guard_group_id"],
            by_sample.loc["b", "hash_guard_group_id"],
        )
        self.assertNotEqual(
            by_sample.loc["a", "hash_guard_group_id"],
            by_sample.loc["c", "hash_guard_group_id"],
        )

    def test_hash_guarded_split_keeps_shared_hash_group_together(self):
        split = pd.DataFrame(
            {
                "sample_id": [f"s{i}" for i in range(6)],
                "label": ["move"] * 6,
                "folder_rel": [f"move/{i}" for i in range(6)],
                "source_track_id": [f"move/{i}" for i in range(6)],
                "frame_paths": ["[]"] * 6,
            }
        )
        frame_hashes = pd.DataFrame(
            {
                "sample_id": ["s0", "s1", "s2", "s3", "s4", "s5"],
                "source_track_id": [f"move/{i}" for i in range(6)],
                "md5": ["shared", "shared", "h2", "h3", "h4", "h5"],
            }
        )

        guarded = add_hash_guard_source_groups(split, frame_hashes)
        assigned = assign_grouped_split(
            guarded,
            group_column="hash_guard_group_id",
            val_fraction=0.5,
            seed=1,
        )

        shared_splits = set(assigned[assigned["sample_id"].isin(["s0", "s1"])]["split"])
        self.assertEqual(len(shared_splits), 1)


if __name__ == "__main__":
    unittest.main()

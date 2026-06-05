import unittest

import pandas as pd

from lab6_analysis.role_audit import (
    build_manual_role_seed_manifest,
    rank_move_role_review_candidates,
    summarize_green_proxy_by_folder,
    validate_model_feature_columns,
    validate_role_labels_are_manual,
)


class RoleAuditTests(unittest.TestCase):
    def test_validate_model_feature_columns_rejects_leaky_metadata(self):
        with self.assertRaises(ValueError):
            validate_model_feature_columns(
                [
                    "embedding_0",
                    "embedding_1",
                    "folder_rel",
                    "mtime",
                    "md5",
                    "hash_guard_group_id",
                    "filename_time_token",
                    "file_size",
                    "ctime",
                    "birthtime",
                ]
            )
        with self.assertRaises(ValueError):
            validate_model_feature_columns(["embedding_0", "md5"])
        with self.assertRaises(ValueError):
            validate_model_feature_columns(["embedding_0", "hash_guard_group_id"])
        with self.assertRaises(ValueError):
            validate_model_feature_columns(["embedding_0", "sample_kind"])
        with self.assertRaises(ValueError):
            validate_model_feature_columns(["embedding_0", "stride", "dilation"])
        with self.assertRaises(ValueError):
            validate_model_feature_columns(["embedding_0", "source_frame_count"])

        validate_model_feature_columns(["embedding_0", "embedding_1"])

    def test_validate_role_labels_are_manual_rejects_weak_action_labels(self):
        role_labels = pd.DataFrame(
            {
                "sample_id": ["a", "b"],
                "role_label": ["employee", "customer"],
                "role_label_source": ["weak_from_action", "manual"],
            }
        )

        with self.assertRaises(ValueError):
            validate_role_labels_are_manual(role_labels)

    def test_build_manual_role_seed_manifest_marks_unknown_for_review(self):
        sample_manifest = pd.DataFrame(
            {
                "sample_id": ["move/a", "work/b"],
                "label": ["move", "work"],
                "folder_rel": ["move/a", "work/b"],
                "frame_paths": ["[]", "[]"],
            }
        )

        seed = build_manual_role_seed_manifest(sample_manifest)

        self.assertEqual(set(seed["role_label"]), {"unknown"})
        self.assertEqual(set(seed["role_label_source"]), {"manual_review_needed"})
        self.assertEqual(seed["sample_id"].tolist(), ["move/a", "work/b"])

    def test_summarize_green_proxy_and_rank_move_review_candidates(self):
        frame_proxy = pd.DataFrame(
            {
                "label": ["move", "move", "move", "work"],
                "folder_rel": ["move/a", "move/a", "move/b", "work/c"],
                "green_excess": [20.0, 18.0, 2.0, 19.0],
                "green_ratio": [0.39, 0.38, 0.34, 0.39],
            }
        )

        summary = summarize_green_proxy_by_folder(frame_proxy)
        ranked = rank_move_role_review_candidates(summary)

        self.assertEqual(ranked.iloc[0]["folder_rel"], "move/a")
        self.assertLess(ranked.iloc[0]["review_priority"], ranked.iloc[1]["review_priority"])
        self.assertIn("green_uniform_proxy", ranked.columns)


if __name__ == "__main__":
    unittest.main()

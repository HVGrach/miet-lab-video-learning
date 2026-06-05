import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"


class RoleTrackOutputsContractTests(unittest.TestCase):
    def test_role_seed_manifest_is_manual_review_only(self):
        role_seed = pd.read_csv(
            OUTPUTS / "manifests" / "manual_role_seed_manifest.csv"
        )

        self.assertEqual(len(role_seed), 364)
        self.assertEqual(set(role_seed["role_label"]), {"unknown"})
        self.assertEqual(set(role_seed["role_label_source"]), {"manual_review_needed"})

    def test_virtual_track_link_candidates_are_filename_based(self):
        links = pd.read_csv(
            OUTPUTS / "reports" / "virtual_track_link_candidates.csv"
        )

        self.assertEqual(len(links), 4)
        self.assertEqual(set(links["link_reason"]), {"same_label_filename_time"})
        self.assertFalse(links["uses_mtime_for_link"].any())
        self.assertEqual(
            links.groupby("source_track_group_id").size().to_dict(),
            {
                "move::filename_time::00-28-44": 2,
                "work::filename_time::00-23-56": 2,
            },
        )

    def test_virtual_track_group_split_audit_detects_current_warning(self):
        split = pd.read_csv(
            OUTPUTS / "manifests" / "train_val_split_with_virtual_track_groups.csv"
        )
        leaking = (
            split.groupby("source_track_group_id")["split"]
            .nunique()
            .loc[lambda s: s > 1]
        )

        self.assertEqual(leaking.index.tolist(), ["move::filename_time::00-28-44"])

    def test_role_track_reports_exist(self):
        private_artifacts = [
            "reports/role_green_proxy_frames.csv",
        ]
        public_artifacts = [
            "reports/role_green_proxy_folder_summary.csv",
            "reports/move_role_review_candidates.csv",
            "reports/folder_time_token_summary.csv",
            "reports/virtual_track_group_split_summary.csv",
            "reports/role_track_acceptance.md",
        ]
        for rel_path in public_artifacts:
            path = OUTPUTS / rel_path
            self.assertTrue(path.exists(), rel_path)
            self.assertGreater(path.stat().st_size, 0, rel_path)
        for rel_path in private_artifacts:
            path = OUTPUTS / rel_path
            if path.exists():
                self.assertGreater(path.stat().st_size, 0, rel_path)


if __name__ == "__main__":
    unittest.main()

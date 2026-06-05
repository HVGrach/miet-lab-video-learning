import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"


class SourceLinkageOutputsContractTests(unittest.TestCase):
    @unittest.skipUnless(
        (OUTPUTS / "manifests" / "file_metadata_manifest.csv").exists(),
        "private per-frame metadata artifact is not published",
    )
    def test_file_metadata_and_duplicate_hash_contract(self):
        metadata = pd.read_csv(OUTPUTS / "manifests" / "file_metadata_manifest.csv")
        clusters = pd.read_csv(
            OUTPUTS / "reports" / "duplicate_frame_cluster_summary.csv"
        )

        self.assertEqual(len(metadata), 24170)
        self.assertNotIn("path", metadata.columns)
        self.assertTrue(metadata["md5"].str.len().eq(32).all())
        self.assertEqual(len(clusters), 369)
        self.assertEqual(int(clusters["cross_label_duplicate"].sum()), 51)

    def test_original_split_hash_leakage_is_documented(self):
        if not (OUTPUTS / "manifests" / "sample_overlap_audit.csv").exists():
            self.skipTest("private sample overlap audit artifact is not published")
        leakage = pd.read_csv(OUTPUTS / "reports" / "hash_split_leakage_summary.csv")
        overlap = pd.read_csv(OUTPUTS / "manifests" / "sample_overlap_audit.csv")
        summary = pd.read_csv(
            OUTPUTS / "reports" / "source_group_split_leakage_summary.csv"
        )

        self.assertEqual(len(leakage), 22)
        self.assertEqual(int(overlap["cross_split_overlap"].sum()), 28)
        by_audit = summary.set_index("audit")
        self.assertEqual(by_audit.loc["original_hash_split_leakage", "leaking_hashes"], 22)

    def test_hash_guarded_split_has_no_hash_leakage(self):
        guarded = pd.read_csv(
            OUTPUTS / "manifests" / "train_val_split_hash_guarded.csv"
        )
        guarded_leakage = pd.read_csv(
            OUTPUTS / "reports" / "hash_guarded_split_leakage_summary.csv"
        )
        virtual_leakage = pd.read_csv(
            OUTPUTS / "reports" / "hash_virtual_guarded_split_leakage_summary.csv"
        )

        self.assertEqual(len(guarded), 1546)
        self.assertIn("hash_guard_group_id", guarded.columns)
        self.assertIn("source_track_group_id", guarded.columns)
        self.assertEqual(guarded["hash_guard_group_id"].nunique(), 272)
        self.assertEqual(len(guarded_leakage), 0)
        self.assertEqual(len(virtual_leakage), 0)
        self.assertEqual(
            guarded.groupby(["label", "split"]).size().to_dict(),
            {
                ("inaction", "train"): 455,
                ("inaction", "val"): 117,
                ("move", "train"): 323,
                ("move", "val"): 105,
                ("work", "train"): 444,
                ("work", "val"): 102,
            },
        )

    def test_source_linkage_artifacts_exist(self):
        private_artifacts = [
            "manifests/duplicate_frame_audit.csv",
            "manifests/split_frame_hashes.csv",
            "manifests/hash_source_linkage_candidates.csv",
        ]
        public_artifacts = [
            "reports/hash_guarded_split_summary.csv",
            "reports/hash_guarded_split_leakage_summary.csv",
            "reports/hash_virtual_guarded_split_leakage_summary.csv",
            "reports/source_track_linkage_acceptance.md",
            "reports/source_group_split_leakage_summary.csv",
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

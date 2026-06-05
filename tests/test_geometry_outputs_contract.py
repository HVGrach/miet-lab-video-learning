import json
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "lab6"
OUTPUTS = PROJECT_ROOT / "outputs"


class GeometryOutputsContractTests(unittest.TestCase):
    @unittest.skipUnless(DATASET_ROOT.exists(), "local lab6 dataset is not published")
    def test_generated_manifests_match_current_dataset_contract(self):
        frame_manifest = pd.read_csv(OUTPUTS / "manifests" / "frame_manifest.csv")
        sample_manifest = pd.read_csv(OUTPUTS / "manifests" / "sample_manifest.csv")

        self.assertEqual(len(frame_manifest), 24170)
        self.assertEqual(set(frame_manifest["label"]), {"inaction", "move", "work"})
        self.assertNotIn("path", frame_manifest.columns)
        self.assertIn("mode", frame_manifest.columns)
        self.assertTrue(frame_manifest["rel_path"].is_unique)

        self.assertEqual(len(sample_manifest), 364)
        self.assertTrue(sample_manifest["sample_id"].is_unique)
        self.assertTrue((sample_manifest["frame_count"] == 8).all())
        self.assertEqual(
            sample_manifest.groupby(["label", "sample_kind"]).size().to_dict(),
            {
                ("inaction", "folder_8"): 4,
                ("inaction", "prefixed_8"): 64,
                ("move", "folder_8"): 18,
                ("move", "prefixed_8"): 31,
                ("work", "folder_8"): 23,
                ("work", "prefixed_8"): 222,
                ("work", "remaining_8"): 2,
            },
        )

        for _, sample in sample_manifest.iterrows():
            frame_paths = json.loads(sample["frame_paths"])
            self.assertEqual(len(frame_paths), 8)
            for rel_path in frame_paths:
                self.assertTrue((DATASET_ROOT / rel_path).exists(), rel_path)

    def test_generated_visual_artifacts_exist(self):
        public_artifacts = [
            "eda/aspect_ratio_with_height_inset.png",
            "eda/height_distribution_by_class.png",
            "eda/width_height_scatter.png",
            "manifests/sample_frames.csv",
            "reports/image_geometry_summary.csv",
            "reports/folder_geometry_summary.csv",
            "reports/sample_geometry_summary.csv",
            "reports/image_geometry_outliers.csv",
            "reports/image_geometry_outlier_reason_summary.csv",
            "reports/geometry_eda_acceptance.md",
        ]
        private_artifacts = [
            "eda/geometry_outliers_contact_sheet.png",
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

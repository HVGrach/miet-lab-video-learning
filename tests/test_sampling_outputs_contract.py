import json
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "lab6"
OUTPUTS = PROJECT_ROOT / "outputs"


class SamplingOutputsContractTests(unittest.TestCase):
    @unittest.skipUnless(DATASET_ROOT.exists(), "local lab6 dataset is not published")
    def test_training_candidates_match_default_sampling_contract(self):
        candidates = pd.read_csv(
            OUTPUTS / "manifests" / "training_sample_candidates.csv"
        )

        self.assertEqual(len(candidates), 1546)
        self.assertNotIn("path", candidates.columns)
        self.assertTrue(candidates["sample_id"].is_unique)
        self.assertTrue((candidates["frame_count"] == 8).all())
        self.assertNotIn("inaction_skip_shuffle", set(candidates["sample_kind"]))
        self.assertEqual(
            candidates.groupby("label").size().to_dict(),
            {"inaction": 572, "move": 428, "work": 546},
        )
        self.assertEqual(
            candidates.groupby(["label", "sample_kind"]).size().to_dict(),
            {
                ("inaction", "explicit_folder_8"): 4,
                ("inaction", "explicit_prefixed_8"): 64,
                ("inaction", "inaction_skip_ordered"): 504,
                ("move", "explicit_folder_8"): 18,
                ("move", "explicit_prefixed_8"): 31,
                ("move", "stride_dilation"): 379,
                ("work", "explicit_folder_8"): 23,
                ("work", "explicit_prefixed_8"): 222,
                ("work", "explicit_remaining_8"): 2,
                ("work", "stride_dilation"): 299,
            },
        )

        for _, sample in candidates.iterrows():
            frame_paths = json.loads(sample["frame_paths"])
            frame_indices = json.loads(sample["frame_indices"])
            self.assertEqual(len(frame_paths), 8)
            self.assertEqual(len(frame_indices), 8)
            for rel_path in frame_paths:
                self.assertTrue((DATASET_ROOT / rel_path).exists(), rel_path)
            if sample["sample_kind"] == "stride_dilation":
                start = int(sample["start_index"])
                dilation = int(sample["dilation"])
                self.assertEqual(
                    frame_indices,
                    [start + index * dilation for index in range(8)],
                )

    def test_train_val_split_has_no_track_leakage(self):
        split = pd.read_csv(OUTPUTS / "manifests" / "train_val_split.csv")

        self.assertEqual(len(split), 1546)
        self.assertEqual(set(split["split"]), {"train", "val"})
        leaking_tracks = (
            split.groupby("source_track_id")["split"].nunique().loc[lambda s: s > 1]
        )
        self.assertEqual(leaking_tracks.index.tolist(), [])
        self.assertEqual(
            split.groupby(["label", "split"]).size().to_dict(),
            {
                ("inaction", "train"): 449,
                ("inaction", "val"): 123,
                ("move", "train"): 337,
                ("move", "val"): 91,
                ("work", "train"): 422,
                ("work", "val"): 124,
            },
        )

    def test_sampling_visual_and_reports_exist(self):
        public_artifacts = [
            "manifests/inaction_shuffle_ablation_candidates.csv",
            "reports/training_candidate_summary.csv",
            "reports/train_val_split_summary.csv",
            "reports/sampling_preprocessing_acceptance.md",
        ]
        private_artifacts = [
            "eda/preprocessing_sequence_contact_sheet.png",
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

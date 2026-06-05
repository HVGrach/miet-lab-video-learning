import unittest

import pandas as pd

from lab6_analysis.splitting import assign_grouped_split, validate_grouped_split


class SplittingTests(unittest.TestCase):
    def test_assign_grouped_split_keeps_tracks_disjoint(self):
        rows = []
        for label in ("inaction", "move", "work"):
            for track_index in range(6):
                for sample_index in range(2):
                    rows.append(
                        {
                            "sample_id": f"{label}/track_{track_index}::{sample_index}",
                            "label": label,
                            "source_track_id": f"{label}/track_{track_index}",
                        }
                    )
        candidates = pd.DataFrame(rows)

        split = assign_grouped_split(candidates, val_fraction=0.33, seed=42)
        report = validate_grouped_split(split)

        self.assertTrue(report["is_valid"])
        self.assertEqual(report["overlapping_tracks"], [])
        self.assertEqual(set(split["split"]), {"train", "val"})
        for label in ("inaction", "move", "work"):
            label_split = split[split["label"] == label]
            self.assertIn("train", set(label_split["split"]))
            self.assertIn("val", set(label_split["split"]))

    def test_assign_grouped_split_is_deterministic(self):
        candidates = pd.DataFrame(
            {
                "sample_id": [f"move/track_{i}::0" for i in range(10)],
                "label": ["move"] * 10,
                "source_track_id": [f"move/track_{i}" for i in range(10)],
            }
        )

        first = assign_grouped_split(candidates, val_fraction=0.3, seed=5)
        second = assign_grouped_split(candidates, val_fraction=0.3, seed=5)

        self.assertEqual(first["split"].tolist(), second["split"].tolist())

    def test_validate_grouped_split_detects_track_leakage(self):
        leaked = pd.DataFrame(
            {
                "sample_id": ["a", "b"],
                "label": ["move", "move"],
                "source_track_id": ["move/track", "move/track"],
                "split": ["train", "val"],
            }
        )

        report = validate_grouped_split(leaked)

        self.assertFalse(report["is_valid"])
        self.assertEqual(report["overlapping_tracks"], ["move/track"])

    def test_validate_grouped_split_can_use_virtual_source_group(self):
        leaked_group = pd.DataFrame(
            {
                "sample_id": ["a", "b"],
                "label": ["move", "move"],
                "source_track_id": ["move/a", "move/b"],
                "source_track_group_id": ["move::00-37-05", "move::00-37-05"],
                "split": ["train", "val"],
            }
        )

        report = validate_grouped_split(
            leaked_group,
            group_column="source_track_group_id",
        )

        self.assertFalse(report["is_valid"])
        self.assertEqual(report["overlapping_tracks"], ["move::00-37-05"])


if __name__ == "__main__":
    unittest.main()

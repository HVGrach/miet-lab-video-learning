import tempfile
import unittest
import json
from pathlib import Path

from PIL import Image

from lab6_analysis.data_audit import (
    build_frame_manifest,
    build_sample_manifest,
    explode_sample_frames,
    find_geometry_outliers,
    summarize_folder_geometry,
    summarize_sample_geometry,
    summarize_dimension_stats,
)


class FrameManifestTests(unittest.TestCase):
    def test_build_frame_manifest_records_dimensions_and_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "move" / "sample_001"
            sample.mkdir(parents=True)
            Image.new("RGB", (20, 40), color="red").save(sample / "im1.jpg")
            Image.new("RGB", (30, 10), color="blue").save(sample / "im2.jpg")

            manifest = build_frame_manifest(root)

            self.assertEqual(len(manifest), 2)
            self.assertEqual(set(manifest["label"]), {"move"})
            self.assertEqual(set(manifest["folder_rel"]), {"move/sample_001"})
            by_name = manifest.set_index("file_name")
            self.assertEqual(by_name.loc["im1.jpg", "width"], 20)
            self.assertEqual(by_name.loc["im1.jpg", "height"], 40)
            self.assertEqual(by_name.loc["im1.jpg", "mode"], "RGB")
            self.assertAlmostEqual(by_name.loc["im1.jpg", "aspect_ratio"], 0.5)
            self.assertEqual(by_name.loc["im2.jpg", "area"], 300)
            self.assertNotIn("path", manifest.columns)

    def test_build_sample_manifest_recognizes_prefixed_eight_frame_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "work" / "multi_window_track"
            sample.mkdir(parents=True)
            for prefix in ("0", "1"):
                for index in range(1, 9):
                    Image.new("RGB", (10 + index, 20), color="green").save(
                        sample / f"{prefix}_im{index}.jpg"
                    )

            samples = build_sample_manifest(root)

            self.assertEqual(len(samples), 2)
            self.assertEqual(set(samples["label"]), {"work"})
            self.assertEqual(set(samples["frame_count"]), {8})
            self.assertEqual(set(samples["sample_kind"]), {"prefixed_8"})
            self.assertEqual(set(samples["group_key"]), {"0", "1"})
            first_paths = json.loads(samples.iloc[0]["frame_paths"])
            self.assertEqual(len(first_paths), 8)
            self.assertTrue(first_paths[0].endswith("_im1.jpg"))
            self.assertTrue(first_paths[-1].endswith("_im8.jpg"))

    def test_explode_sample_frames_returns_one_row_per_ordered_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "inaction" / "plain"
            sample.mkdir(parents=True)
            for index in range(1, 9):
                Image.new("RGB", (10, 10), color="black").save(sample / f"im{index}.jpg")

            samples = build_sample_manifest(root)
            frames = explode_sample_frames(samples)

            self.assertEqual(len(frames), 8)
            self.assertEqual(frames["frame_index"].tolist(), list(range(1, 9)))
            self.assertEqual(frames["label"].unique().tolist(), ["inaction"])
            self.assertTrue(frames.iloc[0]["rel_path"].endswith("im1.jpg"))
            self.assertTrue(frames.iloc[-1]["rel_path"].endswith("im8.jpg"))

    def test_build_sample_manifest_marks_unmatched_remaining_eight_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "work" / "mixed"
            sample.mkdir(parents=True)
            for index in range(1, 9):
                Image.new("RGB", (10, 10), color="white").save(sample / f"0_im{index}.jpg")
                Image.new("RGB", (10, 10), color="white").save(sample / f"im{index}.jpg")

            samples = build_sample_manifest(root)

            self.assertEqual(set(samples["sample_kind"]), {"prefixed_8", "remaining_8"})
            remaining = samples[samples["sample_kind"] == "remaining_8"].iloc[0]
            self.assertEqual(remaining["group_key"], "remaining")

    def test_summarize_dimension_stats_reports_classwise_quantiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label, sizes in {
                "inaction": [(10, 20), (20, 20), (30, 10)],
                "work": [(40, 40), (40, 80)],
            }.items():
                sample = root / label / "sample"
                sample.mkdir(parents=True)
                for index, size in enumerate(sizes, start=1):
                    Image.new("RGB", size, color="white").save(sample / f"im{index}.jpg")

            manifest = build_frame_manifest(root)
            summary = summarize_dimension_stats(manifest)

            by_label = summary.set_index("label")
            self.assertEqual(by_label.loc["inaction", "frame_count"], 3)
            self.assertEqual(by_label.loc["inaction", "height_min"], 10)
            self.assertEqual(by_label.loc["inaction", "height_median"], 20)
            self.assertEqual(by_label.loc["work", "unique_sizes"], 2)
            self.assertAlmostEqual(by_label.loc["work", "aspect_ratio_median"], 0.75)

    def test_find_geometry_outliers_marks_small_and_extreme_aspect_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "move" / "sample"
            sample.mkdir(parents=True)
            Image.new("RGB", (20, 20), color="white").save(sample / "small.jpg")
            Image.new("RGB", (300, 50), color="white").save(sample / "wide.jpg")
            Image.new("RGB", (100, 160), color="white").save(sample / "normal.jpg")

            manifest = build_frame_manifest(root)
            outliers = find_geometry_outliers(
                manifest,
                min_height=64,
                min_width=32,
                aspect_low=0.3,
                aspect_high=3.0,
            )

            self.assertEqual(set(outliers["file_name"]), {"small.jpg", "wide.jpg"})
            reasons = dict(zip(outliers["file_name"], outliers["outlier_reason"]))
            self.assertIn("small_width", reasons["small.jpg"])
            self.assertIn("small_height", reasons["small.jpg"])
            self.assertIn("high_aspect", reasons["wide.jpg"])

    def test_summarize_folder_geometry_reports_within_folder_variability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "work" / "track_001"
            folder.mkdir(parents=True)
            for index, size in enumerate([(20, 40), (25, 40), (20, 60)], start=1):
                Image.new("RGB", size, color="white").save(folder / f"im{index}.jpg")

            manifest = build_frame_manifest(root)
            summary = summarize_folder_geometry(manifest)

            self.assertEqual(len(summary), 1)
            row = summary.iloc[0]
            self.assertEqual(row["label"], "work")
            self.assertEqual(row["folder_rel"], "work/track_001")
            self.assertEqual(row["frame_count"], 3)
            self.assertEqual(row["unique_sizes"], 3)
            self.assertTrue(row["has_size_variability"])
            self.assertEqual(row["height_min"], 40)
            self.assertEqual(row["height_max"], 60)

    def test_summarize_sample_geometry_reports_eight_frame_sample_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "inaction" / "plain"
            sample.mkdir(parents=True)
            for index, size in enumerate(
                [(10, 20), (12, 20), (14, 22), (16, 22), (18, 24), (20, 24), (22, 26), (24, 26)],
                start=1,
            ):
                Image.new("RGB", size, color="white").save(sample / f"im{index}.jpg")

            frame_manifest = build_frame_manifest(root)
            sample_manifest = build_sample_manifest(root)
            summary = summarize_sample_geometry(frame_manifest, sample_manifest)

            self.assertEqual(len(summary), 1)
            row = summary.iloc[0]
            self.assertEqual(row["frame_count"], 8)
            self.assertEqual(row["unique_sizes"], 8)
            self.assertEqual(row["height_min"], 20)
            self.assertEqual(row["height_max"], 26)
            self.assertTrue(row["has_size_variability"])


if __name__ == "__main__":
    unittest.main()

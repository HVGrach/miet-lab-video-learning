import json
import unittest

import pandas as pd

from lab6_analysis.sampling import (
    build_training_sample_candidates,
    make_dilated_window,
    make_inaction_skip_shuffle_samples,
    make_strided_dilated_windows,
    sort_frame_paths_temporally,
)


class SamplingTests(unittest.TestCase):
    def test_sort_frame_paths_temporally_handles_lab6_filename_styles(self):
        paths = [
            "move/track/im_01-32-52_10.jpg",
            "move/track/im_01-32-52_2.jpg",
            "move/track/im_01-32-52_1.jpg",
            "move/track/im000003_00-25-31.jpg",
            "move/track/25_im000004_00-02-04.jpg",
        ]

        ordered = sort_frame_paths_temporally(paths)

        self.assertEqual(
            ordered,
            [
                "move/track/im_01-32-52_1.jpg",
                "move/track/im_01-32-52_2.jpg",
                "move/track/im000003_00-25-31.jpg",
                "move/track/25_im000004_00-02-04.jpg",
                "move/track/im_01-32-52_10.jpg",
            ],
        )

    def test_make_dilated_window_selects_lower_fps_frames(self):
        paths = [f"work/track/im{i:02d}.jpg" for i in range(20)]

        window = make_dilated_window(paths, start=1, dilation=2, frame_count=8)

        self.assertEqual(
            window,
            [
                "work/track/im01.jpg",
                "work/track/im03.jpg",
                "work/track/im05.jpg",
                "work/track/im07.jpg",
                "work/track/im09.jpg",
                "work/track/im11.jpg",
                "work/track/im13.jpg",
                "work/track/im15.jpg",
            ],
        )

    def test_make_strided_dilated_windows_records_strategy(self):
        paths = [f"move/track/im{i:02d}.jpg" for i in range(20)]

        windows = make_strided_dilated_windows(
            paths,
            label="move",
            folder_rel="move/track",
            dilation=2,
            stride=4,
            frame_count=8,
        )

        self.assertEqual([row["start_index"] for row in windows], [0, 4])
        self.assertEqual({row["sample_kind"] for row in windows}, {"stride_dilation"})
        self.assertEqual({row["dilation"] for row in windows}, {2})
        first_paths = json.loads(windows[0]["frame_paths"])
        self.assertEqual(first_paths[0], "move/track/im00.jpg")
        self.assertEqual(first_paths[-1], "move/track/im14.jpg")

    def test_inaction_skip_shuffle_is_deterministic_and_shuffled(self):
        paths = [f"inaction/track/im{i:02d}.jpg" for i in range(30)]

        first = make_inaction_skip_shuffle_samples(
            paths,
            folder_rel="inaction/track",
            samples_per_folder=2,
            seed=123,
            shuffle=True,
        )
        second = make_inaction_skip_shuffle_samples(
            paths,
            folder_rel="inaction/track",
            samples_per_folder=2,
            seed=123,
            shuffle=True,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        frame_paths = json.loads(first[0]["frame_paths"])
        self.assertEqual(len(frame_paths), 8)
        self.assertEqual(len(set(frame_paths)), 8)
        self.assertNotEqual(frame_paths, sort_frame_paths_temporally(frame_paths))
        self.assertEqual(first[0]["sample_kind"], "inaction_skip_shuffle")

    def test_build_training_sample_candidates_combines_explicit_and_generated(self):
        frame_manifest = pd.DataFrame(
            {
                "label": ["move"] * 20 + ["inaction"] * 20,
                "folder_rel": ["move/track"] * 20 + ["inaction/track"] * 20,
                "rel_path": [
                    *(f"move/track/im{i:02d}.jpg" for i in range(20)),
                    *(f"inaction/track/im{i:02d}.jpg" for i in range(20)),
                ],
            }
        )
        explicit = pd.DataFrame(
            {
                "sample_id": ["move/explicit::folder"],
                "label": ["move"],
                "folder_rel": ["move/explicit"],
                "group_key": ["folder"],
                "sample_kind": ["folder_8"],
                "frame_count": [8],
                "frame_paths": [
                    json.dumps([f"move/explicit/im{i}.jpg" for i in range(1, 9)])
                ],
            }
        )

        candidates = build_training_sample_candidates(
            frame_manifest,
            explicit_sample_manifest=explicit,
            move_work_dilations=(1, 2),
            move_work_stride=8,
            max_windows_per_folder_per_dilation=1,
            inaction_samples_per_folder=1,
            seed=7,
        )

        self.assertNotIn("path", candidates.columns)
        self.assertEqual(set(candidates["label"]), {"move", "inaction"})
        self.assertIn("explicit_folder_8", set(candidates["sample_kind"]))
        self.assertIn("stride_dilation", set(candidates["sample_kind"]))
        self.assertIn("inaction_skip_ordered", set(candidates["sample_kind"]))
        self.assertTrue((candidates["frame_count"] == 8).all())
        self.assertIn("source_track_id", candidates.columns)
        self.assertIn("frame_indices", candidates.columns)

    def test_build_training_sample_candidates_can_enable_inaction_shuffle_ablation(self):
        frame_manifest = pd.DataFrame(
            {
                "label": ["inaction"] * 20,
                "folder_rel": ["inaction/track"] * 20,
                "rel_path": [f"inaction/track/im{i:02d}.jpg" for i in range(20)],
            }
        )

        candidates = build_training_sample_candidates(
            frame_manifest,
            inaction_samples_per_folder=1,
            inaction_shuffle=True,
            seed=7,
        )

        self.assertEqual(set(candidates["sample_kind"]), {"inaction_skip_shuffle"})
        frame_paths = json.loads(candidates.iloc[0]["frame_paths"])
        self.assertNotEqual(frame_paths, sort_frame_paths_temporally(frame_paths))

    def test_build_training_sample_candidates_supports_strategy_pairs_and_label_caps(self):
        frame_manifest = pd.DataFrame(
            {
                "label": ["move"] * 50 + ["work"] * 50,
                "folder_rel": ["move/track"] * 50 + ["work/track"] * 50,
                "rel_path": [
                    *(f"move/track/im{i:02d}.jpg" for i in range(50)),
                    *(f"work/track/im{i:02d}.jpg" for i in range(50)),
                ],
            }
        )

        candidates = build_training_sample_candidates(
            frame_manifest,
            move_work_strategies=((1, 8), (2, 8), (4, 16)),
            max_windows_per_folder_per_dilation_by_label={"move": 2, "work": 1},
            inaction_samples_per_folder=0,
        )

        by_label = candidates.groupby("label").size().to_dict()
        self.assertEqual(by_label["move"], 6)
        self.assertEqual(by_label["work"], 3)
        strategies = {
            (int(row["dilation"]), int(row["stride"]))
            for _, row in candidates.iterrows()
        }
        self.assertEqual(strategies, {(1, 8), (2, 8), (4, 16)})

    def test_build_training_sample_candidates_api_defaults_match_project_default(self):
        frame_manifest = pd.DataFrame(
            {
                "label": ["inaction"] * 20 + ["move"] * 90 + ["work"] * 90,
                "folder_rel": ["inaction/track"] * 20
                + ["move/track"] * 90
                + ["work/track"] * 90,
                "rel_path": [
                    *(f"inaction/track/im{i:02d}.jpg" for i in range(20)),
                    *(f"move/track/im{i:02d}.jpg" for i in range(90)),
                    *(f"work/track/im{i:02d}.jpg" for i in range(90)),
                ],
            }
        )

        candidates = build_training_sample_candidates(frame_manifest)

        self.assertEqual(
            candidates[candidates["label"] == "inaction"]["sample_kind"].tolist(),
            ["inaction_skip_ordered"] * 24,
        )
        move = candidates[candidates["label"] == "move"]
        work = candidates[candidates["label"] == "work"]
        self.assertEqual(len(move), 8)
        self.assertEqual(len(work), 4)
        self.assertEqual(
            {
                (int(row["dilation"]), int(row["stride"]))
                for _, row in move.iterrows()
            },
            {(1, 8), (2, 8), (4, 16), (8, 32)},
        )
        self.assertNotIn("inaction_skip_shuffle", set(candidates["sample_kind"]))


if __name__ == "__main__":
    unittest.main()

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from lab6_analysis.inference import (
    collect_inference_images,
    load_lab6_predictor,
    predict_folder,
)
from lab6_analysis.training import LABEL_TO_INDEX, TinySequenceClassifier
from lab6_analysis.training import build_sequence_model


class InferenceTests(unittest.TestCase):
    def _write_checkpoint(self, path: Path, *, image_size=(32, 32)) -> None:
        model = TinySequenceClassifier(num_classes=3)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "label_to_index": LABEL_TO_INDEX,
                "preprocessing": {"image_size": list(image_size), "letterbox": True},
            },
            path,
        )

    def _write_mobilenet_checkpoint(self, path: Path, *, image_size=(32, 32)) -> None:
        model = build_sequence_model(
            "mobilenet_v3_small",
            num_classes=3,
            pretrained=False,
            freeze_backbone=True,
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "label_to_index": LABEL_TO_INDEX,
                "preprocessing": {"image_size": list(image_size), "letterbox": True},
                "model_config": {
                    "model_name": "mobilenet_v3_small",
                    "pretrained": False,
                    "freeze_backbone": True,
                },
            },
            path,
        )

    def _write_frames(self, folder: Path, count: int = 8) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            name = f"frame_{index + 1}.jpg"
            Image.new(
                "RGB",
                (12 + index, 18),
                color=(20 + index, 40, 80),
            ).save(folder / name)

    def test_collect_inference_images_requires_exactly_eight_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "sample"
            self._write_frames(folder, count=7)

            with self.assertRaises(ValueError):
                collect_inference_images(folder)

            Image.new("RGB", (10, 10), color=(1, 2, 3)).save(folder / "frame_8.jpg")
            (folder / "notes.txt").write_text("ignore me")

            paths = collect_inference_images(folder)

            self.assertEqual(len(paths), 8)
            self.assertEqual(paths[0].name, "frame_1.jpg")
            self.assertEqual(paths[-1].name, "frame_8.jpg")

    def test_predict_folder_returns_label_and_probabilities_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.pt"
            folder = root / "sample"
            self._write_checkpoint(checkpoint)
            self._write_frames(folder)

            predictor = load_lab6_predictor(checkpoint, device="cpu")
            result = predict_folder(folder, predictor)

            self.assertIn(result.label, {"inaction", "move", "work"})
            self.assertEqual(set(result.probabilities), {"inaction", "move", "work"})
            self.assertAlmostEqual(sum(result.probabilities.values()), 1.0, places=5)

    def test_predict_folder_supports_mobilenet_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.pt"
            folder = root / "sample"
            self._write_mobilenet_checkpoint(checkpoint)
            self._write_frames(folder)

            predictor = load_lab6_predictor(checkpoint, device="cpu")
            result = predict_folder(folder, predictor)

            self.assertIn(result.label, {"inaction", "move", "work"})

    def test_infer_cli_prints_only_one_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.pt"
            folder = root / "sample"
            self._write_checkpoint(checkpoint)
            self._write_frames(folder)

            completed = subprocess.run(
                [
                    sys.executable,
                    "infer.py",
                    str(folder),
                    "--checkpoint",
                    str(checkpoint),
                    "--device",
                    "cpu",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                text=True,
                capture_output=True,
            )

            output_lines = completed.stdout.strip().splitlines()
            self.assertEqual(len(output_lines), 1)
            self.assertIn(output_lines[0], {"inaction", "move", "work"})
            self.assertEqual(completed.stderr, "")


if __name__ == "__main__":
    unittest.main()

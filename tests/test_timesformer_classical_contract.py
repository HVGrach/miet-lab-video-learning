import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.config import load_labels
from src.data_io import collect_frame_paths
from src.model_timesformer import load_weights_into_model
from src.preprocess_frames import preprocess_frame_paths


class TimesformerClassicalContractTests(unittest.TestCase):
    def test_labels_json_matches_lab_classes(self):
        labels = load_labels("weights/labels.json")

        self.assertEqual(labels, {0: "inaction", 1: "move", 2: "work"})

    def test_frame_collection_is_lexicographic_and_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in ["b.jpg", "a.jpg", "c.png", "d.jpg", "e.jpg", "f.jpg", "g.jpg", "h.jpg"]:
                Image.new("RGB", (12, 10), color=(1, 2, 3)).save(folder / name)

            paths = collect_frame_paths(folder)

            self.assertEqual([path.name for path in paths], ["a.jpg", "b.jpg", "c.png", "d.jpg", "e.jpg", "f.jpg", "g.jpg", "h.jpg"])

            (folder / "notes.txt").write_text("not an image")
            with self.assertRaises(ValueError):
                collect_frame_paths(folder)

    def test_preprocess_224_no_pad_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for index in range(8):
                Image.new("RGB", (13 + index, 21), color=(index, 2, 3)).save(folder / f"{index}.jpg")

            tensor = preprocess_frame_paths(collect_frame_paths(folder))

            self.assertEqual(tuple(tensor.shape), (1, 8, 3, 224, 224))

    def test_predict_embeddings_cli_prints_single_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            emb = Path(tmp) / "object.npy"
            np.save(emb, np.zeros((768,), dtype=np.float32))

            completed = subprocess.run(
                [sys.executable, "-m", "src.predict_embeddings", "--input", str(emb)],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn(completed.stdout.strip(), {"inaction", "move", "work"})

    def test_predict_frames_cli_matches_checkpoint_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for index in range(8):
                Image.new("RGB", (16, 16), color=(index, 2, 3)).save(folder / f"{index}.jpg")

            completed = subprocess.run(
                [sys.executable, "-m", "src.predict_frames", "--input", str(folder)],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
            )

            if Path("weights/timesformer_checkpoint.pt").is_file():
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn(completed.stdout.strip(), {"inaction", "move", "work"})
            else:
                self.assertEqual(completed.returncode, 2)
                self.assertEqual(completed.stdout, "")
                self.assertIn("timesformer_checkpoint.pt", completed.stderr)

    def test_checkpoint_loader_rejects_incompatible_state_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "bad.pt"
            torch.save({"model": {"totally_wrong.weight": torch.zeros(1)}}, checkpoint)
            model = torch.nn.Linear(4, 3)

            with self.assertRaises(RuntimeError):
                load_weights_into_model(model, checkpoint)


if __name__ == "__main__":
    unittest.main()

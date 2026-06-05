import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from lab6_analysis.training import (
    LABEL_TO_INDEX,
    Lab6SequenceDataset,
    TinySequenceClassifier,
    build_sequence_model,
    compute_classification_metrics,
    compute_class_weights,
    evaluate_model,
    train_one_epoch,
)


class TrainingTests(unittest.TestCase):
    def test_sequence_dataset_returns_tensor_label_and_sample_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "move" / "sample"
            folder.mkdir(parents=True)
            frame_paths = []
            for index in range(8):
                rel_path = f"move/sample/im{index + 1}.jpg"
                Image.new("RGB", (12, 8), color=(index, 30, 60)).save(root / rel_path)
                frame_paths.append(rel_path)
            manifest = pd.DataFrame(
                {
                    "sample_id": ["move/sample::folder"],
                    "label": ["move"],
                    "frame_paths": [json.dumps(frame_paths)],
                    "split": ["train"],
                    "folder_rel": ["move/sample"],
                    "hash_guard_group_id": ["move/sample"],
                }
            )

            dataset = Lab6SequenceDataset(root, manifest, output_size=(32, 32))
            item = dataset[0]

            self.assertEqual(item["frames"].shape, (8, 3, 32, 32))
            self.assertEqual(item["label"].item(), LABEL_TO_INDEX["move"])
            self.assertEqual(item["sample_id"], "move/sample::folder")
            self.assertNotIn("folder_rel", item)
            self.assertNotIn("hash_guard_group_id", item)

    def test_sequence_dataset_rejects_non_eight_frame_sample(self):
        manifest = pd.DataFrame(
            {
                "sample_id": ["bad"],
                "label": ["work"],
                "frame_paths": [json.dumps(["work/a/im1.jpg"])],
            }
        )

        with self.assertRaises(ValueError):
            Lab6SequenceDataset(Path("."), manifest)

    def test_tiny_sequence_classifier_forward_and_train_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for label in ("inaction", "move", "work"):
                folder = root / label / "sample"
                folder.mkdir(parents=True)
                frame_paths = []
                for index in range(8):
                    rel_path = f"{label}/sample/im{index + 1}.jpg"
                    Image.new("RGB", (10, 10), color=(40, 80, 120)).save(root / rel_path)
                    frame_paths.append(rel_path)
                rows.append(
                    {
                        "sample_id": f"{label}/sample::folder",
                        "label": label,
                        "frame_paths": json.dumps(frame_paths),
                    }
                )
            dataset = Lab6SequenceDataset(root, pd.DataFrame(rows), output_size=(32, 32))
            loader = DataLoader(dataset, batch_size=3)
            model = TinySequenceClassifier(num_classes=3)
            batch = next(iter(loader))

            logits = model(batch["frames"])
            self.assertEqual(logits.shape, (3, 3))

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            loss = train_one_epoch(
                model,
                loader,
                optimizer,
                torch.device("cpu"),
                max_batches=1,
            )
            self.assertTrue(torch.isfinite(torch.tensor(loss["loss"])))

    def test_build_sequence_model_supports_frozen_mobilenet(self):
        model = build_sequence_model(
            "mobilenet_v3_small",
            num_classes=3,
            pretrained=False,
            freeze_backbone=True,
        )

        logits = model(torch.zeros(2, 8, 3, 64, 64))

        self.assertEqual(logits.shape, (2, 3))
        frozen_params = [p.requires_grad for p in model.frame_encoder.parameters()]
        self.assertTrue(frozen_params)
        self.assertFalse(any(frozen_params))

    def test_compute_classification_metrics_reports_macro_f1(self):
        metrics = compute_classification_metrics(
            y_true=[0, 1, 2, 2],
            y_pred=[0, 1, 1, 2],
        )

        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertIn("macro_f1", metrics)
        self.assertEqual(metrics["confusion_matrix"].shape, (3, 3))
        self.assertEqual(set(metrics["per_class"].keys()), {"inaction", "move", "work"})

    def test_compute_class_weights_is_inverse_frequency(self):
        weights = compute_class_weights([0, 0, 1, 2])

        self.assertEqual(weights.shape, (3,))
        self.assertGreater(weights[1], weights[0])
        self.assertGreater(weights[2], weights[0])

    def test_evaluate_model_returns_predictions_and_metrics(self):
        frames = torch.zeros(4, 8, 3, 32, 32)
        labels = torch.tensor([0, 1, 2, 2])
        loader = DataLoader(
            [{"frames": frames[i], "label": labels[i], "sample_id": f"s{i}"} for i in range(4)],
            batch_size=2,
        )
        model = TinySequenceClassifier(num_classes=3)

        result = evaluate_model(model, loader, torch.device("cpu"))

        self.assertEqual(len(result["predictions"]), 4)
        self.assertIn("macro_f1", result["metrics"])


if __name__ == "__main__":
    unittest.main()

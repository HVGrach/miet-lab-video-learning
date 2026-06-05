import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from lab6_analysis.experiment import BaselineExperimentConfig, run_baseline_experiment


class ExperimentRunnerTests(unittest.TestCase):
    def _write_sample(self, root: Path, label: str, name: str) -> list[str]:
        folder = root / label / name
        folder.mkdir(parents=True)
        paths = []
        for index in range(8):
            rel_path = f"{label}/{name}/im{index + 1}.jpg"
            Image.new(
                "RGB",
                (14 + index, 16),
                color=(30 + index, 60, 90),
            ).save(root / rel_path)
            paths.append(rel_path)
        return paths

    def test_run_baseline_experiment_writes_reproducible_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            rows = []
            for split in ("train", "val"):
                for label in ("inaction", "move", "work"):
                    sample_id = f"{split}-{label}"
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "label": label,
                            "frame_paths": json.dumps(
                                self._write_sample(dataset_root, label, sample_id)
                            ),
                            "split": split,
                        }
                    )
            manifest_path = root / "split.csv"
            pd.DataFrame(rows).to_csv(manifest_path, index=False)

            config = BaselineExperimentConfig(
                run_id="unit_tiny",
                dataset_root=dataset_root,
                split_manifest=manifest_path,
                output_root=root / "runs",
                image_size=(32, 32),
                batch_size=3,
                epochs=1,
                max_train_batches=1,
                max_val_batches=1,
                device="cpu",
                augment=False,
            )

            summary = run_baseline_experiment(config)

            run_dir = root / "runs" / "unit_tiny"
            self.assertEqual(summary["run_id"], "unit_tiny")
            self.assertEqual(summary["model_name"], "tiny_sequence")
            self.assertEqual(summary["train_samples"], 3)
            self.assertEqual(summary["val_samples"], 3)
            for name in [
                "config.json",
                "environment.json",
                "metrics.json",
                "epoch_metrics.csv",
                "per_class_metrics.csv",
                "confusion_matrix.csv",
                "val_predictions.csv",
                "model_best.pt",
            ]:
                self.assertTrue((run_dir / name).exists(), name)

            epochs = pd.read_csv(run_dir / "epoch_metrics.csv")
            predictions = pd.read_csv(run_dir / "val_predictions.csv")
            self.assertEqual(len(epochs), 1)
            self.assertEqual(len(predictions), 3)

            import torch

            checkpoint = torch.load(run_dir / "model_best.pt", map_location="cpu")
            self.assertEqual(checkpoint["model_config"]["model_name"], "tiny_sequence")


if __name__ == "__main__":
    unittest.main()

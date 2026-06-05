import json
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
RUN_DIR = OUTPUTS / "baseline_runs" / "tiny_sequence_smoke"


class BaselineOutputsContractTests(unittest.TestCase):
    @unittest.skipUnless(RUN_DIR.exists(), "baseline run directory is not published")
    def test_baseline_run_artifacts_exist(self):
        for rel_path in [
            "config.json",
            "environment.json",
            "metrics.json",
            "per_class_metrics.csv",
            "confusion_matrix.csv",
            "val_predictions.csv",
            "model_best.pt",
        ]:
            path = RUN_DIR / rel_path
            self.assertTrue(path.exists(), rel_path)
            self.assertGreater(path.stat().st_size, 0, rel_path)

    @unittest.skipUnless(RUN_DIR.exists(), "baseline run directory is not published")
    def test_baseline_metrics_use_hash_guarded_split(self):
        metrics_path = RUN_DIR / "metrics.json"
        config_path = RUN_DIR / "config.json"
        with metrics_path.open() as handle:
            metrics = json.load(handle)
        with config_path.open() as handle:
            config = json.load(handle)

        self.assertTrue(metrics["is_smoke"])
        self.assertEqual(
            metrics["split_manifest"],
            "outputs/manifests/train_val_split_hash_guarded.csv",
        )
        self.assertEqual(
            config["split_manifest"],
            "outputs/manifests/train_val_split_hash_guarded.csv",
        )
        self.assertEqual(metrics["train_samples"], 1222)
        self.assertEqual(metrics["val_samples"], 324)
        self.assertIn("split_manifest_sha256", metrics)
        self.assertEqual(len(metrics["split_manifest_sha256"]), 64)
        self.assertEqual(metrics["hash_leakage_rows"], 0)
        self.assertEqual(metrics["virtual_leakage_rows"], 0)
        self.assertGreaterEqual(metrics["val_accuracy"], 0.0)
        self.assertLessEqual(metrics["val_accuracy"], 1.0)
        self.assertGreaterEqual(metrics["val_macro_f1"], 0.0)
        self.assertLessEqual(metrics["val_macro_f1"], 1.0)

    @unittest.skipUnless(RUN_DIR.exists(), "baseline run directory is not published")
    def test_baseline_prediction_and_metric_shapes(self):
        predictions = pd.read_csv(RUN_DIR / "val_predictions.csv")
        per_class = pd.read_csv(RUN_DIR / "per_class_metrics.csv")
        confusion = pd.read_csv(RUN_DIR / "confusion_matrix.csv")

        self.assertEqual(len(predictions), 324)
        self.assertEqual(set(per_class["label"]), {"inaction", "move", "work"})
        self.assertEqual(confusion.shape, (3, 4))
        self.assertEqual(
            set(predictions["pred_label"]).issubset({"inaction", "move", "work"}),
            True,
        )


if __name__ == "__main__":
    unittest.main()

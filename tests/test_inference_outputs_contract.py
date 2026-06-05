import unittest
from pathlib import Path

import torch

from lab6_analysis.inference import DEFAULT_CHECKPOINT


class InferenceOutputsContractTests(unittest.TestCase):
    def test_default_checkpoint_is_trained_baseline_artifact(self):
        self.assertEqual(DEFAULT_CHECKPOINT, Path("outputs/models/lab6_action_classifier.pt"))
        self.assertTrue(DEFAULT_CHECKPOINT.exists())

        checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location="cpu")
        self.assertEqual(checkpoint["preprocessing"]["image_size"], [160, 160])
        self.assertEqual(
            checkpoint["summary"]["run_id"],
            "mobilenet_v3_small_160_frozen_e30_balanced_minclass",
        )
        self.assertEqual(checkpoint["model_config"]["model_name"], "mobilenet_v3_small")
        self.assertGreaterEqual(checkpoint["summary"]["val_macro_f1"], 0.53)
        self.assertGreaterEqual(checkpoint["summary"]["val_work_f1"], 0.55)

    def test_inference_docker_acceptance_report_exists(self):
        report = Path("outputs/reports/inference_docker_acceptance.md")
        self.assertTrue(report.exists())
        text = report.read_text()

        self.assertIn("lab6_action_classifier.pt", text)
        self.assertIn("lab6-video-infer:latest", text)
        self.assertIn("Docker run", text)
        self.assertIn("3.11", text)


if __name__ == "__main__":
    unittest.main()

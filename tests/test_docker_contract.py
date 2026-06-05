import unittest
from pathlib import Path


class DockerContractTests(unittest.TestCase):
    def test_dockerfile_packages_cpu_safe_inference_entrypoint(self):
        dockerfile = Path("Dockerfile")
        self.assertTrue(dockerfile.exists())
        text = dockerfile.read_text()

        self.assertIn("requirements-inference.txt", text)
        self.assertIn("outputs/models/lab6_action_classifier.pt", text)
        self.assertIn("weights/timesformer_checkpoint.pt", text)
        self.assertIn("weights/timesformer_hf", text)
        self.assertIn('ENTRYPOINT ["python", "-m", "src.predict_frames"]', text)
        self.assertIn('CMD ["--input", "/app/input"]', text)
        self.assertNotIn("lab6/", text)

    def test_dockerignore_excludes_training_dataset_and_notebook_outputs(self):
        dockerignore = Path(".dockerignore")
        self.assertTrue(dockerignore.exists())
        text = dockerignore.read_text()

        self.assertIn("lab6", text)
        self.assertIn("notebooks", text)
        self.assertIn("outputs/*", text)
        self.assertIn("!outputs/models/lab6_action_classifier.pt", text)

    def test_inference_requirements_are_pinned_and_include_import_dependencies(self):
        requirements = Path("requirements-inference.txt")
        self.assertTrue(requirements.exists())
        lines = {
            line.strip()
            for line in requirements.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }

        self.assertIn("numpy==2.4.3", lines)
        self.assertIn("Pillow==12.1.1", lines)
        self.assertIn("pandas==3.0.1", lines)
        self.assertIn("torch==2.11.0", lines)
        self.assertIn("torchvision==0.26.0", lines)


if __name__ == "__main__":
    unittest.main()

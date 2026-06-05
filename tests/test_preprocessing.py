import unittest

from PIL import Image, ImageChops

from lab6_analysis.preprocessing import (
    SequenceAugmentationConfig,
    apply_sequence_preprocessing,
    build_sequence_augmentation_params,
    letterbox_resize,
)


class PreprocessingTests(unittest.TestCase):
    def test_letterbox_resize_preserves_aspect_ratio_and_centers_image(self):
        image = Image.new("RGB", (20, 10), color="white")

        resized, meta = letterbox_resize(image, size=(32, 32), fill=(0, 0, 0))

        self.assertEqual(resized.size, (32, 32))
        self.assertEqual(meta["resized_size"], (32, 16))
        self.assertEqual(meta["padding"], (0, 8, 0, 8))
        self.assertEqual(resized.getpixel((16, 7)), (0, 0, 0))
        self.assertEqual(resized.getpixel((16, 8)), (255, 255, 255))

    def test_sequence_augmentation_params_are_deterministic(self):
        config = SequenceAugmentationConfig(
            brightness=(0.8, 1.2),
            contrast=(0.9, 1.1),
            color=(0.7, 1.3),
            jpeg_quality=(50, 80),
            downscale=(0.5, 0.75),
        )

        first = build_sequence_augmentation_params(config, seed=42)
        second = build_sequence_augmentation_params(config, seed=42)

        self.assertEqual(first, second)
        self.assertGreaterEqual(first.jpeg_quality, 50)
        self.assertLessEqual(first.jpeg_quality, 80)
        self.assertGreaterEqual(first.downscale, 0.5)
        self.assertLessEqual(first.downscale, 0.75)

    def test_apply_sequence_preprocessing_uses_same_params_for_all_frames(self):
        images = [
            Image.new("RGB", (24, 16), color=(120, 80, 40)),
            Image.new("RGB", (24, 16), color=(120, 80, 40)),
        ]
        config = SequenceAugmentationConfig(
            brightness=(0.9, 0.9),
            contrast=(1.1, 1.1),
            color=(0.8, 0.8),
            jpeg_quality=(70, 70),
            downscale=(0.5, 0.5),
        )

        processed, params = apply_sequence_preprocessing(
            images,
            output_size=(32, 32),
            config=config,
            seed=3,
        )

        self.assertEqual(len(processed), 2)
        self.assertEqual(processed[0].size, (32, 32))
        self.assertEqual(processed[1].size, (32, 32))
        self.assertIsNone(ImageChops.difference(processed[0], processed[1]).getbbox())
        self.assertEqual(params.jpeg_quality, 70)
        self.assertEqual(params.downscale, 0.5)


if __name__ == "__main__":
    unittest.main()

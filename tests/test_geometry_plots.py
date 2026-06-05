import tempfile
import unittest
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from PIL import Image

from lab6_analysis.geometry_plots import (
    create_outlier_contact_sheet,
    plot_aspect_ratio_with_height_inset,
    plot_height_distribution_by_class,
    plot_width_height_scatter,
)


class GeometryPlotTests(unittest.TestCase):
    def test_plot_aspect_ratio_with_height_inset_returns_figure_and_saves_png(self):
        manifest = pd.DataFrame(
            {
                "label": ["inaction", "inaction", "move", "work"],
                "width": [50, 60, 100, 120],
                "height": [100, 100, 200, 160],
                "aspect_ratio": [0.5, 0.6, 0.5, 0.75],
                "area": [5000, 6000, 20000, 19200],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "geometry.png"

            fig = plot_aspect_ratio_with_height_inset(manifest, output_path=output_path)

            self.assertIsInstance(fig, Figure)
            self.assertGreaterEqual(len(fig.axes), 2)
            self.assertIn("Aspect ratio", fig.axes[0].get_xlabel())
            self.assertIn("height", fig.axes[1].get_xlabel().lower())
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_extra_geometry_plots_return_figures(self):
        manifest = pd.DataFrame(
            {
                "label": ["inaction", "move", "work"],
                "width": [50, 100, 120],
                "height": [100, 200, 160],
                "aspect_ratio": [0.5, 0.5, 0.75],
                "area": [5000, 20000, 19200],
            }
        )

        height_fig = plot_height_distribution_by_class(manifest)
        scatter_fig = plot_width_height_scatter(manifest)

        self.assertIsInstance(height_fig, Figure)
        self.assertIsInstance(scatter_fig, Figure)

    def test_create_outlier_contact_sheet_saves_image_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "lab6" / "move" / "sample"
            image_dir.mkdir(parents=True)
            for name, size in {"small.jpg": (20, 20), "wide.jpg": (80, 20)}.items():
                Image.new("RGB", size, color="white").save(image_dir / name)
            outliers = pd.DataFrame(
                {
                    "rel_path": ["move/sample/small.jpg", "move/sample/wide.jpg"],
                    "label": ["move", "move"],
                    "width": [20, 80],
                    "height": [20, 20],
                    "aspect_ratio": [1.0, 4.0],
                    "outlier_reason": ["small_width,small_height", "high_aspect"],
                }
            )
            output_path = root / "outliers.png"

            create_outlier_contact_sheet(root / "lab6", outliers, output_path, max_images=2)

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

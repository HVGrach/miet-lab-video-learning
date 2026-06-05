from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw

CLASS_COLORS = {
    "inaction": "#4C78A8",
    "move": "#F58518",
    "work": "#54A24B",
}


def plot_aspect_ratio_with_height_inset(
    frame_manifest: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    bins: int = 48,
):
    """Plot aspect-ratio distribution with an inset height distribution."""
    _require_columns(frame_manifest, {"label", "aspect_ratio", "height"})
    if frame_manifest.empty:
        raise ValueError("frame_manifest is empty")

    fig, ax = plt.subplots(figsize=(12, 7))
    labels = sorted(frame_manifest["label"].unique())

    for label in labels:
        subset = frame_manifest[frame_manifest["label"] == label]
        color = CLASS_COLORS.get(label)
        ax.hist(
            subset["aspect_ratio"],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=color,
            label=f"{label} (n={len(subset)})",
        )

    ax.set_title("Aspect ratio distribution by class with height inset")
    ax.set_xlabel("Aspect ratio = width / height")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    inset = fig.add_axes([0.55, 0.44, 0.32, 0.30])
    for label in labels:
        subset = frame_manifest[frame_manifest["label"] == label]
        color = CLASS_COLORS.get(label)
        inset.hist(
            subset["height"],
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.18,
            color=color,
        )
        inset.hist(
            subset["height"],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.3,
            color=color,
            label=label,
        )

    inset.set_title("Height distribution", fontsize=10)
    inset.set_xlabel("height px", fontsize=9)
    inset.set_ylabel("density", fontsize=9)
    inset.grid(alpha=0.20)
    inset.tick_params(axis="both", labelsize=8)

    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.10, top=0.92)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def plot_height_distribution_by_class(
    frame_manifest: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    bins: int = 64,
):
    """Plot frame-height distributions by class."""
    _require_columns(frame_manifest, {"label", "height"})
    if frame_manifest.empty:
        raise ValueError("frame_manifest is empty")

    fig, ax = plt.subplots(figsize=(12, 6))
    for label in sorted(frame_manifest["label"].unique()):
        subset = frame_manifest[frame_manifest["label"] == label]
        ax.hist(
            subset["height"],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=CLASS_COLORS.get(label),
            label=f"{label} (n={len(subset)})",
        )
        ax.axvline(
            subset["height"].median(),
            color=CLASS_COLORS.get(label),
            linestyle="--",
            alpha=0.75,
        )

    ax.set_title("Frame height distribution by class")
    ax.set_xlabel("Height, px")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return _save_and_return(fig, output_path)


def plot_width_height_scatter(
    frame_manifest: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    alpha: float = 0.25,
):
    """Plot frame width versus height, colored by class."""
    _require_columns(frame_manifest, {"label", "width", "height"})
    if frame_manifest.empty:
        raise ValueError("frame_manifest is empty")

    fig, ax = plt.subplots(figsize=(9, 8))
    for label in sorted(frame_manifest["label"].unique()):
        subset = frame_manifest[frame_manifest["label"] == label]
        ax.scatter(
            subset["width"],
            subset["height"],
            s=9,
            alpha=alpha,
            color=CLASS_COLORS.get(label),
            label=f"{label} (n={len(subset)})",
            edgecolors="none",
        )

    ax.set_title("Width vs height by class")
    ax.set_xlabel("Width, px")
    ax.set_ylabel("Height, px")
    ax.grid(alpha=0.25)
    ax.legend(markerscale=2)
    fig.tight_layout()
    return _save_and_return(fig, output_path)


def create_outlier_contact_sheet(
    dataset_root: str | Path,
    outliers: pd.DataFrame,
    output_path: str | Path,
    *,
    max_images: int = 24,
    thumb_size: tuple[int, int] = (128, 128),
    columns: int = 6,
) -> Path:
    """Save a thumbnail grid for manual inspection of geometry outliers."""
    _require_columns(
        outliers,
        {"rel_path", "label", "width", "height", "aspect_ratio", "outlier_reason"},
    )
    if outliers.empty:
        raise ValueError("outliers is empty")

    root = Path(dataset_root)
    selected = outliers.head(max_images).reset_index(drop=True)
    rows = (len(selected) + columns - 1) // columns
    cell_width = thumb_size[0]
    cell_height = thumb_size[1] + 58
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, row in selected.iterrows():
        image_path = root / str(row["rel_path"])
        with Image.open(image_path) as image:
            thumbnail = image.convert("RGB")
            thumbnail.thumbnail(thumb_size)
        col = idx % columns
        row_idx = idx // columns
        x = col * cell_width + (cell_width - thumbnail.width) // 2
        y = row_idx * cell_height + 4
        sheet.paste(thumbnail, (x, y))
        text_y = row_idx * cell_height + thumb_size[1] + 8
        label = (
            f"{row['label']} {int(row['width'])}x{int(row['height'])}\n"
            f"ar={float(row['aspect_ratio']):.2f}\n"
            f"{row['outlier_reason']}"
        )
        draw.multiline_text(
            (col * cell_width + 4, text_y),
            label[:90],
            fill="black",
            spacing=1,
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"frame_manifest is missing columns: {sorted(missing)}")


def _save_and_return(fig, output_path: str | Path | None):
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageEnhance


@dataclass(frozen=True)
class SequenceAugmentationConfig:
    brightness: tuple[float, float] = (1.0, 1.0)
    contrast: tuple[float, float] = (1.0, 1.0)
    color: tuple[float, float] = (1.0, 1.0)
    jpeg_quality: tuple[int, int] | None = None
    downscale: tuple[float, float] | None = None


@dataclass(frozen=True)
class SequenceAugmentationParams:
    brightness: float
    contrast: float
    color: float
    jpeg_quality: int | None
    downscale: float | None


def letterbox_resize(
    image: Image.Image,
    *,
    size: tuple[int, int] = (224, 224),
    fill: tuple[int, int, int] = (0, 0, 0),
    resample=Image.Resampling.BICUBIC,
) -> tuple[Image.Image, dict[str, object]]:
    """Resize preserving aspect ratio and pad to the requested size."""
    source = image.convert("RGB")
    target_width, target_height = size
    width, height = source.size
    if width <= 0 or height <= 0:
        raise ValueError("image has invalid size")

    scale = min(target_width / width, target_height / height)
    resized_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    resized = source.resize(resized_size, resample=resample)
    canvas = Image.new("RGB", size, fill)
    left = (target_width - resized_size[0]) // 2
    top = (target_height - resized_size[1]) // 2
    canvas.paste(resized, (left, top))
    right = target_width - resized_size[0] - left
    bottom = target_height - resized_size[1] - top
    return canvas, {
        "source_size": (width, height),
        "resized_size": resized_size,
        "padding": (left, top, right, bottom),
        "scale": scale,
    }


def build_sequence_augmentation_params(
    config: SequenceAugmentationConfig,
    *,
    seed: int | None = None,
) -> SequenceAugmentationParams:
    """Sample one augmentation parameter set for the whole 8-frame object."""
    rng = random.Random(seed)
    jpeg_quality = None
    if config.jpeg_quality is not None:
        low, high = config.jpeg_quality
        jpeg_quality = rng.randint(low, high)

    downscale = None
    if config.downscale is not None:
        downscale = _uniform(rng, config.downscale)

    return SequenceAugmentationParams(
        brightness=_uniform(rng, config.brightness),
        contrast=_uniform(rng, config.contrast),
        color=_uniform(rng, config.color),
        jpeg_quality=jpeg_quality,
        downscale=downscale,
    )


def apply_sequence_preprocessing(
    images: list[Image.Image],
    *,
    output_size: tuple[int, int] = (224, 224),
    config: SequenceAugmentationConfig | None = None,
    seed: int | None = None,
    params: SequenceAugmentationParams | None = None,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> tuple[list[Image.Image], SequenceAugmentationParams]:
    """Apply the same preprocessing/augmentation parameters to all frames."""
    if not images:
        raise ValueError("images must be non-empty")
    if config is None:
        config = SequenceAugmentationConfig()
    if params is None:
        params = build_sequence_augmentation_params(config, seed=seed)

    processed: list[Image.Image] = []
    for image in images:
        frame, _ = letterbox_resize(image, size=output_size, fill=fill)
        frame = _apply_color_jitter(frame, params)
        frame = _apply_downscale(frame, params.downscale)
        frame = _apply_jpeg_distortion(frame, params.jpeg_quality)
        processed.append(frame)
    return processed, params


def create_preprocessing_contact_sheet(
    dataset_root: str | Path,
    frame_paths: list[str],
    output_path: str | Path,
    *,
    output_size: tuple[int, int] = (128, 128),
    config: SequenceAugmentationConfig | None = None,
    seed: int = 20260602,
) -> Path:
    """Save a before/after contact sheet for one 8-frame sequence."""
    if len(frame_paths) != 8:
        raise ValueError("frame_paths must contain exactly 8 frames")

    root = Path(dataset_root)
    images = [Image.open(root / rel_path).convert("RGB") for rel_path in frame_paths]
    processed, params = apply_sequence_preprocessing(
        images,
        output_size=output_size,
        config=config,
        seed=seed,
    )

    cell_w, cell_h = output_size
    label_h = 42
    sheet = Image.new("RGB", (8 * cell_w, 2 * (cell_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (rel_path, before, after) in enumerate(zip(frame_paths, images, processed)):
        before_thumb, _ = letterbox_resize(before, size=output_size, fill=(0, 0, 0))
        x = index * cell_w
        sheet.paste(before_thumb, (x, 0))
        sheet.paste(after, (x, cell_h + label_h))
        draw.text((x + 4, cell_h + 4), f"before {index + 1}", fill="black")
        draw.text((x + 4, 2 * cell_h + label_h + 4), f"after {index + 1}", fill="black")

    draw.text(
        (4, cell_h + 20),
        (
            f"brightness={params.brightness:.2f}, contrast={params.contrast:.2f}, "
            f"color={params.color:.2f}, jpeg={params.jpeg_quality}, "
            f"downscale={params.downscale}"
        ),
        fill="black",
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def _uniform(rng: random.Random, value_range: tuple[float, float]) -> float:
    low, high = value_range
    if low > high:
        raise ValueError("augmentation range min must be <= max")
    if low == high:
        return low
    return rng.uniform(low, high)


def _apply_color_jitter(
    image: Image.Image, params: SequenceAugmentationParams
) -> Image.Image:
    frame = ImageEnhance.Brightness(image).enhance(params.brightness)
    frame = ImageEnhance.Contrast(frame).enhance(params.contrast)
    frame = ImageEnhance.Color(frame).enhance(params.color)
    return frame


def _apply_downscale(image: Image.Image, downscale: float | None) -> Image.Image:
    if downscale is None or downscale >= 1.0:
        return image
    if downscale <= 0:
        raise ValueError("downscale must be positive")

    width, height = image.size
    small_size = (
        max(1, round(width * downscale)),
        max(1, round(height * downscale)),
    )
    small = image.resize(small_size, resample=Image.Resampling.BILINEAR)
    return small.resize(image.size, resample=Image.Resampling.BILINEAR)


def _apply_jpeg_distortion(
    image: Image.Image, jpeg_quality: int | None
) -> Image.Image:
    if jpeg_quality is None:
        return image
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [1, 100]")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    buffer.seek(0)
    with Image.open(buffer) as distorted:
        return distorted.convert("RGB")

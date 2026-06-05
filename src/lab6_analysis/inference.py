from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from PIL import Image

from lab6_analysis.preprocessing import apply_sequence_preprocessing
from lab6_analysis.sampling import EXPECTED_FRAME_COUNT, sort_frame_paths_temporally
from lab6_analysis.training import LABELS, build_sequence_model


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CHECKPOINT = Path("outputs/models/lab6_action_classifier.pt")


@dataclass(frozen=True)
class Lab6Predictor:
    model: torch.nn.Module
    index_to_label: dict[int, str]
    image_size: tuple[int, int]
    device: torch.device


@dataclass(frozen=True)
class InferenceResult:
    label: str
    probabilities: dict[str, float]


def collect_inference_images(folder: str | Path) -> list[Path]:
    """Return exactly eight image files from a folder in stable temporal order."""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"inference input must be a folder: {folder_path}")

    image_paths = [
        path
        for path in folder_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if len(image_paths) != EXPECTED_FRAME_COUNT:
        raise ValueError(
            f"inference folder must contain exactly {EXPECTED_FRAME_COUNT} images; "
            f"found {len(image_paths)}"
        )

    sorted_paths = sort_frame_paths_temporally(str(path) for path in image_paths)
    return [Path(path) for path in sorted_paths]


def load_lab6_predictor(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    *,
    device: str | torch.device = "cpu",
) -> Lab6Predictor:
    """Load a lab6 checkpoint for deterministic inference."""
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint_file}")

    resolved_device = torch.device(device)
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    label_to_index = _validate_label_mapping(checkpoint.get("label_to_index", {}))
    index_to_label = {index: label for label, index in label_to_index.items()}
    image_size = _read_image_size(checkpoint.get("preprocessing", {}))

    model_config = _read_model_config(checkpoint)
    model = build_sequence_model(
        str(model_config["model_name"]),
        num_classes=len(label_to_index),
        pretrained=False,
        freeze_backbone=bool(model_config["freeze_backbone"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()
    return Lab6Predictor(
        model=model,
        index_to_label=index_to_label,
        image_size=image_size,
        device=resolved_device,
    )


@torch.no_grad()
def predict_folder(folder: str | Path, predictor: Lab6Predictor) -> InferenceResult:
    """Predict one action class for a folder with exactly eight images."""
    image_paths = collect_inference_images(folder)
    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))

    processed, _ = apply_sequence_preprocessing(
        images,
        output_size=predictor.image_size,
    )
    frames = torch.stack([_image_to_tensor(image) for image in processed])
    logits = predictor.model(frames.unsqueeze(0).to(predictor.device))
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().tolist()
    pred_index = int(np.argmax(probs))
    probabilities = {
        predictor.index_to_label[index]: float(prob)
        for index, prob in enumerate(probs)
    }
    return InferenceResult(
        label=predictor.index_to_label[pred_index],
        probabilities=probabilities,
    )


def _validate_label_mapping(raw_mapping: Mapping[str, object]) -> dict[str, int]:
    label_to_index = {str(label): int(index) for label, index in raw_mapping.items()}
    if set(label_to_index) != set(LABELS):
        raise ValueError(f"checkpoint label mapping must contain {LABELS}")
    if sorted(label_to_index.values()) != list(range(len(LABELS))):
        raise ValueError("checkpoint label indexes must be contiguous from zero")
    return label_to_index


def _read_image_size(preprocessing: Mapping[str, object]) -> tuple[int, int]:
    image_size = preprocessing.get("image_size", [64, 64])
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError("checkpoint preprocessing.image_size must contain two values")
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("checkpoint preprocessing.image_size must be positive")
    return (width, height)


def _read_model_config(checkpoint: Mapping[str, object]) -> dict[str, object]:
    raw_config = checkpoint.get("model_config")
    if raw_config is None:
        return {
            "model_name": "tiny_sequence",
            "freeze_backbone": True,
        }
    if not isinstance(raw_config, Mapping):
        raise ValueError("checkpoint model_config must be a mapping")
    model_name = str(raw_config.get("model_name", "tiny_sequence"))
    return {
        "model_name": model_name,
        "freeze_backbone": bool(raw_config.get("freeze_backbone", True)),
    }


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)

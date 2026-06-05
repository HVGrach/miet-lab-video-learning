from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from .config import DEPLOYMENT


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


def preprocess_frame_paths(paths: Sequence[str | Path]) -> torch.Tensor:
    frames = []
    for path in paths:
        try:
            with Image.open(path) as image:
                frames.append(preprocess_image(image))
        except Exception as error:  # noqa: BLE001
            raise ValueError(f"could not read image {path}: {error}") from error
    return torch.stack(frames, dim=0).unsqueeze(0)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    # 224_no_pad from the notebook: bicubic resize directly to a square.
    image = image.convert("RGB").resize(
        (DEPLOYMENT.image_size, DEPLOYMENT.image_size),
        Image.Resampling.BICUBIC,
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    tensor = torch.from_numpy(array)
    return (tensor - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]

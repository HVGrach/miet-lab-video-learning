from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_DIR = Path(os.environ.get("LAB6_WEIGHTS_DIR", PROJECT_ROOT / "weights"))


@dataclass(frozen=True)
class DeploymentConfig:
    track_name: str = "track35_timesformer"
    timesformer_model_name: str = "facebook/timesformer-base-finetuned-k400"
    preprocess_name: str = "224_no_pad"
    video_peft_type: str = "last_block_only"
    video_training_scope: str = "last_n_blocks"
    video_last_n_blocks: int = 1
    video_adapter_dim: int = 64
    video_embedding_type: str = "default"
    video_frame_pool_type: str = "default"
    batch_size: int = 4
    grad_accum_steps: int = 1
    adamw_lr: float = 0.0001
    weight_decay: float = 0.03
    use_pk_sampler: bool = True
    num_frames: int = 8
    image_size: int = 224
    class_names: tuple[str, str, str] = ("inaction", "move", "work")


DEPLOYMENT = DeploymentConfig()
DEFAULT_TIMESFORMER_CHECKPOINT = DEFAULT_WEIGHTS_DIR / "timesformer_checkpoint.pt"
DEFAULT_BAGGING_MODEL = DEFAULT_WEIGHTS_DIR / "bagging_best.joblib"
DEFAULT_LABELS = DEFAULT_WEIGHTS_DIR / "labels.json"
DEFAULT_HF_MODEL_DIR = DEFAULT_WEIGHTS_DIR / "timesformer_hf"


def load_labels(path: str | Path = DEFAULT_LABELS) -> dict[int, str]:
    labels_path = Path(path)
    if not labels_path.is_file():
        return {index: name for index, name in enumerate(DEPLOYMENT.class_names)}

    raw = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = _normalise_label_mapping(raw)
    expected = set(DEPLOYMENT.class_names)
    if set(labels.values()) != expected:
        raise ValueError(f"labels.json must contain exactly {sorted(expected)}")
    if sorted(labels) != list(range(len(labels))):
        raise ValueError("labels.json indexes must be contiguous from zero")
    return labels


def _normalise_label_mapping(raw: Mapping[str, object]) -> dict[int, str]:
    labels: dict[int, str] = {}
    for key, value in raw.items():
        if str(key).isdigit():
            labels[int(key)] = str(value)
        else:
            labels[int(value)] = str(key)
    return labels

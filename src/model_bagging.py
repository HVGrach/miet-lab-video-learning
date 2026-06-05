from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import DEFAULT_BAGGING_MODEL


def load_bagging_model(path: str | Path = DEFAULT_BAGGING_MODEL):
    try:
        import joblib
    except Exception as error:  # noqa: BLE001
        raise RuntimeError("joblib is required to load bagging_best.joblib") from error

    model_path = Path(path)
    if not model_path.is_file():
        raise ValueError(f"bagging model does not exist: {model_path}")
    try:
        return joblib.load(model_path)
    except Exception as error:  # noqa: BLE001
        raise ValueError(f"could not load bagging model {model_path}: {error}") from error


def predict_bagging_labels(model, features: np.ndarray, labels: dict[int, str]) -> list[str]:
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and int(expected) != int(features.shape[1]):
        raise ValueError(
            f"embedding feature dimension mismatch: model expects {expected}, "
            f"got {features.shape[1]}"
        )
    raw = model.predict(features)
    return [_label_name(value, labels) for value in raw]


def _label_name(value, labels: dict[int, str]) -> str:
    if isinstance(value, str):
        if value not in labels.values():
            raise ValueError(f"model returned unknown class label: {value}")
        return value
    index = int(value)
    if index not in labels:
        raise ValueError(f"model returned unknown class index: {index}")
    return labels[index]

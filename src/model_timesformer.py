from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .config import DEPLOYMENT, DEFAULT_HF_MODEL_DIR, DEFAULT_TIMESFORMER_CHECKPOINT


def load_timesformer_model(
    *,
    checkpoint_path: str | Path = DEFAULT_TIMESFORMER_CHECKPOINT,
    hf_model_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    allow_runtime_downloads: bool = False,
) -> torch.nn.Module:
    effective_allow_downloads = (
        allow_runtime_downloads or os.environ.get("LAB6_ALLOW_RUNTIME_DOWNLOADS") == "1"
    )
    try:
        from transformers import TimesformerForVideoClassification
    except Exception as error:  # noqa: BLE001
        raise RuntimeError("transformers with TimesformerForVideoClassification is required") from error

    resolved_device = torch.device(device)
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise ValueError(
            f"TimeSFormer checkpoint is required but was not found: {checkpoint_file}"
        )

    model_source = _resolve_model_source(hf_model_path, effective_allow_downloads)
    model = TimesformerForVideoClassification.from_pretrained(
        model_source,
        num_labels=len(DEPLOYMENT.class_names),
        ignore_mismatched_sizes=True,
        local_files_only=not effective_allow_downloads,
    )
    load_weights_into_model(model, checkpoint_file, strict=False)
    model.to(resolved_device)
    model.eval()
    return model


@torch.inference_mode()
def predict_timesformer_label(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    labels: dict[int, str],
    *,
    device: str | torch.device = "cpu",
) -> str:
    resolved_device = torch.device(device)
    logits = model(pixel_values=pixel_values.to(resolved_device)).logits
    index = int(torch.argmax(logits, dim=1).item())
    return labels[index]


def load_weights_into_model(
    model: torch.nn.Module,
    ckpt_path: str | Path,
    *,
    strict: bool = False,
    min_match_ratio: float = 0.8,
) -> dict[str, Any]:
    checkpoint = _safe_torch_load(Path(ckpt_path))
    state_dict = _extract_state_dict(checkpoint, ckpt_path)
    model_state = model.state_dict()
    required_keys = {"classifier.weight", "classifier.bias"}

    candidates = []
    for prefix in ["", "module.", "model.", "backbone.", "net.", "_orig_mod."]:
        cleaned = {}
        for key, value in state_dict.items():
            new_key = key[len(prefix) :] if prefix and key.startswith(prefix) else key
            cleaned[new_key] = value
        candidates.append(cleaned)

    first_error: Exception | None = None
    best_report = {"matched": 0, "ratio": 0.0, "missing_required": sorted(required_keys)}
    for candidate in candidates:
        matched_keys = _matched_tensor_keys(model_state, candidate)
        matched_required = required_keys.intersection(matched_keys)
        missing_required = sorted(required_keys - matched_required)
        match_ratio = len(matched_keys) / max(1, len(model_state))
        if missing_required or match_ratio < min_match_ratio:
            if len(matched_keys) > int(best_report["matched"]):
                best_report = {
                    "matched": len(matched_keys),
                    "ratio": match_ratio,
                    "missing_required": missing_required,
                }
            continue
        try:
            incompatible = model.load_state_dict(candidate, strict=strict)
            return {
                "loaded": True,
                "path": str(ckpt_path),
                "missing": list(incompatible.missing_keys),
                "unexpected": list(incompatible.unexpected_keys),
                "matched": len(matched_keys),
                "match_ratio": match_ratio,
                "strict": strict,
            }
        except Exception as error:  # noqa: BLE001
            first_error = error

    raise RuntimeError(
        "checkpoint is not compatible enough to load safely: "
        f"{ckpt_path}; best_match={best_report}; "
        f"required={sorted(required_keys)}; min_match_ratio={min_match_ratio}"
    ) from first_error


def _resolve_model_source(
    hf_model_path: str | Path | None,
    allow_runtime_downloads: bool,
) -> str:
    if hf_model_path is not None:
        path = Path(hf_model_path)
        if path.is_dir():
            return str(path)
        if not allow_runtime_downloads:
            raise ValueError(f"local HuggingFace model directory does not exist: {path}")
    if DEFAULT_HF_MODEL_DIR.is_dir():
        return str(DEFAULT_HF_MODEL_DIR)
    if allow_runtime_downloads:
        return DEPLOYMENT.timesformer_model_name
    raise ValueError(
        f"local HuggingFace model directory does not exist: {DEFAULT_HF_MODEL_DIR}; "
        "set LAB6_ALLOW_RUNTIME_DOWNLOADS=1 only outside offline deployment"
    )


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state_dict(checkpoint: Any, ckpt_path: str | Path) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "weights", "module", "ema_state_dict"]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        tensor_values = [torch.is_tensor(value) for value in checkpoint.values()]
        if tensor_values and sum(tensor_values) / len(tensor_values) > 0.8:
            return checkpoint
    raise ValueError(f"cannot find state_dict inside checkpoint: {ckpt_path}")


def _matched_tensor_keys(
    model_state: dict[str, torch.Tensor],
    candidate_state: dict[str, torch.Tensor],
) -> set[str]:
    return {
        key
        for key, value in candidate_state.items()
        if key in model_state
        and torch.is_tensor(value)
        and model_state[key].shape == value.shape
    }

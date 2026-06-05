from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEPLOYMENT  # noqa: E402
from src.preprocess_frames import preprocess_image  # noqa: E402


LABEL_TO_INDEX = {label: index for index, label in enumerate(DEPLOYMENT.class_names)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}


class TimeSFormerManifestDataset(Dataset):
    def __init__(self, dataset_root: Path, frame: pd.DataFrame) -> None:
        self.dataset_root = dataset_root
        self.frame = frame.reset_index(drop=True).copy()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        rel_paths = json.loads(row["frame_paths"])
        tensors = []
        for rel_path in rel_paths:
            with Image.open(self.dataset_root / rel_path) as image:
                tensors.append(preprocess_image(image))
        return {
            "pixel_values": torch.stack(tensors, dim=0),
            "labels": torch.tensor(LABEL_TO_INDEX[row["label"]], dtype=torch.long),
            "sample_id": str(row["sample_id"]),
        }


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
        "sample_id": [item["sample_id"] for item in batch],
    }


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(hf_model_path: Path):
    from transformers import TimesformerForVideoClassification

    return TimesformerForVideoClassification.from_pretrained(
        str(hf_model_path),
        num_labels=len(DEPLOYMENT.class_names),
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )


def freeze_except_classifier(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if name.startswith("classifier."):
            parameter.requires_grad = True


def class_weights(labels: pd.Series) -> torch.Tensor:
    indexes = [LABEL_TO_INDEX[label] for label in labels]
    counts = np.bincount(indexes, minlength=len(DEPLOYMENT.class_names)).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, loader, optimizer, criterion, device, max_batches: int | None) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch_index, batch in enumerate(loader):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(pixel_values=pixel_values).logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        count = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * count
        total_count += count
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    return {"loss": total_loss / max(1, total_count)}


@torch.inference_mode()
def evaluate(model, loader, criterion, device, max_batches: int | None) -> tuple[dict[str, object], pd.DataFrame]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    rows: list[dict[str, object]] = []
    total_loss = 0.0
    total_count = 0
    for batch_index, batch in enumerate(loader):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        logits = model(pixel_values=pixel_values).logits
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)
        true = labels.detach().cpu().numpy()
        for sample_id, true_idx, pred_idx, prob in zip(batch["sample_id"], true, preds, probs):
            y_true.append(int(true_idx))
            y_pred.append(int(pred_idx))
            rows.append(
                {
                    "sample_id": sample_id,
                    "true_label": INDEX_TO_LABEL[int(true_idx)],
                    "pred_label": INDEX_TO_LABEL[int(pred_idx)],
                    "prob_inaction": float(prob[0]),
                    "prob_move": float(prob[1]),
                    "prob_work": float(prob[2]),
                }
            )
        count = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * count
        total_count += count
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, total_count)
    return metrics, pd.DataFrame(rows)


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, object]:
    confusion = np.zeros((len(DEPLOYMENT.class_names), len(DEPLOYMENT.class_names)), dtype=np.int64)
    for true_idx, pred_idx in zip(y_true, y_pred):
        confusion[true_idx, pred_idx] += 1

    per_class = {}
    f1_values = []
    for index, label in INDEX_TO_LABEL.items():
        tp = float(confusion[index, index])
        fp = float(confusion[:, index].sum() - confusion[index, index])
        fn = float(confusion[index, :].sum() - confusion[index, index])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[index, :].sum()),
        }
        f1_values.append(f1)
    accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if y_true else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a minimal TimeSFormer classifier head.")
    parser.add_argument("--dataset-root", default="lab6")
    parser.add_argument("--split-csv", default="outputs/manifests/train_val_split_hash_guarded.csv")
    parser.add_argument("--hf-model-path", default="weights/timesformer_hf")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--copy-deploy", action="store_true")
    args = parser.parse_args(argv)

    run_id = args.run_id or time.strftime("timesformer_head_%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "outputs" / "timesformer_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    split = pd.read_csv(PROJECT_ROOT / args.split_csv)
    train_df = split[split["split"] == "train"].copy()
    val_df = split[split["split"] == "val"].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("guarded split must contain train and val rows")

    split.to_csv(run_dir / "train_val_split_hash_guarded.csv", index=False)
    config = {
        "run_id": run_id,
        "dataset_root": args.dataset_root,
        "split_csv": args.split_csv,
        "hf_model_path": args.hf_model_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": args.device,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
        "deployment": DEPLOYMENT.__dict__,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    device = select_device(args.device)
    train_loader = DataLoader(
        TimeSFormerManifestDataset(PROJECT_ROOT / args.dataset_root, train_df),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        TimeSFormerManifestDataset(PROJECT_ROOT / args.dataset_root, val_df),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    model = load_model(PROJECT_ROOT / args.hf_model_path)
    freeze_except_classifier(model)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights(train_df["label"]).to(device))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=DEPLOYMENT.weight_decay,
    )

    best_macro_f1 = -1.0
    epoch_rows = []
    best_predictions = pd.DataFrame()
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, args.max_train_batches)
        val_metrics, predictions = evaluate(model, val_loader, criterion, device, args.max_val_batches)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        epoch_rows.append(row)
        pd.DataFrame(epoch_rows).to_csv(run_dir / "epoch_metrics.csv", index=False)
        (run_dir / "metrics_latest.json").write_text(json.dumps(val_metrics, indent=2, ensure_ascii=False))
        predictions.to_csv(run_dir / "val_predictions_latest.csv", index=False)
        print(json.dumps(row, ensure_ascii=False))

        if float(val_metrics["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_predictions = predictions.copy()
            checkpoint = {
                "model": model.state_dict(),
                "label_to_index": LABEL_TO_INDEX,
                "config": config,
                "epoch": epoch,
                "val_metrics": val_metrics,
            }
            torch.save(checkpoint, run_dir / "model_best.pt")
            (run_dir / "metrics.json").write_text(json.dumps(val_metrics, indent=2, ensure_ascii=False))

    per_class = pd.DataFrame.from_dict(
        json.loads((run_dir / "metrics.json").read_text())["per_class"],
        orient="index",
    )
    per_class.index.name = "label"
    per_class.reset_index().to_csv(run_dir / "per_class_metrics.csv", index=False)
    confusion = np.asarray(json.loads((run_dir / "metrics.json").read_text())["confusion_matrix"], dtype=np.int64)
    pd.DataFrame(confusion, index=DEPLOYMENT.class_names, columns=DEPLOYMENT.class_names).to_csv(run_dir / "confusion_matrix.csv")
    best_predictions.to_csv(run_dir / "val_predictions.csv", index=False)

    if args.copy_deploy:
        shutil.copy2(run_dir / "model_best.pt", PROJECT_ROOT / "weights" / "timesformer_checkpoint.pt")

    print(f"saved: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

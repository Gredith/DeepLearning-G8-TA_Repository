"""Reproducible Xception training entry point for BCE and binary focal loss."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

np = torch = accuracy_score = confusion_matrix = roc_auc_score = DataLoader = None
FaceCropDataset = BinaryFocalLoss = build_xception = None


def _load_runtime() -> None:
    """Import heavy ML dependencies only when training is requested."""
    global np, torch, accuracy_score, confusion_matrix, roc_auc_score, DataLoader
    global FaceCropDataset, BinaryFocalLoss, build_xception
    import numpy as numpy_module
    import torch as torch_module
    from sklearn.metrics import accuracy_score as accuracy_metric
    from sklearn.metrics import confusion_matrix as confusion_metric
    from sklearn.metrics import roc_auc_score as auc_metric
    from torch.utils.data import DataLoader as loader_class

    try:
        from .data_pipeline.dataset import FaceCropDataset as dataset_class
        from .models.losses import BinaryFocalLoss as focal_loss
        from .models.xception import build_xception as model_factory
    except ImportError:
        from data_pipeline.dataset import FaceCropDataset as dataset_class
        from models.losses import BinaryFocalLoss as focal_loss
        from models.xception import build_xception as model_factory

    np, torch = numpy_module, torch_module
    accuracy_score, confusion_matrix, roc_auc_score = accuracy_metric, confusion_metric, auc_metric
    DataLoader, FaceCropDataset = loader_class, dataset_class
    BinaryFocalLoss, build_xception = focal_loss, model_factory


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _safe_auc(labels: list[int], probabilities: list[float]) -> float:
    return float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else float("nan")


def collect_metrics(labels: list[int], probabilities: list[float], video_ids: list[str], threshold: float) -> dict:
    by_video: dict[str, list[tuple[int, float]]] = {}
    for label, probability, video_id in zip(labels, probabilities, video_ids):
        by_video.setdefault(video_id, []).append((label, probability))
    video_labels = [int(np.mean([item[0] for item in values]) >= 0.5) for values in by_video.values()]
    video_probabilities = [float(np.mean([item[1] for item in values])) for values in by_video.values()]
    frame_predictions = [int(value >= threshold) for value in probabilities]
    return {
        "auc_frame": _safe_auc(labels, probabilities),
        "auc_video": _safe_auc(video_labels, video_probabilities),
        "accuracy_frame": float(accuracy_score(labels, frame_predictions)),
        "confusion_matrix_frame": confusion_matrix(labels, frame_predictions, labels=[0, 1]).tolist(),
        "n_frames": len(labels),
        "n_videos": len(by_video),
    }


def _loader(config: dict, split: str, *, shuffle: bool) -> DataLoader:
    paths = config["paths"]
    data = config["data"]
    train_manipulations = list(data["train_manipulations"])
    dataset = FaceCropDataset(
        manifest=paths.get("crop_manifest", "results/splits/crop_manifest.csv"),
        split=split,
        processed_root=paths["processed_root"],
        horizontal_flip=bool(data.get("horizontal_flip", True)) and split == "train",
        manipulations=train_manipulations,
        normalization_mean=tuple(data.get("normalization_mean", (0.485, 0.456, 0.406))),
        normalization_std=tuple(data.get("normalization_std", (0.229, 0.224, 0.225))),
    )
    generator = torch.Generator().manual_seed(int(config.get("seed", 42))) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(config["training"].get("num_workers", 0)) > 0,
        generator=generator,
    )


def _run_epoch(model, loader, criterion, device, *, optimizer=None, scaler=None, amp_enabled=False, threshold=0.5):
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    labels_all: list[int] = []
    probabilities_all: list[float] = []
    video_ids_all: list[str] = []
    for images, labels, video_ids, _manipulations in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images).reshape(-1)
                loss = criterion(logits, labels)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        losses.append(float(loss.detach().cpu()))
        probabilities_all.extend(torch.sigmoid(logits.detach()).float().cpu().tolist())
        labels_all.extend(labels.detach().int().cpu().tolist())
        video_ids_all.extend(list(video_ids))
    metrics = collect_metrics(labels_all, probabilities_all, video_ids_all, threshold=threshold)
    return float(np.mean(losses)), metrics


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _plot_history(history: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_auc_video"] for row in history], marker="o")
    axes[1].set(title="Validation AUC by video", xlabel="Epoch", ylabel="AUC", ylim=(0, 1))
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def train(config: dict, run_name: str, loss_name: str, gamma: float | None = None) -> Path:
    _load_runtime()
    set_seed(int(config.get("seed", 42)))
    run_config = deepcopy(config)
    run_config["loss"]["type"] = loss_name
    if gamma is not None:
        run_config["loss"]["gamma"] = float(gamma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_xception(
        pretrained=bool(run_config["model"].get("pretrained", True)),
        name=run_config["model"].get("name", "legacy_xception"),
    ).to(device)
    criterion = (
        torch.nn.BCEWithLogitsLoss()
        if loss_name == "bce"
        else BinaryFocalLoss(
            gamma=float(run_config["loss"].get("gamma", 2.0)),
            alpha=run_config["loss"].get("alpha"),
        )
    )
    training_config = run_config["training"]
    optimizer_name = str(training_config.get("optimizer", "adam")).lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=float(training_config["learning_rate"]))
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_config["learning_rate"]))
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    train_loader = _loader(run_config, "train", shuffle=True)
    val_loader = _loader(run_config, "val", shuffle=False)
    amp_enabled = bool(training_config.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    output_root = Path(run_config["paths"].get("results_root", "results"))
    checkpoint_path = output_root / "checkpoints" / f"{run_name}.pt"
    metrics_path = output_root / "metrics" / f"{run_name}.csv"
    metadata_path = output_root / "metrics" / f"{run_name}.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_auc = -math.inf
    for epoch in range(1, int(training_config["epochs"]) + 1):
        train_loss, _ = _run_epoch(model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler, amp_enabled=amp_enabled)
        val_loss, val_metrics = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            amp_enabled=False,
            threshold=float(run_config.get("evaluation", {}).get("threshold", 0.5)),
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc_frame": val_metrics["auc_frame"],
            "val_auc_video": val_metrics["auc_video"],
            "val_accuracy_frame": val_metrics["accuracy_frame"],
        }
        history.append(row)
        print(
            f"epoch {epoch}/{training_config['epochs']} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_auc_video={val_metrics['auc_video']:.4f}"
        )
        if math.isfinite(val_metrics["auc_video"]) and val_metrics["auc_video"] > best_auc:
            best_auc = val_metrics["auc_video"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val_auc_video": best_auc,
                "config": run_config,
                "git_commit": _git_commit(),
            }, checkpoint_path)
        with metrics_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

    if best_auc == -math.inf:
        raise RuntimeError("Validation AUC by video is undefined; verify that val includes both classes.")
    _plot_history(history, output_root / "figures" / f"{run_name}_training.png")
    metadata = {
        "run_name": run_name,
        "loss": run_config["loss"],
        "seed": run_config.get("seed"),
        "git_commit": _git_commit(),
        "device": str(device),
        "packages": {name: _package_version(name) for name in ("torch", "torchvision", "timm", "scikit-learn")},
        "best_val_auc_video": best_auc,
        "checkpoint": str(checkpoint_path),
        "config": run_config,
        "history": history,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--processed-root", help="Override paths.processed_root, useful on Kaggle")
    parser.add_argument("--crop-manifest", help="Override paths.crop_manifest")
    parser.add_argument("--results-root", help="Override paths.results_root")
    args = parser.parse_args()
    import yaml

    with Path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if args.processed_root:
        config["paths"]["processed_root"] = args.processed_root
    if args.crop_manifest:
        config["paths"]["crop_manifest"] = args.crop_manifest
    if args.results_root:
        config["paths"]["results_root"] = args.results_root
    train(config, args.run_name, args.loss, args.gamma)


if __name__ == "__main__":
    main()

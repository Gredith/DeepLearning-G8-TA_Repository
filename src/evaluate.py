"""Evaluate a trained checkpoint by frame and by video."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


def _load_runtime():
    import numpy as np
    import torch
    import yaml
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
    from torch.utils.data import DataLoader

    try:
        from .data_pipeline.dataset import FaceCropDataset
        from .models.xception import build_xception
    except ImportError:
        from data_pipeline.dataset import FaceCropDataset
        from models.xception import build_xception
    return np, torch, yaml, accuracy_score, confusion_matrix, roc_auc_score, DataLoader, FaceCropDataset, build_xception


def evaluate(
    config: dict,
    checkpoint_path: Path,
    split: str = "test",
    manipulations: list[str] | None = None,
    path_overrides: dict | None = None,
) -> dict:
    np, torch, _yaml, accuracy_score, confusion_matrix, roc_auc_score, DataLoader, FaceCropDataset, build_xception = _load_runtime()
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only were introduced
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_config = deepcopy(checkpoint.get("config", config))
    if path_overrides:
        saved_config.setdefault("paths", {}).update(path_overrides)
    model_cfg = saved_config["model"]
    model = build_xception(
        pretrained=False,
        name=model_cfg.get("name", "legacy_xception"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    paths = saved_config["paths"]
    selected_manipulations = None if manipulations is None else ["original", *manipulations]
    dataset = FaceCropDataset(
        manifest=paths.get("crop_manifest", "results/splits/crop_manifest.csv"),
        split=split,
        processed_root=paths["processed_root"],
        manipulations=selected_manipulations,
        horizontal_flip=False,
        normalization_mean=tuple(saved_config["data"].get("normalization_mean", (0.485, 0.456, 0.406))),
        normalization_std=tuple(saved_config["data"].get("normalization_std", (0.229, 0.224, 0.225))),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(saved_config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(saved_config["training"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    labels: list[int] = []
    probabilities: list[float] = []
    video_ids: list[str] = []
    manipulations_seen: list[str] = []
    with torch.inference_mode():
        for images, targets, ids, manipulation_names in loader:
            logits = model(images.to(device, non_blocking=True)).reshape(-1)
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(targets.int().tolist())
            video_ids.extend(ids)
            manipulations_seen.extend(manipulation_names)

    by_video: dict[str, list[tuple[int, float]]] = {}
    for label, probability, video_id in zip(labels, probabilities, video_ids):
        by_video.setdefault(video_id, []).append((label, probability))
    video_labels = [int(np.mean([item[0] for item in values]) >= 0.5) for values in by_video.values()]
    video_probabilities = [float(np.mean([item[1] for item in values])) for values in by_video.values()]
    threshold = float(saved_config.get("evaluation", {}).get("threshold", 0.5))
    frame_predictions = [int(value >= threshold) for value in probabilities]
    metric = {
        "split": split,
        "manipulations": sorted(set(manipulations_seen)),
        "auc_frame": float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None,
        "auc_video": float(roc_auc_score(video_labels, video_probabilities)) if len(set(video_labels)) == 2 else None,
        "accuracy_frame": float(accuracy_score(labels, frame_predictions)),
        "confusion_matrix_frame": confusion_matrix(labels, frame_predictions, labels=[0, 1]).tolist(),
        "n_frames": len(labels),
        "n_videos": len(by_video),
        "checkpoint": str(checkpoint_path),
        "git_commit": checkpoint.get("git_commit"),
    }
    if metric["n_frames"] == 0:
        raise ValueError(f"No samples found for split={split!r} and manipulations={manipulations}")
    output_root = Path(saved_config["paths"].get("results_root", "results")) / "metrics"
    output_root.mkdir(parents=True, exist_ok=True)
    output_name = f"{checkpoint_path.stem}_{split}"
    if manipulations:
        output_name += "_" + "-".join(manipulations)
    (output_root / f"{output_name}.json").write_text(json.dumps(metric, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metric, indent=2))
    return metric


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--manipulation", action="append", dest="manipulations", help="Can be repeated; includes original faces for AUC")
    parser.add_argument("--processed-root", help="Override paths.processed_root, useful on Kaggle")
    parser.add_argument("--crop-manifest", help="Override paths.crop_manifest")
    parser.add_argument("--results-root", help="Override paths.results_root")
    args = parser.parse_args()
    import yaml

    with Path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    path_overrides = {
        key: value
        for key, value in {
            "processed_root": args.processed_root,
            "crop_manifest": args.crop_manifest,
            "results_root": args.results_root,
        }.items()
        if value is not None
    }
    evaluate(config, Path(args.checkpoint), args.split, args.manipulations, path_overrides)


if __name__ == "__main__":
    main()

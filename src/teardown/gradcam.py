"""Generate matched BCE/focal Grad-CAM grids selected from BCE test errors."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import yaml


class FakeLogitTarget:
    """Target the single fake-class logit emitted by the binary model."""

    def __call__(self, model_output):
        return model_output.reshape(-1)[0]


def _load_checkpoint(path: Path, device):
    import torch
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    saved = checkpoint.get("config", {})
    try:
        from ..models.xception import build_xception
    except ImportError:  # pragma: no cover
        from models.xception import build_xception
    model_config = saved.get("model", {})
    model = build_xception(pretrained=False, name=model_config.get("name", "legacy_xception"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, saved


def _probabilities(model, dataset, batch_size: int, workers: int, device, threshold: float) -> list[dict]:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    rows = []
    offset = 0
    with torch.inference_mode():
        for images, labels, video_ids, manipulations in loader:
            logits = model(images.to(device)).reshape(-1)
            probs = torch.sigmoid(logits).cpu().tolist()
            labels = labels.int().tolist()
            for label, probability, video_id, manipulation in zip(labels, probs, video_ids, manipulations):
                rows.append({
                    "dataset_index": offset,
                    "label": label,
                    "probability": float(probability),
                    "prediction": int(probability >= threshold),
                    "video_id": video_id,
                    "manipulation": manipulation,
                })
                offset += 1
    return rows


def _select_cases(bce_rows: list[dict], count: int = 8) -> dict[str, list[dict]]:
    candidates = {
        "real_correct": [row for row in bce_rows if row["label"] == 0 and row["prediction"] == 0],
        "fake_correct": [row for row in bce_rows if row["label"] == 1 and row["prediction"] == 1],
        "false_negatives": [row for row in bce_rows if row["label"] == 1 and row["prediction"] == 0],
        "false_positives": [row for row in bce_rows if row["label"] == 0 and row["prediction"] == 1],
    }
    categories = {}
    for name, rows in candidates.items():
        selected = []
        used_videos = set()
        for row in rows:
            if row["video_id"] not in used_videos:
                selected.append(row)
                used_videos.add(row["video_id"])
                if len(selected) == count:
                    break
        categories[name] = selected
    insufficient = {name: len(rows) for name, rows in categories.items() if len(rows) < count}
    if insufficient:
        raise ValueError(f"Need at least {count} examples in each BCE category; available: {insufficient}")
    return {name: rows[:count] for name, rows in categories.items()}


def generate(
    config: dict,
    bce_checkpoint: Path,
    focal_checkpoint: Path,
    split: str,
    output_dir: Path,
    path_overrides: dict | None = None,
) -> list[dict]:
    try:
        import numpy as np
        import torch
        from PIL import Image
        from matplotlib import pyplot as plt
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError as exc:
        raise RuntimeError("Grad-CAM requires torch, timm, matplotlib, and pytorch-grad-cam") from exc

    try:
        from ..data_pipeline.dataset import FaceCropDataset
    except ImportError:  # pragma: no cover
        from data_pipeline.dataset import FaceCropDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bce_model, bce_config = _load_checkpoint(bce_checkpoint, device)
    focal_model, _focal_config = _load_checkpoint(focal_checkpoint, device)
    source_config = deepcopy(bce_config or config)
    if path_overrides:
        source_config.setdefault("paths", {}).update(path_overrides)
    data_config = source_config["data"]
    paths = source_config["paths"]
    eval_manipulations = list(data_config.get("train_manipulations", ["original", "Deepfakes"]))
    dataset = FaceCropDataset(
        manifest=paths.get("crop_manifest", "results/splits/crop_manifest.csv"),
        split=split,
        processed_root=paths["processed_root"],
        horizontal_flip=False,
        manipulations=eval_manipulations,
        normalization_mean=tuple(data_config.get("normalization_mean", (0.485, 0.456, 0.406))),
        normalization_std=tuple(data_config.get("normalization_std", (0.229, 0.224, 0.225))),
    )
    threshold = float(source_config.get("evaluation", {}).get("threshold", 0.5))
    bce_rows = _probabilities(bce_model, dataset, int(source_config["training"]["batch_size"]), 0, device, threshold)
    focal_rows = _probabilities(focal_model, dataset, int(source_config["training"]["batch_size"]), 0, device, threshold)
    categories = _select_cases(bce_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    chosen = []
    index_to_row = dataset.rows
    # Keep the exact IDs and both model outputs so every panel can be reproduced.
    focal_by_index = {row["dataset_index"]: row for row in focal_rows}
    for category, rows in categories.items():
        figure, axes = plt.subplots(len(rows), 3, figsize=(12, 2.9 * len(rows)), squeeze=False)
        for row_index, bce_row in enumerate(rows):
            sample_index = bce_row["dataset_index"]
            sample = dataset[sample_index][0].unsqueeze(0).to(device)
            manifest_row = index_to_row[sample_index]
            original = Image.open(Path(paths["processed_root"]) / manifest_row["image_path"]).convert("RGB")
            rgb_image = np.asarray(original, dtype=np.float32) / 255.0
            for col, (model_name, model) in enumerate((("BCE", bce_model), ("Focal", focal_model))):
                with GradCAM(model=model, target_layers=[model.conv4]) as cam:
                    heatmap = cam(input_tensor=sample, targets=[FakeLogitTarget()])[0]
                overlay = show_cam_on_image(rgb_image, heatmap, use_rgb=True)
                axes[row_index, col + 1].imshow(overlay)
                output_row = bce_row if model_name == "BCE" else focal_by_index[sample_index]
                axes[row_index, col + 1].set_title(f"{model_name}: p(fake)={output_row['probability']:.3f}")
                axes[row_index, col + 1].axis("off")
            axes[row_index, 0].imshow(rgb_image)
            axes[row_index, 0].set_title(f"{'REAL' if bce_row['label'] == 0 else 'FAKE'} | {bce_row['video_id']} | {manifest_row['manipulation']}")
            axes[row_index, 0].axis("off")
            chosen.append({
                "category_selected_by_bce": category,
                "split": split,
                "video_id": bce_row["video_id"],
                "manipulation": bce_row["manipulation"],
                "label": bce_row["label"],
                "bce_probability": bce_row["probability"],
                "bce_prediction": bce_row["prediction"],
                "focal_probability": focal_by_index[sample_index]["probability"],
                "focal_prediction": focal_by_index[sample_index]["prediction"],
                "image_path": manifest_row["image_path"],
            })
        figure.suptitle(f"{category.replace('_', ' ').title()} — Grad-CAM to fake logit")
        figure.tight_layout()
        figure.savefig(output_dir / f"gradcam_bce_vs_focal_{category}.png", dpi=150)
        plt.close(figure)

    selection_path = output_dir / "gradcam_selection.csv"
    with selection_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(chosen[0]))
        writer.writeheader()
        writer.writerows(chosen)
    (output_dir / "gradcam_metadata.json").write_text(json.dumps({
        "split": split,
        "selected_by": f"BCE model at threshold {threshold}",
        "target_layer": "conv4",
        "cam_target": "single fake-class logit",
        "bce_checkpoint": str(bce_checkpoint),
        "focal_checkpoint": str(focal_checkpoint),
        "count_per_category": 8,
    }, indent=2) + "\n", encoding="utf-8")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--bce-checkpoint", required=True)
    parser.add_argument("--focal-checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", default="results/figures/gradcam")
    parser.add_argument("--processed-root", help="Override paths.processed_root")
    parser.add_argument("--crop-manifest", help="Override paths.crop_manifest")
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    path_overrides = {
        key: value
        for key, value in {"processed_root": args.processed_root, "crop_manifest": args.crop_manifest}.items()
        if value is not None
    }
    selected = generate(
        config,
        Path(args.bce_checkpoint),
        Path(args.focal_checkpoint),
        args.split,
        Path(args.output_dir),
        path_overrides,
    )
    print(f"Generated {len(selected)} matched Grad-CAM examples in {args.output_dir}")


if __name__ == "__main__":
    main()

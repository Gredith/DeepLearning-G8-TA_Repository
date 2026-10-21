"""Evaluate BCE/focal checkpoints on in-domain and cross-manipulation FF++ test data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

try:
    from .evaluate import evaluate
except ImportError:  # pragma: no cover - direct CLI invocation
    from evaluate import evaluate

MANIPULATIONS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Checkpoint must use LABEL=PATH, e.g. BCE=results/checkpoints/xception-bce.pt")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("Checkpoint label and path must both be non-empty")
    return label, Path(path)


def evaluate_checkpoints(
    config: dict,
    checkpoints: list[tuple[str, Path]],
    split: str,
    output_path: Path,
    path_overrides: dict | None = None,
) -> list[dict]:
    rows = []
    for label, checkpoint in checkpoints:
        in_domain = evaluate(config, checkpoint, split, ["Deepfakes"], path_overrides)
        # Gamma and seed are in the checkpoint metadata; read once without loading model weights.
        import torch
        try:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            saved = torch.load(checkpoint, map_location="cpu")
        run_config = saved.get("config", {})
        loss_cfg = run_config.get("loss", {})
        gamma = None if label.lower() == "bce" else loss_cfg.get("gamma")
        seed = run_config.get("seed")
        for manipulation in MANIPULATIONS:
            cross = in_domain if manipulation == "Deepfakes" else evaluate(config, checkpoint, split, [manipulation], path_overrides)
            auc_in_frame = in_domain.get("auc_frame")
            auc_cross_frame = cross.get("auc_frame")
            auc_in_video = in_domain.get("auc_video")
            auc_cross_video = cross.get("auc_video")
            rows.append({
                "model": label,
                "gamma": gamma,
                "seed": seed,
                "split": split,
                "manipulation": manipulation,
                "in_domain_auc_frame": auc_in_frame,
                "cross_auc_frame": auc_cross_frame,
                "relative_auc_drop_frame": (
                    (auc_in_frame - auc_cross_frame) / auc_in_frame
                    if auc_in_frame not in (None, 0) and auc_cross_frame is not None else None
                ),
                "in_domain_auc_video": auc_in_video,
                "cross_auc_video": auc_cross_video,
                "relative_auc_drop_video": (
                    (auc_in_video - auc_cross_video) / auc_in_video
                    if auc_in_video not in (None, 0) and auc_cross_video is not None else None
                ),
                "n_frames": cross.get("n_frames"),
                "n_videos": cross.get("n_videos"),
            })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True, help="Repeat as LABEL=PATH for BCE, Focal-g0, Focal-g2")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", default="results/metrics/cross_manipulation.csv")
    parser.add_argument("--processed-root")
    parser.add_argument("--crop-manifest")
    parser.add_argument("--results-root")
    args = parser.parse_args()
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
    rows = evaluate_checkpoints(config, args.checkpoint, args.split, Path(args.output), path_overrides)
    print(f"Wrote {len(rows)} model/manipulation rows to {args.output}")


if __name__ == "__main__":
    main()

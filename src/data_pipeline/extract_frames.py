"""Extract uniformly spaced frames from the video manifest using OpenCV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

try:  # supports both ``python src/data_pipeline/extract_frames.py`` and package imports
    from .build_splits import load_config
except ImportError:  # pragma: no cover - used by direct script invocation
    from build_splits import load_config


def extract_frames(config: dict, video_manifest: Path, frame_manifest: Path) -> int:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Frame extraction requires opencv-python and numpy from environment/requirements.txt") from exc

    data_root = Path(config["paths"]["data_root"]).expanduser().resolve()
    processed_root = Path(config["paths"]["processed_root"]).expanduser().resolve()
    frames_per_video = int(config["data"].get("frames_per_video", 25))
    if frames_per_video < 1:
        raise ValueError("data.frames_per_video must be at least 1")

    frame_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_root = processed_root / "frames"
    fields = ["split", "image_path", "video_id", "source_ids", "label", "manipulation", "frame_index"]
    rows: list[dict] = []
    failures: Counter[str] = Counter()
    with video_manifest.open(newline="", encoding="utf-8") as stream:
        videos = list(csv.DictReader(stream))

    for video in videos:
        video_path = data_root / video["video_path"]
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            failures["open_failed"] += 1
            cap.release()
            continue
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            failures["empty_video"] += 1
            cap.release()
            continue
        indices = np.unique(np.linspace(0, count - 1, min(frames_per_video, count), dtype=int))
        clip_dir = out_root / video["manipulation"] / video["video_id"]
        clip_dir.mkdir(parents=True, exist_ok=True)
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame_bgr = cap.read()
            if not ok:
                failures["read_failed"] += 1
                continue
            image_path = clip_dir / f"frame_{int(index):06d}.jpg"
            if not cv2.imwrite(str(image_path), frame_bgr):
                failures["write_failed"] += 1
                continue
            rows.append({
                "split": video["split"],
                "image_path": image_path.relative_to(processed_root).as_posix(),
                "video_id": video["video_id"],
                "source_ids": video["source_ids"],
                "label": video["label"],
                "manipulation": video["manipulation"],
                "frame_index": int(index),
            })
        cap.release()

    with frame_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter((row["split"], row["manipulation"], int(row["label"])) for row in rows)
    counts_path = frame_manifest.parent / "frame_counts.csv"
    with counts_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "manipulation", "label", "frames"])
        for (split, manipulation, label), total in sorted(counts.items()):
            writer.writerow([split, manipulation, label, total])

    failure_path = frame_manifest.parent / "frame_extraction_failures.json"
    failure_path.write_text(json.dumps(dict(failures), indent=2) + "\n", encoding="utf-8")
    print(f"Extracted {len(rows)} frames from {len(videos)} videos; failures: {dict(failures)}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--video-manifest", default="results/splits/video_manifest.csv")
    parser.add_argument("--output", default="results/splits/frame_manifest.csv")
    parser.add_argument("--data-root", help="Override paths.data_root")
    parser.add_argument("--processed-root", help="Override paths.processed_root")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data_root:
        config["paths"]["data_root"] = args.data_root
    if args.processed_root:
        config["paths"]["processed_root"] = args.processed_root
    extract_frames(config, Path(args.video_manifest), Path(args.output))


if __name__ == "__main__":
    main()

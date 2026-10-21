"""Detect the largest face, expand to a square, and cache resized crops."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

try:
    from .build_splits import load_config
except ImportError:  # pragma: no cover - used by direct script invocation
    from build_splits import load_config


def expanded_square(box: tuple[float, float, float, float], image_size: tuple[int, int], margin: float = 1.3) -> tuple[int, int, int, int]:
    """Return a clamped square crop centered on a detection, with a scale margin."""
    x1, y1, x2, y2 = box
    width, height = image_size
    side = max(x2 - x1, y2 - y1) * margin
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = max(0, int(round(cx - side / 2)))
    top = max(0, int(round(cy - side / 2)))
    right = min(width, int(round(cx + side / 2)))
    bottom = min(height, int(round(cy + side / 2)))
    if right <= left or bottom <= top:
        raise ValueError(f"Detected face box is outside the image bounds: {box}")
    # Keep the crop square where the image boundary permits it.
    side_px = min(right - left, bottom - top)
    return left, top, left + side_px, top + side_px


def make_sample_grid(rows: list[dict], processed_root: Path, output_path: Path) -> bool:
    selected = {label: [] for label in (0, 1)}
    seen_videos = {label: set() for label in (0, 1)}
    for row in sorted(rows, key=lambda item: item["image_path"]):
        label = int(row["label"])
        if len(selected[label]) < 16 and row["video_id"] not in seen_videos[label]:
            selected[label].append(row)
            seen_videos[label].add(row["video_id"])
    if any(len(class_rows) < 16 for class_rows in selected.values()):
        return False
    canvas = Image.new("RGB", (4 * 320, 8 * 340), "white")
    draw = ImageDraw.Draw(canvas)
    # First 16 real, then 16 fake, laid out in two adjacent four-column blocks.
    for block, label in enumerate((0, 1)):
        for i, row in enumerate(selected[label]):
            x = (block * 2 + (i % 2)) * 320
            y = (i // 2) * 340
            image = Image.open(processed_root / row["image_path"]).convert("RGB")
            image.thumbnail((300, 300))
            canvas.paste(image, (x + (300 - image.width) // 2, y))
            draw.text((x + 4, y + 304), f"{'REAL' if label == 0 else 'FAKE'} | {row['manipulation']} | {row['video_id']}", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return True


def crop_faces(config: dict, frame_manifest: Path, output_manifest: Path, grid_path: Path | None = None) -> int:
    try:
        import torch
        from facenet_pytorch import MTCNN
    except ImportError as exc:
        raise RuntimeError("Face cropping requires torch and facenet-pytorch from environment/requirements.txt") from exc

    processed_root = Path(config["paths"]["processed_root"]).expanduser().resolve()
    image_size = int(config["data"].get("image_size", 299))
    margin = float(config["data"].get("face_margin", 1.3))
    if image_size < 1 or margin < 1:
        raise ValueError("image_size must be positive and face_margin must be >= 1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = MTCNN(keep_all=True, post_process=False, device=device)
    output_root = processed_root / "faces"
    output_root.mkdir(parents=True, exist_ok=True)

    with frame_manifest.open(newline="", encoding="utf-8") as stream:
        frames = list(csv.DictReader(stream))
    output_fields = ["split", "image_path", "video_id", "source_ids", "label", "manipulation", "frame_index"]
    rows: list[dict] = []
    failures: list[dict] = []
    reason_counts: Counter[str] = Counter()

    for frame in frames:
        source_path = processed_root / frame["image_path"]
        image = Image.open(source_path).convert("RGB")
        boxes, probabilities = detector.detect(image)
        if boxes is None or probabilities is None:
            failures.append({"image_path": frame["image_path"], "reason": "no_face"})
            reason_counts["no_face"] += 1
            continue
        valid = [(box, float(prob)) for box, prob in zip(boxes, probabilities) if box is not None and prob is not None]
        if not valid:
            failures.append({"image_path": frame["image_path"], "reason": "no_face"})
            reason_counts["no_face"] += 1
            continue
        box, _ = max(valid, key=lambda item: max(0, item[0][2] - item[0][0]) * max(0, item[0][3] - item[0][1]))
        try:
            crop_box = expanded_square(tuple(float(v) for v in box), image.size, margin=margin)
        except ValueError:
            failures.append({"image_path": frame["image_path"], "reason": "invalid_box"})
            reason_counts["invalid_box"] += 1
            continue
        crop = image.crop(crop_box).resize((image_size, image_size), Image.Resampling.BILINEAR)
        relative = Path("faces") / frame["manipulation"] / frame["video_id"] / f"frame_{int(frame['frame_index']):06d}.jpg"
        target = processed_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        crop.save(target, quality=95)
        rows.append({
            "split": frame["split"],
            "image_path": relative.as_posix(),
            "video_id": frame["video_id"],
            "source_ids": frame["source_ids"],
            "label": frame["label"],
            "manipulation": frame["manipulation"],
            "frame_index": frame["frame_index"],
        })

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)
    failures_path = output_manifest.parent / "face_detection_failures.csv"
    with failures_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_path", "reason"])
        writer.writeheader()
        writer.writerows(failures)
    counts = Counter((row["split"], row["manipulation"], int(row["label"])) for row in rows)
    counts_path = output_manifest.parent / "face_crop_counts.csv"
    with counts_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "manipulation", "label", "faces"])
        for (split, manipulation, label), total in sorted(counts.items()):
            writer.writerow([split, manipulation, label, total])

    if grid_path is not None:
        made_grid = make_sample_grid(rows, processed_root, grid_path)
        print(f"Sanity grid {'saved' if made_grid else 'not created (need real and fake crops)'}: {grid_path}")
    print(f"Saved {len(rows)} face crops; skipped {len(failures)} frames: {dict(reason_counts)}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--input", default="results/splits/frame_manifest.csv")
    parser.add_argument("--output", default="results/splits/crop_manifest.csv")
    parser.add_argument("--grid", default="results/figures/data_sanity_grid.jpg")
    parser.add_argument("--processed-root", help="Override paths.processed_root")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.processed_root:
        config["paths"]["processed_root"] = args.processed_root
    crop_faces(config, Path(args.input), Path(args.output), Path(args.grid))


if __name__ == "__main__":
    main()

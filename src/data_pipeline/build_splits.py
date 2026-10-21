"""Build an auditable video manifest from the official FF++ pair splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

import yaml

OFFICIAL_SPLIT_BASE = "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits"
SPLITS = ("train", "val", "test")
VIDEO_ID_RE = re.compile(r"\d{3}")


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def read_split_pairs(path: str | Path) -> list[tuple[str, str]]:
    """Parse the official JSON format: a list of two-source-video pairs."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Split file must contain a non-empty list of pairs: {path}")
    pairs: list[tuple[str, str]] = []
    for row in raw:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"Invalid FF++ split row in {path}: {row!r}")
        ids = tuple(str(value).zfill(3) for value in row)
        if any(not VIDEO_ID_RE.fullmatch(value) for value in ids):
            raise ValueError(f"Expected three-digit video IDs in {path}: {row!r}")
        if ids[0] == ids[1]:
            raise ValueError(f"A split pair cannot contain the same video twice: {row!r}")
        pairs.append(ids)
    return pairs


def download_official_splits(destination: Path) -> dict[str, Path]:
    """Download only the public split metadata, never the FF++ media."""
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split in SPLITS:
        path = destination / f"{split}.json"
        url = f"{OFFICIAL_SPLIT_BASE}/{split}.json"
        request = urllib.request.Request(url, headers={"User-Agent": "DL-G8-TA-Repository/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
        # Validate before persisting so an error page cannot become a split file.
        parsed = json.loads(payload)
        if not isinstance(parsed, list) or not parsed:
            raise ValueError(f"Official split download was not a non-empty JSON list: {url}")
        path.write_bytes(payload)
        read_split_pairs(path)
        paths[split] = path
    return paths


def source_id_split_map(split_paths: dict[str, Path]) -> tuple[dict[str, str], dict[str, list[tuple[str, str]]]]:
    id_to_split: dict[str, str] = {}
    pairs_by_split: dict[str, list[tuple[str, str]]] = {}
    for split in SPLITS:
        pairs = read_split_pairs(split_paths[split])
        pairs_by_split[split] = pairs
        for source_id in {video_id for pair in pairs for video_id in pair}:
            previous = id_to_split.setdefault(source_id, split)
            if previous != split:
                raise ValueError(f"Source video {source_id} appears in both {previous} and {split}")
    return id_to_split, pairs_by_split


def save_split_metadata(split_paths: dict[str, Path], output_dir: Path) -> dict:
    """Write normalized source-ID lists and provenance without requiring FF++ media."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _id_to_split, pairs_by_split = source_id_split_map(split_paths)
    metadata = {"source": f"{OFFICIAL_SPLIT_BASE}/<train|val|test>.json", "splits": {}}
    for split, pairs in pairs_by_split.items():
        source_ids = sorted({video_id for pair in pairs for video_id in pair})
        (output_dir / f"{split}_source_ids.json").write_text(json.dumps(source_ids, indent=2) + "\n", encoding="utf-8")
        metadata["splits"][split] = {
            "pairs": len(pairs),
            "unique_source_ids": len(source_ids),
            "source_json": str(split_paths[split]),
            "sha256": hashlib.sha256(split_paths[split].read_bytes()).hexdigest(),
        }
    (output_dir / "official_split_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def parse_video_identity(path: Path, manipulation: str) -> tuple[str, list[str]]:
    """Return a unique clip ID and the source IDs encoded in its FF++ filename."""
    stem = path.stem
    ids = VIDEO_ID_RE.findall(stem)
    if manipulation == "original":
        if not ids or ids[0] != stem:
            raise ValueError(f"Cannot read source ID from original video filename: {path.name}")
        source_ids = [ids[0]]
    else:
        if len(ids) < 2:
            raise ValueError(f"Cannot read source pair from manipulated video filename: {path.name}")
        source_ids = list(dict.fromkeys(ids[:2]))
        if len(source_ids) != 2:
            raise ValueError(f"Manipulated video must reference two source IDs: {path.name}")
    return stem, source_ids


def find_manipulation(path: Path, known: set[str]) -> str | None:
    # FF++ stores videos under original_sequences/... or manipulated_sequences/<name>/...
    parts = set(path.parts)
    if "original_sequences" in parts:
        return "original"
    for manipulation in known:
        if manipulation in parts:
            return manipulation
    return None


def build_video_manifest(config: dict, split_paths: dict[str, Path], output_path: Path) -> dict:
    data_cfg = config["data"]
    paths_cfg = config["paths"]
    data_root = Path(paths_cfg["data_root"]).expanduser().resolve()
    compression = str(data_cfg.get("compression", "c23"))
    manipulations = set(data_cfg["train_manipulations"]) | set(data_cfg["evaluation_manipulations"])
    id_to_split, pairs_by_split = source_id_split_map(split_paths)

    # Also write normalized, attributable ID lists for downstream analysis.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_split_metadata(split_paths, output_path.parent)

    videos = sorted(data_root.rglob("*.mp4"))
    rows = []
    skipped = Counter()
    for video_path in videos:
        manipulation = find_manipulation(video_path, manipulations)
        if manipulation is None:
            skipped["unknown_manipulation"] += 1
            continue
        if compression not in video_path.parts:
            skipped["wrong_compression"] += 1
            continue
        video_id, source_ids = parse_video_identity(video_path, manipulation)
        assignments = {id_to_split.get(source_id) for source_id in source_ids}
        if None in assignments:
            skipped["source_id_not_in_official_splits"] += 1
            continue
        if len(assignments) != 1:
            raise ValueError(f"Manipulated clip crosses official splits: {video_path} -> {assignments}")
        split = assignments.pop()
        rows.append({
            "split": split,
            "video_path": str(video_path.relative_to(data_root)),
            "video_id": video_id,
            "source_ids": ";".join(source_ids),
            "label": 0 if manipulation == "original" else 1,
            "manipulation": manipulation,
            "compression": compression,
        })

    if not rows:
        raise FileNotFoundError(
            f"No matching FF++ videos found beneath {data_root}. Expected original_sequences/ and "
            f"manipulated_sequences/ with a {compression} directory."
        )
    fieldnames = list(rows[0])
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = Counter((row["split"], row["manipulation"], row["label"]) for row in rows)
    summary_path = output_path.parent / "video_counts.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "manipulation", "label", "videos"])
        for (split, manipulation, label), count in sorted(summary.items()):
            writer.writerow([split, manipulation, label, count])

    provenance = {
        "source": f"{OFFICIAL_SPLIT_BASE}/<train|val|test>.json",
        "files": {
            split: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for split, path in split_paths.items()
        },
        "unique_source_ids": {split: len({i for pair in pairs for i in pair}) for split, pairs in pairs_by_split.items()},
        "videos_found": len(rows),
        "skipped": dict(skipped),
        "compression": compression,
    }
    (output_path.parent / "split_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    assert_disjoint_video_manifest(rows)
    return provenance


def assert_disjoint_video_manifest(rows: list[dict]) -> None:
    owners: dict[str, str] = {}
    for row in rows:
        for source_id in row["source_ids"].split(";"):
            previous = owners.setdefault(source_id, row["split"])
            if previous != row["split"]:
                raise AssertionError(f"Source video ID {source_id} leaks between {previous} and {row['split']}")


def assert_disjoint_splits(rows: list[dict]) -> None:
    """Public spelling used by dataset and test code for the source-ID leakage check."""
    assert_disjoint_video_manifest(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--split-dir", help="Directory containing official train.json, val.json and test.json")
    parser.add_argument("--data-root", help="Override paths.data_root, e.g. /kaggle/input/<dataset>/FF_data")
    parser.add_argument("--download-official", action="store_true", help="Download public split metadata from the FF++ reference repository")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root:
        config["paths"]["data_root"] = args.data_root
    split_dir = Path(args.split_dir or config["paths"].get("splits_root", "results/splits"))
    split_paths = {split: split_dir / f"{split}.json" for split in SPLITS}
    if args.download_official or not all(path.exists() for path in split_paths.values()):
        split_paths = download_official_splits(split_dir)
    metadata = save_split_metadata(split_paths, split_dir)
    output = split_dir / "video_manifest.csv"
    provenance = build_video_manifest(config, split_paths, output)
    print(f"Video manifest: {output} ({provenance['videos_found']} videos)")
    print(f"Split source IDs: { {name: details['unique_source_ids'] for name, details in metadata['splits'].items()} }")


if __name__ == "__main__":
    main()

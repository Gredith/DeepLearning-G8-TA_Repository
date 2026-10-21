"""Dependency-light checks for the FF++ split and manifest logic."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.data_pipeline.build_splits import (
    SPLITS,
    assert_disjoint_splits,
    assert_disjoint_video_manifest,
    build_video_manifest,
    parse_video_identity,
    read_split_pairs,
    save_split_metadata,
    source_id_split_map,
)
from PIL import Image

from src.data_pipeline.crop_faces import expanded_square, make_sample_grid


class DataPipelineTests(unittest.TestCase):
    def _temporary_directory(self):
        # Keep test writes inside the workspace; the managed sandbox does not expose OS temp paths.
        return tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "results")

    def _split_files(self, root: Path) -> dict[str, Path]:
        pairs = {
            "train": [["001", "002"]],
            "val": [["003", "004"]],
            "test": [["005", "006"]],
        }
        result = {}
        for split, rows in pairs.items():
            path = root / f"{split}.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            result[split] = path
        return result

    def test_reads_official_pair_format_and_builds_split_map(self):
        with self._temporary_directory() as tmp:
            root = Path(tmp)
            paths = self._split_files(root)
            self.assertEqual(read_split_pairs(paths["train"]), [("001", "002")])
            id_map, _ = source_id_split_map(paths)
            self.assertEqual(id_map["001"], "train")
            self.assertEqual(id_map["006"], "test")

    def test_saves_normalized_source_ids_with_provenance(self):
        with self._temporary_directory() as tmp:
            root = Path(tmp)
            paths = self._split_files(root)
            metadata = save_split_metadata(paths, root / "output")
            self.assertEqual(metadata["splits"]["train"]["unique_source_ids"], 2)
            self.assertEqual(json.loads((root / "output" / "train_source_ids.json").read_text()), ["001", "002"])
            self.assertTrue((root / "output" / "official_split_metadata.json").is_file())

    def test_rejects_source_id_leakage_between_official_splits(self):
        with self._temporary_directory() as tmp:
            root = Path(tmp)
            paths = self._split_files(root)
            paths["val"].write_text(json.dumps([["002", "004"]]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "appears in both"):
                source_id_split_map(paths)

    def test_parses_real_and_manipulated_ffpp_ids(self):
        self.assertEqual(parse_video_identity(Path("123.mp4"), "original"), ("123", ["123"]))
        self.assertEqual(parse_video_identity(Path("123_456_0.mp4"), "Deepfakes"), ("123_456_0", ["123", "456"]))

    def test_manifest_uses_official_partition_and_writes_counts(self):
        with self._temporary_directory() as tmp:
            root = Path(tmp)
            split_dir = root / "splits"
            split_dir.mkdir()
            paths = self._split_files(split_dir)
            data_root = root / "ffpp"
            for relative in (
                "original_sequences/youtube/c23/videos/001.mp4",
                "original_sequences/youtube/c23/videos/003.mp4",
                "original_sequences/youtube/c23/videos/005.mp4",
                "manipulated_sequences/Deepfakes/c23/videos/001_002_0.mp4",
                "manipulated_sequences/FaceSwap/c23/videos/003_004_0.mp4",
                "manipulated_sequences/NeuralTextures/c23/videos/005_006_0.mp4",
            ):
                target = data_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
            config = {
                "paths": {"data_root": str(data_root)},
                "data": {
                    "compression": "c23",
                    "train_manipulations": ["original", "Deepfakes"],
                    "evaluation_manipulations": ["original", "Face2Face", "FaceSwap", "NeuralTextures"],
                },
            }
            output = split_dir / "video_manifest.csv"
            result = build_video_manifest(config, paths, output)
            self.assertEqual(result["videos_found"], 6)
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            assignments = {(row["video_id"], row["split"]) for row in rows}
            self.assertIn(("001_002_0", "train"), assignments)
            self.assertIn(("003_004_0", "val"), assignments)
            self.assertTrue((split_dir / "video_counts.csv").is_file())
            self.assertTrue((split_dir / "split_provenance.json").is_file())
            self.assertEqual(set(SPLITS), {row["split"] for row in rows})

    def test_leakage_assertions_check_sources_not_just_clip_names(self):
        rows = [
            {"video_id": "001", "source_ids": "001", "split": "train"},
            {"video_id": "001_002_0", "source_ids": "001;002", "split": "train"},
        ]
        assert_disjoint_video_manifest(rows)
        assert_disjoint_splits(rows)
        rows.append({"video_id": "002", "source_ids": "002", "split": "val"})
        with self.assertRaises(AssertionError):
            assert_disjoint_video_manifest(rows)
        with self.assertRaises(AssertionError):
            assert_disjoint_splits(rows)

    def test_face_box_is_expanded_and_clamped(self):
        left, top, right, bottom = expanded_square((40, 20, 60, 40), (100, 80), margin=1.3)
        self.assertEqual(right - left, bottom - top)
        self.assertGreater(right - left, 20)
        self.assertGreaterEqual(left, 0)
        self.assertLessEqual(right, 100)

    def test_sanity_grid_uses_16_distinct_videos_per_class(self):
        with self._temporary_directory() as tmp:
            root = Path(tmp)
            rows = []
            for label, prefix in ((0, "real"), (1, "fake")):
                for index in range(16):
                    relative = f"{prefix}/{index:03}.jpg"
                    image_path = root / relative
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (12, 12), "white").save(image_path)
                    rows.append({"label": label, "image_path": relative, "video_id": f"{prefix}-{index}", "manipulation": "original" if label == 0 else "Deepfakes"})
            output = root / "grid.jpg"
            self.assertTrue(make_sample_grid(rows, root, output))
            with Image.open(output) as grid:
                self.assertEqual(grid.size, (1280, 2720))


if __name__ == "__main__":
    unittest.main()

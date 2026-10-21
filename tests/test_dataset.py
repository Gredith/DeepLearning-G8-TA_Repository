"""Runtime checks for the PyTorch face-crop dataset (skipped without PyTorch)."""

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is installed in the Kaggle/runtime environment")
class FaceCropDatasetTests(unittest.TestCase):
    def test_batch_shape_and_metadata(self):
        import torch
        from torch.utils.data import DataLoader

        from src.data_pipeline.dataset import FaceCropDataset

        workspace = Path(__file__).resolve().parents[1] / "results"
        with tempfile.TemporaryDirectory(dir=workspace) as tmp:
            root = Path(tmp)
            (root / "faces").mkdir()
            Image.new("RGB", (299, 299), "red").save(root / "faces" / "real.jpg")
            manifest = root / "manifest.csv"
            rows = [
                {"split": "train", "image_path": "faces/real.jpg", "video_id": "001", "source_ids": "001", "label": 0, "manipulation": "original"},
                {"split": "val", "image_path": "faces/real.jpg", "video_id": "002", "source_ids": "002", "label": 0, "manipulation": "original"},
                {"split": "test", "image_path": "faces/real.jpg", "video_id": "003", "source_ids": "003", "label": 0, "manipulation": "original"},
            ]
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            dataset = FaceCropDataset(manifest, "train", root)
            images, labels, video_ids, manipulations = next(iter(DataLoader(dataset, batch_size=1)))
            self.assertEqual(tuple(images.shape), (1, 3, 299, 299))
            self.assertEqual(tuple(labels.shape), (1,))
            self.assertEqual(video_ids, ("001",))
            self.assertEqual(manipulations, ("original",))
            self.assertTrue(torch.isfinite(images).all())


if __name__ == "__main__":
    unittest.main()

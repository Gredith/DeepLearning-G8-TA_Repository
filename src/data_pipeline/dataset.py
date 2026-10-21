"""PyTorch dataset over the precomputed face-crop CSV manifest."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

try:
    from .build_splits import assert_disjoint_splits
except ImportError:  # pragma: no cover
    from build_splits import assert_disjoint_splits

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
REQUIRED_FIELDS = {"split", "image_path", "video_id", "source_ids", "label", "manipulation"}


class FaceCropDataset(Dataset):
    """Return normalized image tensor, binary label, video ID, and manipulation."""

    def __init__(
        self,
        manifest: str | Path,
        split: str,
        processed_root: str | Path,
        transform=None,
        horizontal_flip: bool = False,
        validate_all_splits: bool = True,
        manipulations: list[str] | None = None,
        normalization_mean: tuple[float, float, float] = IMAGENET_MEAN,
        normalization_std: tuple[float, float, float] = IMAGENET_STD,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")
        self.root = Path(processed_root).expanduser().resolve()
        with Path(manifest).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not REQUIRED_FIELDS.issubset(reader.fieldnames):
                raise ValueError(f"Crop manifest must have columns: {sorted(REQUIRED_FIELDS)}")
            all_rows = list(reader)
        if validate_all_splits:
            assert_disjoint_splits(all_rows)
        self.rows = [
            row for row in all_rows
            if row["split"] == split and (manipulations is None or row["manipulation"] in manipulations)
        ]
        if not self.rows:
            raise ValueError(f"No samples found for split={split!r} in {manifest}")
        self.transform = transform
        self.horizontal_flip = horizontal_flip and split == "train"
        self.mean = torch.tensor(normalization_mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(normalization_std, dtype=torch.float32).view(3, 1, 1)
        if torch.any(self.std <= 0):
            raise ValueError("normalization_std values must be positive")
        for row in self.rows:
            try:
                label = int(row["label"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid binary label in manifest row: {row}") from exc
            if label not in (0, 1):
                raise ValueError(f"Labels must be 0 (real) or 1 (fake), got {label}")
            image_path = self.root / row["image_path"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Crop file is missing: {image_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(self.root / row["image_path"]).convert("RGB")
        if self.horizontal_flip and random.random() < 0.5:
            image = ImageOps.mirror(image)
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            array = np.asarray(image, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
            image_tensor = (image_tensor - self.mean) / self.std
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError("transform must return a torch.Tensor")
        if image_tensor.shape != (3, 299, 299):
            raise ValueError(f"Expected image tensor [3, 299, 299], got {tuple(image_tensor.shape)}")
        return (
            image_tensor,
            torch.tensor(int(row["label"]), dtype=torch.float32),
            row["video_id"],
            row["manipulation"],
        )

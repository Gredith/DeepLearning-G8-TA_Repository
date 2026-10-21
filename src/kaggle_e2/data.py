"""Splits oficiales de FaceForensics++ y Dataset de recortes de cara.

Estructura esperada de los recortes (la genera preprocess.py):
    crops/<categoria>/<video_id>/<frame>.png
    categoria ∈ {original, Deepfakes, Face2Face, FaceSwap, NeuralTextures}
    video_id: "071" (real) o "071_054" (falso: target 071, fuente 054)

Partición POR VIDEO con los splits oficiales (results/splits/{train,val,test}.json):
cada par [a, b] aporta los reales a, b y los falsos a_b, b_a al MISMO split, así
ninguna identidad aparece en dos splits (evita fuga de información, D-04).
"""
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
REAL = "original"
MEAN = STD = (0.5, 0.5, 0.5)  # normalización de legacy_xception


def load_pairs(splits_dir, split):
    return json.load(open(Path(splits_dir) / f"{split}.json"))


def subset_pairs(pairs, n_pairs=None, seed=0):
    """Subconjunto determinista de pares (para el presupuesto de GPU)."""
    pairs = list(pairs)
    if n_pairs is None or n_pairs >= len(pairs):
        return pairs
    random.Random(seed).shuffle(pairs)
    return sorted(pairs[:n_pairs])


def video_ids(pairs):
    reals = sorted({v for p in pairs for v in p})
    fakes = sorted({f"{a}_{b}" for a, b in pairs} | {f"{b}_{a}" for a, b in pairs})
    return reals, fakes


def build_items(crops_root, pairs, fake_methods, include_real=True, max_frames=None):
    """Lista de (ruta_png, etiqueta, clave_video). Etiqueta 1 = falso."""
    crops_root = Path(crops_root)
    reals, fakes = video_ids(pairs)
    groups = ([(REAL, reals, 0)] if include_real else []) + [(m, fakes, 1) for m in fake_methods]
    items, missing = [], 0
    for cat, ids, label in groups:
        for vid in ids:
            frames = sorted((crops_root / cat / vid).glob("*.png"))
            if not frames:
                missing += 1
                continue
            if max_frames:
                frames = frames[:max_frames]
            items += [(str(f), label, f"{cat}/{vid}") for f in frames]
    return items, missing


def train_transform(size=299):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform(size=299):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class FrameDataset(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, vkey = self.items[i]
        img = self.transform(Image.open(path).convert("RGB"))
        return img, torch.tensor(label, dtype=torch.float32), vkey


def describe(items):
    n_real = sum(1 for _, y, _ in items if y == 0)
    vids = {v for *_, v in items}
    return {"frames": len(items), "frames_real": n_real, "frames_fake": len(items) - n_real, "videos": len(vids)}

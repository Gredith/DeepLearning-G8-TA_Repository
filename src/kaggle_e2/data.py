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

import numpy as np
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


TRAIN_FAKES = ("deepfakes", "sbi", "deepfakes+sbi")   # D-14


def build_train_items(crops_root, pairs, train_fakes="deepfakes", train_methods=("Deepfakes",), max_frames=None,
                      val=False):
    """Ítems de train/val según la fuente de falsos (D-14). Cada ítem: (ruta, etiqueta, clave_video, tipo).

    - "deepfakes": reales + los métodos de `train_methods` (protocolo D-06, idéntico a las corridas B).
    - "sbi": reales + un SBI por frame real (el SBI se fabrica con ese mismo frame).
    - "deepfakes+sbi": reales + Deepfakes; en train, cada falso se reemplaza por un SBI con probabilidad
      `sbi_prob` (ver FrameDataset), así el tamaño de la época y el balance 1:1 no cambian respecto de D-06.
      En val (`val=True`) se agregan además los SBI deterministas de los reales de val, para que la
      selección del checkpoint mire la misma mezcla que el entrenamiento.
    Tipos: "img" (se usa tal cual), "sbi" (SBI fabricado con la ruta, que es un real), "mix" (Deepfakes o SBI).
    """
    assert train_fakes in TRAIN_FAKES, train_fakes
    reals, n_miss_r = build_items(crops_root, pairs, [], include_real=True, max_frames=max_frames)
    items = [(p, 0, v, "img") for p, _, v in reals]
    sbis = [(p, 1, "SBI/" + v.split("/", 1)[1], "sbi") for p, _, v in reals]
    if train_fakes == "sbi":
        return items + sbis, n_miss_r
    fakes, n_miss_f = build_items(crops_root, pairs, list(train_methods), include_real=False, max_frames=max_frames)
    if train_fakes == "deepfakes":
        return items + [(p, 1, v, "img") for p, _, v in fakes], n_miss_r + n_miss_f
    if val:
        return items + [(p, 1, v, "img") for p, _, v in fakes] + sbis, n_miss_r + n_miss_f
    return items + [(p, 1, v, "mix") for p, _, v in fakes], n_miss_r + n_miss_f


def train_transform(size=299):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform(size=299, jpeg_quality=None):
    """Transformación de evaluación. Con `jpeg_quality` se recomprime cada recorte antes (estrés de compresión simulada)."""
    pre = [JpegRecompress(jpeg_quality)] if jpeg_quality else []
    return transforms.Compose(pre + [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class JpegRecompress:
    """Recompresión JPEG fija (clase y no lambda: se puede enviar a los workers del DataLoader)."""
    def __init__(self, quality):
        self.quality = int(quality)

    def __call__(self, img):
        return _sbi().jpeg(img, self.quality)


def _sbi():
    import sbi  # import diferido: módulo hermano en src/kaggle_e2 (los notebooks lo ponen en sys.path)
    return sbi


class FrameDataset(Dataset):
    """Dataset de recortes. Acepta ítems de 3 campos (ruta, etiqueta, clave) o de 4 con el tipo (D-14).

    - `fixed_seed`: si se da, los SBI son deterministas por índice (val); si no, cambian en cada época.
    - `sbi_prob`: en ítems "mix", probabilidad de cambiar el Deepfakes por un SBI de un real de `real_paths`.
    - `robust`: aplica sbi.robust_aug (JPEG y desenfoque) a TODOS los ítems, reales y falsos (R3).
    La aleatoriedad sale de torch, que el DataLoader siembra por worker y por época a partir de la semilla.
    """
    def __init__(self, items, transform, fixed_seed=None, sbi_prob=0.5, robust=False, real_paths=None):
        self.items, self.transform = items, transform
        self.fixed_seed, self.sbi_prob, self.robust = fixed_seed, sbi_prob, robust
        self.real_paths = real_paths or [it[0] for it in items if it[1] == 0]

    def __len__(self):
        return len(self.items)

    def _rng(self, i):
        if self.fixed_seed is not None:
            return np.random.default_rng([self.fixed_seed, i])
        return np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))

    def __getitem__(self, i):
        path, label, vkey, *rest = self.items[i]
        kind = rest[0] if rest else "img"
        if kind == "img" and not self.robust:
            img = Image.open(path).convert("RGB")
        else:
            rng = self._rng(i)
            if kind == "mix" and rng.random() < self.sbi_prob:
                path, kind = self.real_paths[int(rng.integers(len(self.real_paths)))], "sbi"
            img = Image.open(path).convert("RGB")
            if kind == "sbi":
                img = _sbi().self_blend(img, rng)
            if self.robust:
                img = _sbi().robust_aug(img, rng)
        return self.transform(img), torch.tensor(label, dtype=torch.float32), vkey


def describe(items):
    n_real = sum(1 for it in items if it[1] == 0)
    vids = {it[2] for it in items}
    return {"frames": len(items), "frames_real": n_real, "frames_fake": len(items) - n_real, "videos": len(vids)}

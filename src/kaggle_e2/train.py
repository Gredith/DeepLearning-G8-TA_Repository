"""Entrenamiento y evaluación de Xception (BCE vs Focal Loss).

Uso típico (en Kaggle):
    from train import Config, run
    res = run(Config(gamma=0.0, run_name="bce_s42"))      # línea base
    res = run(Config(gamma=2.0, run_name="focal_g2_s42")) # intervención

Todo lo que no es gamma queda fijo entre corridas (arquitectura, datos, splits,
semilla, épocas, optimizador, augmentación): D-06 en docs/DECISIONS.md.
Las rutas por defecto son relativas a la raíz del repo (los notebooks hacen chdir a ROOT).
"""
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

import data as D
from losses import build_loss
from teardown import build_model


@dataclass
class Config:
    run_name: str = "bce_s42"
    gamma: float = 0.0               # 0 = BCE exacto
    alpha: float | None = None       # sin ponderación por clase (D-07)
    crops_dir: str = "/kaggle/working/crops"
    splits_dir: str = "results/splits"
    out_dir: str = "results/runs"
    ckpt_dir: str = "results/checkpoints"  # ignorado por Git
    train_methods: tuple = ("Deepfakes",)
    test_methods: tuple = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
    n_train_pairs: int | None = 100  # de 360 oficiales (presupuesto de GPU, D-05)
    n_val_pairs: int | None = 30     # de 70
    n_test_pairs: int | None = None  # test oficial completo (70)
    frames_per_video: int = 16
    img_size: int = 299
    batch_size: int = 32
    lr: float = 2e-4                 # FaceForensics++ (Rössler et al., 2019)
    weight_decay: float = 0.0
    warmup_epochs: int = 1           # solo la cabeza (como FF++, que entrena primero la capa final)
    epochs: int = 6                  # total, incluyendo warm-up
    seed: int = 42
    num_workers: int = 2
    amp: bool = True
    easy_threshold: float = 0.9      # p_t > 0.9 = ejemplo "fácil" (diagnóstico de la hipótesis)
    extra: dict = field(default_factory=dict)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def _loader(items, train, cfg):
    ds = D.FrameDataset(items, D.train_transform(cfg.img_size) if train else D.eval_transform(cfg.img_size))
    g = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=train, num_workers=cfg.num_workers,
                      pin_memory=torch.cuda.is_available(), generator=g, drop_last=False)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels, vkeys = [], [], []
    for x, y, v in loader:
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            p = torch.sigmoid(model(x.to(device)).float().view(-1))
        probs.append(p.cpu()); labels.append(y); vkeys += list(v)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy(), vkeys


def metrics(probs, labels, vkeys):
    out = {"n_frames": int(len(labels))}
    if len(set(labels)) == 2:
        out["auc_frame"] = float(roc_auc_score(labels, probs))
    out["acc_frame@0.5"] = float(accuracy_score(labels, probs > 0.5))
    # nivel video: promedio de probabilidades de sus frames
    vp, vl = {}, {}
    for p, l, v in zip(probs, labels, vkeys):
        vp.setdefault(v, []).append(p); vl[v] = l
    ks = sorted(vp)
    pv, lv = np.array([np.mean(vp[k]) for k in ks]), np.array([vl[k] for k in ks])
    out["n_videos"] = int(len(ks))
    if len(set(lv)) == 2:
        out["auc_video"] = float(roc_auc_score(lv, pv))
    out["acc_video@0.5"] = float(accuracy_score(lv, pv > 0.5))
    return out


def run(cfg: Config, device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    out_dir, ckpt_dir = Path(cfg.out_dir), Path(cfg.ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True); ckpt_dir.mkdir(parents=True, exist_ok=True)

    tr_pairs = D.subset_pairs(D.load_pairs(cfg.splits_dir, "train"), cfg.n_train_pairs, cfg.seed)
    va_pairs = D.subset_pairs(D.load_pairs(cfg.splits_dir, "val"), cfg.n_val_pairs, cfg.seed)
    te_pairs = D.subset_pairs(D.load_pairs(cfg.splits_dir, "test"), cfg.n_test_pairs, cfg.seed)
    tr_items, tr_miss = D.build_items(cfg.crops_dir, tr_pairs, cfg.train_methods, max_frames=cfg.frames_per_video)
    va_items, va_miss = D.build_items(cfg.crops_dir, va_pairs, cfg.train_methods, max_frames=cfg.frames_per_video)
    assert tr_items and va_items, f"No hay recortes en {cfg.crops_dir} para los splits pedidos"
    data_info = {"train": D.describe(tr_items) | {"videos_faltantes": tr_miss},
                 "val": D.describe(va_items) | {"videos_faltantes": va_miss}}
    if verbose:
        print(json.dumps(data_info, indent=1))
    tr_loader, va_loader = _loader(tr_items, True, cfg), _loader(va_items, False, cfg)

    model = build_model(pretrained=True, num_classes=1).to(device)
    criterion = build_loss(cfg.gamma, cfg.alpha)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device == "cuda")
    history, best_auc, best_path = [], -1.0, ckpt_dir / f"{cfg.run_name}.pt"

    def make_opt(params):
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(cfg.epochs):
        head_only = epoch < cfg.warmup_epochs
        for n, p in model.named_parameters():
            p.requires_grad = (not head_only) or n.startswith("fc.")
        if epoch == 0 or epoch == cfg.warmup_epochs:
            opt = make_opt([p for p in model.parameters() if p.requires_grad])
        model.train()
        t0, tot, n_seen, easy, pt_sum = time.time(), 0.0, 0, 0, 0.0
        for x, y, _ in tr_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=cfg.amp and device == "cuda"):
                logits = model(x).view(-1)
            loss = criterion(logits.float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            with torch.no_grad():
                p = torch.sigmoid(logits.float())
                p_t = torch.where(y > 0.5, p, 1 - p)
                easy += int((p_t > cfg.easy_threshold).sum()); pt_sum += float(p_t.sum())
            tot += loss.item() * len(y); n_seen += len(y)
        val = metrics(*predict(model, va_loader, device))
        row = {"epoch": epoch + 1, "fase": "cabeza" if head_only else "red completa",
               "train_loss": tot / n_seen, "train_mean_p_t": pt_sum / n_seen,
               "train_pct_faciles": 100 * easy / n_seen, "seg": time.time() - t0,
               **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        if verbose:
            print(f"[{cfg.run_name}] ep {row['epoch']} ({row['fase']}) loss={row['train_loss']:.4f} "
                  f"fáciles={row['train_pct_faciles']:.1f}% val_auc_frame={val.get('auc_frame', float('nan')):.4f} "
                  f"val_auc_video={val.get('auc_video', float('nan')):.4f} ({row['seg']:.0f}s)")
        score = val.get("auc_frame", -1)
        if score > best_auc:
            best_auc = score
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch + 1}, best_path)

    # Evaluación final con el mejor checkpoint: in-domain y cross-manipulation (Fase C)
    model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    test = {}
    for m in cfg.test_methods:
        items, miss = D.build_items(cfg.crops_dir, te_pairs, [m], max_frames=cfg.frames_per_video)
        if not any(y == 1 for _, y, _ in items):
            test[m] = {"nota": "sin videos de test descargados"}
            continue
        test[m] = metrics(*predict(model, _loader(items, False, cfg), device)) | {"videos_faltantes": miss}
        if verbose:
            print(f"  test {m:15s} AUC frame={test[m].get('auc_frame', float('nan')):.4f} "
                  f"video={test[m].get('auc_video', float('nan')):.4f}")
    result = {"config": asdict(cfg), "data": data_info, "history": history, "test": test,
              "best_val_auc_frame": best_auc, "device": device,
              "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}
    json.dump(result, open(out_dir / f"{cfg.run_name}.json", "w"), indent=1, default=str)
    return result

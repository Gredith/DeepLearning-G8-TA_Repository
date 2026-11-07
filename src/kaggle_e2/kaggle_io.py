"""Entrada/salida en Kaggle: empaquetar recortes, encontrarlos como input y guardar figuras privadas.

Por qué existe (ver D-12 en docs/DECISIONS.md):
- Kaggle no logra crear un Dataset desde un output con ~20 k PNG sueltos (la interfaz devuelve
  «Unexpected token '<' … is not valid JSON»). El notebook 02 entrega un único `crops.tar`.
- Kaggle sincroniza los notebooks con GitHub **con sus salidas**. Las figuras que muestran caras de
  FF++ no se muestran en la celda: se guardan en `privado/` (output privado de Kaggle) y se cierran.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

KAGGLE = Path("/kaggle").exists()
WORK = Path("/kaggle/working") if KAGGLE else Path.cwd() / "results" / "kaggle_local"
PRIVADO = WORK / "privado"          # figuras con caras y logs del script de descarga: nunca en GitHub
TAR_NAME = "crops.tar"


def pack_crops(crops_dir, out_dir=WORK, extra_files=()):
    """Empaqueta `crops/` en un único tar sin compresión (los PNG ya están comprimidos)."""
    crops_dir, out_dir = Path(crops_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / TAR_NAME
    with tarfile.open(tar_path, "w") as tar:
        tar.add(crops_dir, arcname="crops")
    for f in extra_files:
        f = Path(f)
        if f.exists() and f.parent != out_dir:
            (out_dir / f.name).write_bytes(f.read_bytes())
    return tar_path


def _count_pngs(root):
    return sum(1 for _ in Path(root).rglob("*.png"))


def find_crops(dst=Path("/tmp/crops_g8"), input_root=Path("/kaggle/input")):
    """Devuelve la carpeta `crops/` lista para usar.

    Orden: $CROPS_DIR → `crops.tar` en algún input (se extrae a `dst`) → carpeta `crops/` en algún input
    (por si Kaggle descomprimió el tar) → /kaggle/working/crops (misma sesión que el 02).
    """
    env = os.environ.get("CROPS_DIR")
    if env and Path(env).exists():
        return Path(env)
    if (dst / "crops").exists() and _count_pngs(dst / "crops"):
        return dst / "crops"
    if input_root.exists():
        tars = sorted(input_root.rglob(TAR_NAME))
        if tars:
            dst.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tars[0]) as tar:
                tar.extractall(dst, filter="data") if sys.version_info >= (3, 12) else tar.extractall(dst)
            return dst / "crops"
        dirs = [p for p in input_root.rglob("crops") if p.is_dir() and (p / "original").exists()]
        if dirs:
            return dirs[0]
    local = WORK / "crops"
    if local.exists():
        return local
    raise FileNotFoundError("No se encontraron recortes: agreguen el Dataset privado ffpp-crops-g8 como input "
                            "(Add Input) o definan CROPS_DIR.")


def find_manifest(crops_dir, input_root=Path("/kaggle/input")):
    cands = [Path(crops_dir) / "manifest.csv", Path(crops_dir).parent / "manifest.csv"]
    if input_root.exists():
        cands += sorted(input_root.rglob("manifest.csv"))
    return next((p for p in cands if p.exists()), None)


def check_counts(crops_dir, manifest_path):
    """Compara los PNG en disco con el manifiesto del 02. Devuelve DataFrame por categoría."""
    import pandas as pd
    man = pd.read_csv(manifest_path)
    esperado = man.groupby("categoria")["recortes"].sum()
    en_disco = pd.Series({c: _count_pngs(Path(crops_dir) / c) for c in esperado.index})
    tab = pd.DataFrame({"manifiesto": esperado, "en_disco": en_disco})
    tab["ok"] = tab.manifiesto == tab.en_disco
    return tab


def save_private(fig, name, dpi=150):
    """Guarda una figura con caras en `privado/` y la cierra SIN mostrarla (no queda en la salida)."""
    import matplotlib.pyplot as plt
    PRIVADO.mkdir(parents=True, exist_ok=True)
    path = PRIVADO / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("figura privada guardada (no se muestra por licencia FF++):", path.relative_to(WORK))
    return path


def snapshot_env(root, out_path):
    """Registra GPU, versiones, semilla y commit para la trazabilidad de la corrida."""
    import torch
    def _v(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    (out_path.parent / "pip_freeze.txt").write_text(freeze)
    info = {"commit": commit, "python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            **{m: _v(m) for m in ["timm", "torchvision", "numpy", "sklearn", "facenet_pytorch"]}}
    out_path.write_text(json.dumps(info, indent=1))
    return info


__all__ = ["KAGGLE", "WORK", "PRIVADO", "pack_crops", "find_crops", "find_manifest", "check_counts",
           "save_private", "snapshot_env"]

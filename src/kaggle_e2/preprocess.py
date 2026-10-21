"""Extracción de frames y recorte de caras de FaceForensics++.

Entrada (estructura del script oficial de descarga):
    <ffpp_root>/original_sequences/youtube/c23/videos/071.mp4
    <ffpp_root>/manipulated_sequences/<Metodo>/c23/videos/071_054.mp4
Salida:
    <crops_dir>/<categoria>/<video_id>/<idx_frame>.png   (299×299)

Recorte (D-03): cara más grande detectada con MTCNN (facenet-pytorch); si no está
instalada se usa el Haar cascade de OpenCV. La caja se agranda 1.3× y se hace
cuadrada, como en FaceForensics++. Si un frame no tiene cara, se descarta.
"""
from pathlib import Path

import cv2
import numpy as np

from data import METHODS, REAL

_DETECTOR = None


def video_path(ffpp_root, category, vid, comp="c23"):
    root = Path(ffpp_root)
    if category == REAL:
        return root / "original_sequences" / "youtube" / comp / "videos" / f"{vid}.mp4"
    return root / "manipulated_sequences" / category / comp / "videos" / f"{vid}.mp4"


def _detector(device="cpu"):
    global _DETECTOR
    if _DETECTOR is None:
        try:
            from facenet_pytorch import MTCNN
            _DETECTOR = ("mtcnn", MTCNN(keep_all=True, device=device, post_process=False))
        except ImportError:
            casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if casc.empty():
                raise RuntimeError("No hay detector de caras: instalar `pip install --no-deps facenet-pytorch`")
            _DETECTOR = ("haar", casc)
    return _DETECTOR


def detect_largest_face(rgb, device="cpu"):
    kind, det = _detector(device)
    if kind == "mtcnn":
        boxes, _ = det.detect(rgb)
        if boxes is None or len(boxes) == 0:
            return None
        boxes = [tuple(b) for b in boxes]
    else:
        faces = det.detectMultiScale(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        boxes = [(x, y, x + w, y + h) for x, y, w, h in faces]
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def square_crop(rgb, box, scale=1.3, size=299):
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * scale
    x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x2, y2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    crop = rgb[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA) if crop.size else None


def process_video(src, dst_dir, n_frames=16, size=299, device="cpu"):
    """Extrae n_frames equiespaciados y guarda los recortes. Devuelve # recortes guardados."""
    dst_dir = Path(dst_dir)
    if dst_dir.exists() and len(list(dst_dir.glob("*.png"))) >= n_frames:
        return len(list(dst_dir.glob("*.png")))
    cap = cv2.VideoCapture(str(src))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return 0
    idxs = np.linspace(0, total - 1, n_frames).astype(int)
    dst_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        box = detect_largest_face(rgb, device)
        if box is None:
            continue
        crop = square_crop(rgb, box, size=size)
        if crop is None:
            continue
        cv2.imwrite(str(dst_dir / f"{i:05d}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        saved += 1
    cap.release()
    return saved


def process_all(ffpp_root, crops_dir, video_ids_by_cat, n_frames=16, comp="c23", device="cpu"):
    """video_ids_by_cat: {categoria: [ids]}. Devuelve lista de dicts (manifiesto)."""
    from tqdm.auto import tqdm
    rows = []
    for cat, ids in video_ids_by_cat.items():
        for vid in tqdm(ids, desc=cat):
            src = video_path(ffpp_root, cat, vid, comp)
            n = process_video(src, Path(crops_dir) / cat / vid, n_frames, device=device) if src.exists() else 0
            rows.append({"categoria": cat, "video": vid, "existe_video": src.exists(), "recortes": n})
    return rows


__all__ = ["METHODS", "REAL", "process_all", "process_video", "video_path"]

"""Self-Blended Images (SBI) y aumentos de robustez para la Fase D (D-14).

SBI (Shiohara y Yamasaki, CVPR 2022): un falso sintético se fabrica con UN solo frame real.
Se copia el frame, se altera la copia (color, frecuencia y una pequeña desalineación) y se mezcla
con el original dentro de una máscara con forma de cara. El resultado tiene los artefactos que
comparten casi todas las falsificaciones (borde de mezcla, diferencias de color y de nitidez entre
la cara y su entorno), pero no la huella de ningún generador concreto.

Aproximación de este proyecto: los recortes no guardan landmarks, así que la máscara es una
elipse deformada. Funciona porque el recorte es cuadrado, centrado en la cara y ampliado 1.3×
(D-03): la cara ocupa ~1/1.3 del lado, la misma geometría que usan analysis.region_masks y la demo.

Toda la aleatoriedad entra por un `numpy.random.Generator`, así que una semilla fija da siempre
el mismo SBI (val determinista, tests).
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

FACE_FRAC = 1 / 1.3                      # lado de la caja de la cara dentro del recorte (D-03)
BLEND_RATIOS = (0.25, 0.5, 0.75, 1.0, 1.0, 1.0)   # como el artículo: la mitad de las veces, mezcla completa


# ------------------------------------------------------------------ máscara
def face_mask(size: int, rng: np.random.Generator) -> np.ndarray:
    """Máscara float32 (size×size, valores en [0, 1]) con forma de cara: elipse deformada y difuminada."""
    face = size * FACE_FRAC
    cx = size / 2 + rng.uniform(-0.03, 0.03) * size
    cy = size / 2 + rng.uniform(-0.01, 0.06) * size           # el centro de ojos-nariz-boca cae algo abajo de la caja
    ax = 0.5 * face * rng.uniform(0.62, 0.90)                  # la cara es más angosta que la caja de MTCNN
    ay = 0.5 * face * rng.uniform(0.80, 1.02)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) + 0.5
    dx, dy = (xx - cx) / ax, (yy - cy) / ay
    theta, r = np.arctan2(dy, dx), np.sqrt(dx ** 2 + dy ** 2)
    # deformación suave del contorno: armónicos bajos del ángulo (no hay landmarks que seguir)
    wobble = np.ones_like(theta)
    for k in range(2, 6):
        wobble += rng.uniform(-0.07, 0.07) * np.sin(k * theta + rng.uniform(0, 2 * np.pi))
    hard = (r <= wobble).astype(np.uint8) * 255
    sigma = rng.uniform(0.01, 0.05) * size                    # ancho de la transición del borde de mezcla
    soft = Image.fromarray(hard, "L").filter(ImageFilter.GaussianBlur(sigma))
    return np.asarray(soft, dtype=np.float32) / 255.0


# ------------------------------------------------------------------ transformación de la copia
def _color(img: Image.Image, rng) -> Image.Image:
    choice = rng.integers(3)
    if choice == 0:                                            # RGB shift
        a = np.asarray(img, dtype=np.int16) + rng.integers(-20, 21, size=3, dtype=np.int16)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if choice == 1:                                            # HSV shift
        h, s, v = (np.asarray(c, dtype=np.int16) for c in img.convert("HSV").split())
        h = (h + rng.integers(-8, 9)) % 256
        s = np.clip(s + rng.integers(-25, 26), 0, 255)
        v = np.clip(v + rng.integers(-20, 21), 0, 255)
        hsv = np.stack([h, s, v], -1).astype(np.uint8)
        return Image.fromarray(hsv, "HSV").convert("RGB")
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.9, 1.1))   # brillo y contraste
    return ImageEnhance.Contrast(img).enhance(rng.uniform(0.9, 1.1))


def _frequency(img: Image.Image, rng) -> Image.Image:
    if rng.random() < 0.5:                                     # bajar y volver a subir la resolución: menos detalle fino
        w, h = img.size
        f = rng.uniform(0.35, 0.7)
        small = img.resize((max(8, int(w * f)), max(8, int(h * f))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)
    return img.filter(ImageFilter.UnsharpMask(radius=rng.uniform(1, 3), percent=int(rng.uniform(60, 160)), threshold=0))


def _misalign(img: Image.Image, rng) -> Image.Image:
    """Escala ±4 % y desplazamiento ±3 %: imita una cara pegada que no calza del todo."""
    w, h = img.size
    s = rng.uniform(0.96, 1.04)
    tx, ty = rng.uniform(-0.03, 0.03) * w, rng.uniform(-0.03, 0.03) * h
    cx, cy = w / 2, h / 2
    # PIL usa la transformación inversa: salida(x, y) ← entrada(a·x + b·y + c, d·x + e·y + f)
    coeffs = (1 / s, 0, cx - (cx + tx) / s, 0, 1 / s, cy - (cy + ty) / s)
    moved = img.transform((w, h), Image.AFFINE, coeffs, resample=Image.BILINEAR)
    # los píxeles que quedan fuera de la imagen desplazada se rellenan con el original:
    # un marco negro sería un atajo trivial para distinguir SBI de reales
    valid = Image.new("L", (w, h), 255).transform((w, h), Image.AFFINE, coeffs, resample=Image.NEAREST)
    return Image.composite(moved, img, valid)


def source_transform(img: Image.Image, rng) -> Image.Image:
    out = _color(img, rng)
    if rng.random() < 0.7:
        out = _frequency(out, rng)
    return _misalign(out, rng)


# ------------------------------------------------------------------ SBI
def self_blend(img: Image.Image, rng: np.random.Generator, return_mask: bool = False):
    """Devuelve un SBI (PIL RGB del mismo tamaño) fabricado solo con `img`; etiqueta = falso."""
    img = img.convert("RGB")
    w, h = img.size
    assert w == h, "los recortes son cuadrados (D-03)"
    alt = source_transform(img, rng)
    # cuál va dentro de la máscara: la copia alterada (cara "pegada") o el original (entorno alterado)
    src, tgt = (alt, img) if rng.random() < 0.5 else (img, alt)
    m = face_mask(w, rng) * rng.choice(BLEND_RATIOS)
    a, b = np.asarray(src, dtype=np.float32), np.asarray(tgt, dtype=np.float32)
    out = Image.fromarray(np.clip(a * m[..., None] + b * (1 - m[..., None]), 0, 255).round().astype(np.uint8))
    return (out, m) if return_mask else out


# ------------------------------------------------------------------ robustez (R3) y estrés de compresión
def jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def robust_aug(img: Image.Image, rng, p_jpeg=0.5, q_range=(30, 95), p_blur=0.3, sigma_range=(0.2, 1.5)) -> Image.Image:
    """Compresión JPEG y desenfoque aleatorios. Se aplica igual a reales y falsos: no puede ser un atajo de clase."""
    if rng.random() < p_jpeg:
        img = jpeg(img, rng.integers(q_range[0], q_range[1] + 1))
    if rng.random() < p_blur:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(*sigma_range)))
    return img


__all__ = ["FACE_FRAC", "BLEND_RATIOS", "face_mask", "source_transform", "self_blend", "jpeg", "robust_aug"]

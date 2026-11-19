"""Demo en vivo: detector Xception con Grad-CAM, BCE vs. Focal Loss (γ = 2). Interfaz en español e inglés.

Sube una imagen; la app detecta la cara más grande con MTCNN, la recorta como en el
entrenamiento (caja cuadrada 1.3×, 299×299) y muestra para cada modelo la probabilidad de
«falso», la decisión (umbral 0.5), el Grad-CAM de `block12` y la masa del mapa por región
(interior / borde / fuera), la misma medición del notebook 06, con una lectura guiada por colores.

Uso local (pesos fuera de Git, en results/checkpoints/):
    python demo/app.py                # http://127.0.0.1:7860
    python demo/app.py --share        # enlace temporal de Gradio (respaldo en Kaggle)

Pesos: `bce_s42.pt` y `focal_g2_s42.pt`, del output del notebook 04 (o del 03). Se buscan en
--ckpt-dir, en $DEMO_CKPT_DIR, en results/checkpoints/ y en /kaggle/input.

La app no guarda imágenes ni resultados. Gradio guarda temporalmente los archivos subidos en
su caché; aquí se borra cada 10 minutos y al cerrar la app.
"""
from __future__ import annotations

import argparse
import html
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("MPLBACKEND", "Agg")  # gradcam.py importa pyplot; sin ventanas
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "kaggle_e2"))

import analysis as A  # noqa: E402
import data as D  # noqa: E402
import gradcam as G  # noqa: E402
import preprocess as P  # noqa: E402
import teardown as TD  # noqa: E402

MODELOS = {"bce": ("BCE (γ = 0)", "bce_s42"), "focal": ("Focal (γ = 2)", "focal_g2_s42")}
LAYER, SIZE, UMBRAL = "block12", 299, 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Umbrales de lectura, tomados de los rangos medidos en el notebook 06 (2 frames por video de test):
# respuesta «falso» → masa interior 0.67–0.95 y área 50 % 0.04–0.07; respuesta «real» → masa fuera 0.31–0.62.
INTERIOR_MIN, FUERA_MIN = 0.60, 0.30

# --------------------------------------------------------------------------------------------- textos
T = {
    "es": {
        "lang_label": "Idioma / Language",
        "title": "# ¿Qué mira el detector de deepfakes?",
        "intro": ("Xception entrenada **solo con Deepfakes** de FaceForensics++ (c23, semilla 42). "
                  "Se comparan dos funciones de pérdida: **BCE** y **Focal Loss (γ = 2)**. "
                  "Las imágenes que subes no se guardan.\n\n"
                  "**1.** Sube una foto con una cara · **2.** Pulsa *Analizar* · "
                  "**3.** Compara la decisión y el mapa de calor de cada modelo."),
        "input": "Imagen con una cara",
        "analyze": "Analizar",
        "clear": "Limpiar",
        "options": "Opciones",
        "precrop": "La imagen ya es un recorte de la cara (no detectar)",
        "guides": "Dibujar los límites de las regiones",
        "crop": "Recorte que ve el modelo",
        "cam_bce": "Grad-CAM · BCE (γ = 0)",
        "cam_focal": "Grad-CAM · Focal (γ = 2)",
        "empty": "Los resultados aparecerán aquí después de pulsar <b>Analizar</b>.",
        "no_image": "Primero sube una imagen.",
        "msg_crop": "Cara detectada con MTCNN y recortada como en el entrenamiento (caja cuadrada 1.3×).",
        "msg_precrop": "Se usó la imagen completa como recorte de cara.",
        "msg_noface": "No se detectó ninguna cara. Prueba con otra imagen o marca «La imagen ya es un recorte».",
        "msg_out": "La cara detectada queda fuera de la imagen.",
        "fake": "FALSO", "real": "REAL",
        "pfake": "probabilidad de falso",
        "conf": {"alta": "confianza alta", "media": "confianza media", "baja": "cerca del umbral: poco fiable"},
        "agree": "Ambos modelos coinciden",
        "disagree": "Los modelos no coinciden: el caso está cerca del límite de lo que aprendieron",
        "where": "¿Dónde se concentra el mapa?",
        "reg": {"interior": "interior", "borde": "borde", "fuera": "fuera"},
        "area50": "El 50 % del mapa ocupa el {a:.0f} % del recorte",
        "pat": {"interior": "Evidencia concentrada en el interior de la cara",
                "fuera": "Atención dispersa, sobre todo fuera de la cara",
                "mixto": "Patrón mixto"},
        "coh_ok": "Coincide con lo medido en el test: {txt}",
        "coh_no": "No coincide con el patrón típico del test: interpretar con cautela",
        "coh_fake": "cuando el modelo dice «falso», mira el interior de la cara.",
        "coh_real": "cuando el modelo dice «real», no encuentra esa evidencia y mira fuera de la cara.",
        "find_title": "Qué encontró el proyecto y límites de la demo",
        "find": """
<div class="hallazgos">
  <div class="hz-card hz-find">
    <div class="hz-head">🔍 Qué se esperaba y qué se encontró</div>
    <p><b>Hipótesis:</b> el modelo miraría el <i>borde de la mezcla</i>, donde se pega la cara falsa.</p>
    <p><b>Resultado:</b> no es así. Aprendió una evidencia del <b>interior</b> de la cara propia de Deepfakes.</p>
    <ul>
      <li>Por eso los <b>FaceSwap</b> se escapan: AUC por debajo de 0.5 en las 12 corridas.</li>
      <li><b>Focal Loss</b> cambia poco el mapa, igual que en los números: solo mejora NeuralTextures (+0.04, IC 95 % sin el 0).</li>
    </ul>
  </div>
  <div class="hz-card hz-lim">
    <div class="hz-head">⚠ Límites</div>
    <ul>
      <li>El modelo vio solo videos de FF++ <b>comprimidos (c23)</b>: una foto de cámara o de celular está fuera de esa distribución y puede clasificarse mal.</li>
      <li><b>Un resultado aislado no prueba nada:</b> las conclusiones del proyecto salen de 12 corridas (4 valores de γ × 3 semillas) con intervalos de confianza.</li>
    </ul>
  </div>
</div>""",
        "guide_title": "Cómo interpretar los resultados",
        "guide": """
<div class="guia">
<p><b>Decisión.</b> <span class="chip fake">FALSO</span> si la probabilidad de falso es ≥ 0.5;
<span class="chip real">REAL</span> si es menor. La confianza indica qué tan lejos está del umbral:
<i>alta</i> (&lt; 0.1 o &gt; 0.9), <i>media</i> o <i>cerca del umbral</i> (0.3–0.7).</p>
<p><b>Mapa de calor (Grad-CAM, capa <code>block12</code>).</b> Resalta las zonas que empujan la decisión hacia
«falso»: <span class="sw" style="background:#d7191c"></span> rojo = mucha influencia,
<span class="sw" style="background:#2c7bb6"></span> azul = poca. Muestra <i>dónde</i> mira el modelo, no <i>por qué</i>.</p>
<p><b>Regiones del recorte</b> (recuadros blancos; aproximación geométrica, no una segmentación):
<span class="sw" style="background:var(--c-int)"></span> <b>interior</b>: ojos, nariz y boca ·
<span class="sw" style="background:var(--c-bor)"></span> <b>borde</b>: contorno de la cara ·
<span class="sw" style="background:var(--c-out)"></span> <b>fuera</b>: pelo, cuello y fondo.</p>
<p><b>Patrón del mapa.</b>
<span class="chip int">interior concentrado</span> masa interior ≥ 0.60 ·
<span class="chip out">disperso fuera</span> masa fuera ≥ 0.30 ·
<span class="chip mix">mixto</span> el resto. En el test de FF++ (notebook 06), cuando el modelo responde «falso»
la masa interior es 0.67–0.95 y muy concentrada; cuando responde «real», la masa fuera es 0.31–0.62.</p>
</div>""",
    },
    "en": {
        "lang_label": "Idioma / Language",
        "title": "# What does the deepfake detector look at?",
        "intro": ("Xception trained **only on Deepfakes** from FaceForensics++ (c23, seed 42). "
                  "Two loss functions are compared: **BCE** and **Focal Loss (γ = 2)**. "
                  "Uploaded images are not stored.\n\n"
                  "**1.** Upload a photo with a face · **2.** Press *Analyze* · "
                  "**3.** Compare each model's decision and heat map."),
        "input": "Image with a face",
        "analyze": "Analyze",
        "clear": "Clear",
        "options": "Options",
        "precrop": "The image is already a face crop (skip detection)",
        "guides": "Draw the region boundaries",
        "crop": "Crop the model sees",
        "cam_bce": "Grad-CAM · BCE (γ = 0)",
        "cam_focal": "Grad-CAM · Focal (γ = 2)",
        "empty": "Results will appear here after pressing <b>Analyze</b>.",
        "no_image": "Upload an image first.",
        "msg_crop": "Face detected with MTCNN and cropped as in training (1.3× square box).",
        "msg_precrop": "The whole image was used as the face crop.",
        "msg_noface": "No face detected. Try another image or tick «The image is already a face crop».",
        "msg_out": "The detected face falls outside the image.",
        "fake": "FAKE", "real": "REAL",
        "pfake": "probability of fake",
        "conf": {"alta": "high confidence", "media": "medium confidence", "baja": "near the threshold: unreliable"},
        "agree": "Both models agree",
        "disagree": "The models disagree: the case is near the edge of what they learned",
        "where": "Where is the map concentrated?",
        "reg": {"interior": "interior", "borde": "border", "fuera": "outside"},
        "area50": "50% of the map covers {a:.0f}% of the crop",
        "pat": {"interior": "Evidence concentrated on the face interior",
                "fuera": "Diffuse attention, mostly outside the face",
                "mixto": "Mixed pattern"},
        "coh_ok": "Matches what was measured on the test set: {txt}",
        "coh_no": "Does not match the typical test-set pattern: interpret with caution",
        "coh_fake": "when the model says “fake”, it looks at the face interior.",
        "coh_real": "when the model says “real”, it finds no such evidence and looks outside the face.",
        "find_title": "What the project found and the demo's limits",
        "find": """
<div class="hallazgos">
  <div class="hz-card hz-find">
    <div class="hz-head">🔍 What we expected and what we found</div>
    <p><b>Hypothesis:</b> the model would look at the <i>blending border</i>, where the fake face is pasted.</p>
    <p><b>Result:</b> it does not. It learned <b>interior</b> face evidence specific to Deepfakes.</p>
    <ul>
      <li>That is why <b>FaceSwap</b> fakes slip through: AUC below 0.5 in all 12 runs.</li>
      <li><b>Focal loss</b> barely changes the map, just as in the numbers: it only improves NeuralTextures (+0.04, 95% CI excludes 0).</li>
    </ul>
  </div>
  <div class="hz-card hz-lim">
    <div class="hz-head">⚠ Limits</div>
    <ul>
      <li>The model only saw <b>compressed (c23)</b> FF++ videos: a camera or phone photo is outside that distribution and may be misclassified.</li>
      <li><b>A single result proves nothing:</b> the project's conclusions come from 12 runs (4 γ values × 3 seeds) with confidence intervals.</li>
    </ul>
  </div>
</div>""",
        "guide_title": "How to read the results",
        "guide": """
<div class="guia">
<p><b>Decision.</b> <span class="chip fake">FAKE</span> if the probability of fake is ≥ 0.5;
<span class="chip real">REAL</span> otherwise. Confidence says how far it is from the threshold:
<i>high</i> (&lt; 0.1 or &gt; 0.9), <i>medium</i>, or <i>near the threshold</i> (0.3–0.7).</p>
<p><b>Heat map (Grad-CAM, layer <code>block12</code>).</b> Highlights the areas that push the decision towards
“fake”: <span class="sw" style="background:#d7191c"></span> red = strong influence,
<span class="sw" style="background:#2c7bb6"></span> blue = weak. It shows <i>where</i> the model looks, not <i>why</i>.</p>
<p><b>Crop regions</b> (white boxes; a geometric approximation, not a segmentation):
<span class="sw" style="background:var(--c-int)"></span> <b>interior</b>: eyes, nose and mouth ·
<span class="sw" style="background:var(--c-bor)"></span> <b>border</b>: face contour ·
<span class="sw" style="background:var(--c-out)"></span> <b>outside</b>: hair, neck and background.</p>
<p><b>Map pattern.</b>
<span class="chip int">concentrated interior</span> interior mass ≥ 0.60 ·
<span class="chip out">diffuse outside</span> outside mass ≥ 0.30 ·
<span class="chip mix">mixed</span> otherwise. On the FF++ test set (notebook 06), when the model answers “fake”
the interior mass is 0.67–0.95 and highly concentrated; when it answers “real”, the outside mass is 0.31–0.62.</p>
</div>""",
    },
}
IDIOMAS = {"Español": "es", "English": "en"}

CSS = """
:root { --c-fake:#c0392b; --c-real:#1e8449; --c-int:#e8833a; --c-bor:#8e6bc4; --c-out:#3c7dd9; --c-mix:#7f8c8d; }
.chip { display:inline-block; padding:1px 9px; border-radius:999px; font-size:0.85em; font-weight:700; color:#fff; }
.chip.fake { background:var(--c-fake); } .chip.real { background:var(--c-real); }
.chip.int { background:var(--c-int); } .chip.out { background:var(--c-out); } .chip.mix { background:var(--c-mix); }
.sw { display:inline-block; width:0.9em; height:0.9em; border-radius:3px; vertical-align:-0.1em; }
.res-vacio { opacity:0.7; padding:12px 2px; }
.res-msg { font-size:0.9em; opacity:0.85; margin:2px 0 10px; }
.res-acuerdo { padding:8px 12px; border-radius:8px; margin-bottom:10px; font-weight:600;
               background:rgba(127,140,141,0.15); border-left:4px solid var(--c-mix); }
.res-cards { display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:12px; }
.res-card { border:1px solid rgba(127,127,127,0.35); border-radius:10px; padding:12px 14px; border-top:5px solid var(--c-mix); }
.res-card.fake { border-top-color:var(--c-fake); } .res-card.real { border-top-color:var(--c-real); }
.res-card h4 { margin:0 0 6px; font-size:1em; }
.res-dec { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }
.res-dec .chip { font-size:1.05em; padding:3px 12px; }
.res-p { font-size:1.6em; font-weight:700; font-variant-numeric:tabular-nums; }
.res-conf { font-size:0.85em; opacity:0.8; }
.res-sub { margin:10px 0 4px; font-size:0.85em; font-weight:600; opacity:0.85; }
.res-bar { display:flex; height:22px; border-radius:6px; overflow:hidden; }
.res-bar div { color:#fff; font-size:0.75em; font-weight:700; display:flex; align-items:center; justify-content:center;
               white-space:nowrap; overflow:hidden; }
.res-leg { display:flex; gap:12px; flex-wrap:wrap; font-size:0.8em; margin-top:4px; opacity:0.85; }
.res-pat { margin-top:8px; }
.res-coh { margin-top:6px; font-size:0.85em; }
.res-coh.ok::before { content:"✓ "; color:var(--c-real); font-weight:700; }
.res-coh.no::before { content:"⚠ "; color:var(--c-int); font-weight:700; }
.guia p { margin:0 0 8px; line-height:1.45; }
.hallazgos { display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:12px; }
.hz-card { border-radius:10px; padding:12px 16px; line-height:1.45; }
.hz-card p { margin:0 0 6px; } .hz-card ul { margin:4px 0 0; padding-left:1.2em; } .hz-card li { margin-bottom:4px; }
.hz-head { font-weight:700; margin-bottom:8px; }
.hz-find { background:rgba(60,125,217,0.10); border:1px solid rgba(60,125,217,0.35); }
.hz-lim { background:rgba(232,131,58,0.10); border:1px solid rgba(232,131,58,0.40); }
.cap-wrap { padding:0 !important; min-height:0 !important; margin-bottom:calc(-1 * var(--layout-gap, 12px) + 4px) !important; }
.cap-wrap > div { min-height:0 !important; }
.cap { font-size:0.85em; font-weight:600; opacity:0.85; margin:0; line-height:1.3; }
.cap3 { display:grid; grid-template-columns:repeat(3, 1fr); gap:var(--layout-gap, 12px); }

/* barra superior: enlaces de Gradio (API, Gradio, Configuración) a la izquierda, idioma a la derecha */
#barra { display:grid !important; grid-template-columns:minmax(0, 1fr) auto; align-items:center; gap:12px; padding:0 0 6px;
         border-bottom:1px solid rgba(127,127,127,0.25); margin-bottom:4px; }
#barra > div { min-width:0 !important; }
#enlaces-wrap, #enlaces-wrap > div, #enlaces-top { width:100% !important; min-width:0 !important; }
#enlaces-top footer { margin:0 !important; padding:0 !important; justify-content:flex-start !important;
                      align-items:center; flex-wrap:wrap; gap:2px; font-size:0.9em; border:none !important; }
#enlaces-top footer > button, #enlaces-top footer > a {
  background:none !important; border:none !important; box-shadow:none !important; min-width:0 !important;
  padding:3px 8px !important; margin:0 !important; border-radius:6px !important; font-size:inherit !important;
  font-weight:500 !important; color:var(--body-text-color) !important; opacity:0.8; text-decoration:none !important;
  align-items:center; gap:5px; width:auto !important; white-space:nowrap; }
#enlaces-top footer > button:hover, #enlaces-top footer > a:hover { background:rgba(127,127,127,0.15) !important;
  opacity:1; }
#enlaces-top footer img { width:13px !important; height:13px !important; margin:0 !important; }
#idioma { width:auto !important; justify-self:end; flex:none !important; margin:0 !important; }
#idioma .wrap { display:inline-flex; gap:0; padding:3px; border-radius:999px; flex-wrap:nowrap;
                background:var(--block-background-fill); border:1px solid var(--border-color-primary); }
#idioma .wrap::before { content:"🌐"; padding:0 6px 0 8px; align-self:center; }
#idioma label { margin:0 !important; padding:4px 14px !important; border-radius:999px !important; border:none !important;
                background:transparent !important; box-shadow:none !important; cursor:pointer; font-weight:600;
                transition:background-color .15s ease, color .15s ease; }
#idioma label input { display:none; }
#idioma label.selected { background:var(--color-accent) !important; color:#fff !important; }
#idioma label:not(.selected):hover { background:rgba(127,127,127,0.15) !important; }
"""

# Mueve el pie de Gradio (API, «Construido con Gradio», Configuración) a la barra superior. Se mueve el
# mismo elemento, así que sus botones conservan su comportamiento; si no se encuentra, el pie queda abajo.
JS = """
() => {
  const mover = () => {
    const f = document.querySelector('footer'), s = document.getElementById('enlaces-top');
    if (f && s && !s.contains(f)) { s.appendChild(f); return true; }
    return !!(f && s);
  };
  if (!mover()) {
    const o = new MutationObserver(() => { if (mover()) o.disconnect(); });
    o.observe(document.body, { childList: true, subtree: true });
  }
}
"""


# --------------------------------------------------------------------------------------------- modelo
def buscar_ckpt(nombre: str, ckpt_dir: str | None) -> Path:
    dirs = [ckpt_dir, os.environ.get("DEMO_CKPT_DIR"), ROOT / "results" / "checkpoints", "/kaggle/input"]
    for d in dirs:
        if d and Path(d).exists():
            hit = sorted(Path(d).rglob(f"{nombre}.pt"))
            if hit:
                return hit[0]
    raise FileNotFoundError(
        f"No se encontró {nombre}.pt. Copia los pesos del output del notebook 04 a results/checkpoints/ "
        "o indica la carpeta con --ckpt-dir.")


def cargar_modelos(ckpt_dir: str | None = None) -> dict:
    out = {}
    for clave, (_, nombre) in MODELOS.items():
        m = TD.build_model(False, 1)
        estado = torch.load(buscar_ckpt(nombre, ckpt_dir), map_location="cpu", weights_only=False)
        m.load_state_dict(estado["model"])
        out[clave] = m.to(DEVICE).eval()
    return out


def recortar(rgb: np.ndarray, ya_recortada: bool):
    """Devuelve (recorte 299×299 RGB o None, clave del mensaje). Mismo recorte que el preprocesamiento (D-03)."""
    if ya_recortada:
        return cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA), "msg_precrop"
    caja = P.detect_largest_face(rgb, device=DEVICE)
    if caja is None:
        return None, "msg_noface"
    crop = P.square_crop(rgb, caja, scale=1.3, size=SIZE)
    return (crop, "msg_crop") if crop is not None else (None, "msg_out")


def superponer(crop: np.ndarray, cam: np.ndarray, guias: bool) -> np.ndarray:
    color = (G.plt.get_cmap("jet")(cam)[..., :3] * 255).astype(np.float32)
    img = (0.6 * crop.astype(np.float32) + 0.4 * color).clip(0, 255).astype(np.uint8)
    if guias:  # límites de analysis.region_masks: interior (lado 0.5) y cara (lado 1/1.3)
        for lado in (0.5, 1 / 1.3):
            a, b = int(SIZE * (1 - lado) / 2), int(SIZE * (1 + lado) / 2)
            cv2.rectangle(img, (a, a), (b, b), (255, 255, 255), 1, lineType=cv2.LINE_AA)
    return img


def procesar(modelos: dict, imagen, ya_recortada: bool = False, guias: bool = True) -> dict:
    """Corre detección + modelos. Devuelve un dict sin texto (el texto depende del idioma)."""
    if imagen is None:
        return {"msg": "no_image"}
    rgb = np.asarray(Image.fromarray(imagen).convert("RGB"))
    crop, msg = recortar(rgb, ya_recortada)
    if crop is None:
        return {"msg": msg}
    x = D.eval_transform(SIZE)(Image.fromarray(crop)).unsqueeze(0).to(DEVICE)
    res = {"msg": msg, "crop": crop, "modelos": {}}
    for clave, m in modelos.items():
        cams, probs = G.gradcam(m, x, LAYER)
        st = A.cam_region_stats(cams).iloc[0]
        res["modelos"][clave] = {"p": float(probs[0]), "overlay": superponer(crop, cams[0], guias),
                                 "interior": float(st.masa_interior), "borde": float(st.masa_borde),
                                 "fuera": float(st.masa_fuera), "area50": float(st.area_50)}
    return res


# --------------------------------------------------------------------------------------------- lectura
def confianza(p: float) -> str:
    d = abs(p - UMBRAL)
    return "alta" if d >= 0.4 else "media" if d >= 0.2 else "baja"


def patron(r: dict) -> str:
    if r["interior"] >= INTERIOR_MIN:
        return "interior"
    if r["fuera"] >= FUERA_MIN:
        return "fuera"
    return "mixto"


def coherente(r: dict) -> bool:
    """¿La decisión coincide con el patrón del notebook 06? falso ↔ interior, real ↔ fuera."""
    return (r["p"] >= UMBRAL and patron(r) == "interior") or (r["p"] < UMBRAL and patron(r) == "fuera")


def tarjeta(clave: str, r: dict, t: dict) -> str:
    falso = r["p"] >= UMBRAL
    cls = "fake" if falso else "real"
    seg = "".join(
        f'<div style="width:{100 * r[k]:.1f}%;background:var({v})" title="{t["reg"][k]} {r[k]:.2f}">'
        f'{t["reg"][k] + " " if r[k] >= 0.22 else ""}{r[k]:.2f}</div>' if r[k] >= 0.08 else
        f'<div style="width:{100 * r[k]:.1f}%;background:var({v})" title="{t["reg"][k]} {r[k]:.2f}"></div>'
        for k, v in (("interior", "--c-int"), ("borde", "--c-bor"), ("fuera", "--c-out")))
    leg = "".join(f'<span><span class="sw" style="background:var({v})"></span> {t["reg"][k]} {r[k]:.2f}</span>'
                  for k, v in (("interior", "--c-int"), ("borde", "--c-bor"), ("fuera", "--c-out")))
    pat = patron(r)
    chip_pat = {"interior": "int", "fuera": "out", "mixto": "mix"}[pat]
    coh = (f'<div class="res-coh ok">{t["coh_ok"].format(txt=t["coh_fake"] if falso else t["coh_real"])}</div>'
           if coherente(r) else f'<div class="res-coh no">{t["coh_no"]}</div>')
    return f"""
<div class="res-card {cls}">
  <h4>{html.escape(MODELOS[clave][0])}</h4>
  <div class="res-dec"><span class="chip {cls}">{t["fake"] if falso else t["real"]}</span>
    <span class="res-p">{r["p"]:.3f}</span><span class="res-conf">{t["pfake"]} · {t["conf"][confianza(r["p"])]}</span></div>
  <div class="res-sub">{t["where"]}</div>
  <div class="res-bar">{seg}</div>
  <div class="res-leg">{leg}<span>· {t["area50"].format(a=100 * r["area50"])}</span></div>
  <div class="res-pat"><span class="chip {chip_pat}">{t["pat"][pat]}</span></div>
  {coh}
</div>"""


def render(res: dict | None, lang: str) -> str:
    t = T[lang]
    if not res:
        return f'<div class="res-vacio">{t["empty"]}</div>'
    if "modelos" not in res:
        return f'<div class="res-acuerdo" style="border-left-color:var(--c-int)">{t[res["msg"]]}</div>'
    ms = res["modelos"]
    dec = {k: r["p"] >= UMBRAL for k, r in ms.items()}
    if len(set(dec.values())) == 1:
        falso = next(iter(dec.values()))
        acuerdo = (f'<div class="res-acuerdo" style="border-left-color:var(--c-{"fake" if falso else "real"})">'
                   f'{t["agree"]}: <span class="chip {"fake" if falso else "real"}">{t["fake"] if falso else t["real"]}</span></div>')
    else:
        acuerdo = f'<div class="res-acuerdo" style="border-left-color:var(--c-int)">{t["disagree"]}</div>'
    return (f'<div class="res-msg">{t[res["msg"]]}</div>{acuerdo}'
            f'<div class="res-cards">{"".join(tarjeta(k, r, t) for k, r in ms.items())}</div>')


def analizar(modelos: dict, imagen, ya_recortada: bool = False, guias: bool = True, lang: str = "es"):
    """Atajo para pruebas sin interfaz: (recorte, overlay BCE, overlay Focal, html)."""
    res = procesar(modelos, imagen, ya_recortada, guias)
    ms = res.get("modelos", {})
    return res.get("crop"), ms.get("bce", {}).get("overlay"), ms.get("focal", {}).get("overlay"), render(res, lang)


# --------------------------------------------------------------------------------------------- interfaz
def cabecera_entrada(t: dict) -> str:
    return f'<div class="cap">{t["input"]}</div>'


def cabecera_salidas(t: dict) -> str:
    return f'<div class="cap cap3"><span>{t["crop"]}</span><span>{t["cam_bce"]}</span><span>{t["cam_focal"]}</span></div>'


def construir_ui(modelos: dict):
    import gradio as gr

    t = T["es"]
    with gr.Blocks(title="Deepfake detector · Grad-CAM BCE vs. Focal", delete_cache=(600, 600)) as ui:
        estado = gr.State(None)
        with gr.Row(equal_height=False, elem_id="barra"):
            gr.HTML('<div id="enlaces-top"></div>', padding=False, container=False, elem_id="enlaces-wrap")
            idioma = gr.Radio(list(IDIOMAS), value="Español", label=t["lang_label"], show_label=False,
                              container=False, elem_id="idioma", scale=0, min_width=0)
        titulo = gr.Markdown(t["title"], elem_id="titulo")
        intro = gr.Markdown(t["intro"], elem_id="intro")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=240):
                cap_in = gr.HTML(cabecera_entrada(t), elem_classes="cap-wrap")
                entrada = gr.Image(label=t["input"], show_label=False, type="numpy",
                                   sources=["upload", "clipboard", "webcam"], height=230)
                with gr.Row():
                    boton = gr.Button(t["analyze"], variant="primary", scale=2, min_width=80)
                    limpiar = gr.Button(t["clear"], variant="secondary", scale=1, min_width=80)
                with gr.Accordion(t["options"], open=False) as opciones:
                    ya = gr.Checkbox(label=t["precrop"], value=False)
                    guias = gr.Checkbox(label=t["guides"], value=True)
            with gr.Column(scale=3, min_width=480):
                cap_out = gr.HTML(cabecera_salidas(t), elem_classes="cap-wrap")
                with gr.Row():  # sin etiqueta propia: cambiarla recarga la imagen y parpadea al cambiar de idioma
                    o_crop = gr.Image(label=t["crop"], show_label=False, interactive=False, height=230)
                    o_bce = gr.Image(label=t["cam_bce"], show_label=False, interactive=False, height=230)
                    o_focal = gr.Image(label=t["cam_focal"], show_label=False, interactive=False, height=230)
                resultado = gr.HTML(render(None, "es"), elem_id="resultado")
        with gr.Accordion(t["guide_title"], open=True) as guia_acc:
            guia = gr.HTML(t["guide"], elem_id="guia")
        with gr.Accordion(t["find_title"], open=True) as hall_acc:
            hallazgos = gr.HTML(t["find"], elem_id="hallazgos")

        def correr(im, y, g, lang_nombre):
            res = procesar(modelos, im, y, g)
            ms = res.get("modelos", {})
            return (res.get("crop"), ms.get("bce", {}).get("overlay"), ms.get("focal", {}).get("overlay"),
                    {"msg": res["msg"], "modelos": {k: {kk: vv for kk, vv in r.items() if kk != "overlay"}
                                                    for k, r in ms.items()}} if ms else {"msg": res["msg"]},
                    render(res, IDIOMAS[lang_nombre]))

        def cambiar_idioma(lang_nombre, res):
            tt = T[IDIOMAS[lang_nombre]]
            # Solo se actualizan textos; las imágenes no se tocan.
            return (tt["title"], tt["intro"], cabecera_entrada(tt), cabecera_salidas(tt),
                    gr.update(value=tt["analyze"]), gr.update(value=tt["clear"]), gr.update(label=tt["options"]),
                    gr.update(label=tt["precrop"]), gr.update(label=tt["guides"]),
                    render(res, IDIOMAS[lang_nombre]), gr.update(label=tt["guide_title"]), tt["guide"],
                    gr.update(label=tt["find_title"]), tt["find"])

        def borrar(lang_nombre):
            return None, None, None, None, None, render(None, IDIOMAS[lang_nombre])

        boton.click(correr, [entrada, ya, guias, idioma], [o_crop, o_bce, o_focal, estado, resultado],
                    api_name="analizar")
        limpiar.click(borrar, [idioma], [entrada, o_crop, o_bce, o_focal, estado, resultado], api_name=False)
        idioma.change(cambiar_idioma, [idioma, estado],
                      [titulo, intro, cap_in, cap_out, boton, limpiar, opciones, ya, guias,
                       resultado, guia_acc, guia, hall_acc, hallazgos], api_name=False,
                      show_progress="hidden", queue=False)  # sin la animación de «cargando» en cada texto
    return ui


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt-dir", help="carpeta con bce_s42.pt y focal_g2_s42.pt")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="enlace público temporal de Gradio (p. ej. desde Kaggle)")
    args = ap.parse_args()
    modelos = cargar_modelos(args.ckpt_dir)
    print(f"Modelos cargados en {DEVICE}: {', '.join(v[0] for v in MODELOS.values())}")
    construir_ui(modelos).launch(server_name="127.0.0.1", server_port=args.port, share=args.share, css=CSS, js=JS)


if __name__ == "__main__":
    main()

"""Demo en vivo: detector Xception con Grad-CAM, modelos de la Fase B y de la Fase D. Interfaz en español e inglés.

Sube una imagen; la app detecta la cara más grande con MTCNN, la recorta como en el
entrenamiento (caja cuadrada 1.3×, 299×299) y pasa el recorte por los cuatro modelos de la
semilla 42. Muestra dos a la vez según la comparación elegida (Fase B: BCE vs. Focal; Fase D:
R1 vs. R3; B vs. D: BCE vs. R3), con la probabilidad de «falso», la decisión (umbral 0.5), el
Grad-CAM de `block12` y la masa del mapa por región (interior / borde / fuera), la misma
medición de los notebooks 06 y 08. Cambiar de comparación no vuelve a correr los modelos.

Uso local (pesos fuera de Git, en results/checkpoints/):
    python demo/app.py                # http://127.0.0.1:7860
    python demo/app.py --share        # enlace temporal de Gradio (respaldo en Kaggle)

Pesos: `bce_s42.pt` y `focal_g2_s42.pt` (output del notebook 04) y `r1_sbi_bce_s42.pt` y
`r3_sbi_focal_g2_aug_s42.pt` (output del notebook 07). Se buscan en --ckpt-dir, en
$DEMO_CKPT_DIR, en results/checkpoints/ y en /kaggle/input. Si faltan los de una fase, la
app arranca sin las comparaciones que los necesitan.

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

# clave: (nombre corto, fase, checkpoint, % detectado con umbral 0.5 en el test de FF++ (semilla 42):
#         Deepfakes, Face2Face, FaceSwap, NeuralTextures y % de reales que quedan como reales).
# preds_*.csv.gz de los notebooks 04 (Fase B) y 07 (Fase D), todos los frames de test.
MODELOS = {
    "bce":   ("BCE (γ = 0)", "B", "bce_s42", (97, 12, 1, 21, 95)),
    "focal": ("Focal (γ = 2)", "B", "focal_g2_s42", (97, 7, 0, 17, 96)),
    "r1":    ("R1 · SBI + BCE", "D", "r1_sbi_bce_s42", (96, 17, 3, 19, 97)),
    "r3":    ("R3 · SBI + Focal + JPEG", "D", "r3_sbi_focal_g2_aug_s42", (99, 58, 25, 51, 87)),
}
COMPARACIONES = {"B": ("bce", "focal"), "D": ("r1", "r3"), "BD": ("bce", "r3")}
LAYER, SIZE, UMBRAL = "block12", 299, 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Umbrales de lectura del mapa, de los rangos medidos en los notebooks 06 y 08: cuando el modelo
# detecta, la masa interior es ≥ 0.67; cuando no, la masa fuera de la cara es ≥ 0.31.
INTERIOR_MIN, FUERA_MIN = 0.60, 0.30

# --------------------------------------------------------------------------------------------- textos
T = {
    "es": {
        "lang_label": "Idioma / Language",
        "title": "# ¿Qué mira el detector de deepfakes?",
        "intro": ("Xception entrenada con caras de FaceForensics++ (video c23, semilla 42). Las imágenes no se guardan.\n\n"
                  "**1.** Sube una foto con una cara · **2.** Pulsa *Analizar* · "
                  "**3.** Cambia la comparación para ver los otros modelos (no hace falta volver a analizar)."),
        "comp_label": "Comparar",
        "comp": {"B": "Fase B · BCE vs. Focal", "D": "Fase D · R1 vs. R3", "BD": "Antes y después · BCE vs. R3"},
        "comp_desc": {
            "B": ("**Fase B** · los dos se entrenan solo con Deepfakes; cambia la función de pérdida.  \n"
                  "**BCE:** pérdida estándar · **Focal (γ = 2):** da más peso a los ejemplos difíciles."),
            "D": ("**Fase D** · Deepfakes + falsos sintéticos **SBI** (un frame real mezclado con una copia alterada de sí mismo).  \n"
                  "**R1:** solo agrega SBI (efecto de los datos) · **R3:** SBI + Focal + compresión JPEG y desenfoque al entrenar (modelo final)."),
            "BD": ("**Antes y después** · **BCE** (Fase B: solo Deepfakes) frente a **R3** "
                   "(Fase D, modelo final: SBI + Focal + JPEG)."),
        },
        "missing": "Fase D no disponible: faltan {f} en `results/checkpoints/` (output del notebook 07).",
        "input": "Imagen con una cara",
        "analyze": "Analizar",
        "clear": "Limpiar",
        "options": "Opciones",
        "precrop": "La imagen ya es un recorte de la cara (no detectar)",
        "guides": "Dibujar los límites de las regiones",
        "crop": "Recorte que ve el modelo",
        "empty": "Los resultados aparecerán aquí después de pulsar <b>Analizar</b>.",
        "no_image": "Primero sube una imagen.",
        "msg_crop": "Cara detectada y recortada como en el entrenamiento.",
        "msg_precrop": "Se usó la imagen completa como recorte de cara.",
        "msg_noface": "No se detectó ninguna cara. Prueba con otra imagen o marca «La imagen ya es un recorte».",
        "msg_out": "La cara detectada queda fuera de la imagen.",
        "fake": "EVIDENCIA DE FALSO", "real": "SIN EVIDENCIA",
        "pfake": "p(falso)",
        "conf": {"alta": "lejos del umbral", "media": "", "baja": "cerca del umbral: poco fiable"},
        "agree_fake": "Los dos modelos encuentran evidencia de falso.",
        "agree_real": "Ninguno encuentra evidencia de falso. Eso <b>no prueba</b> que la imagen sea auténtica: "
                      "cada modelo solo reconoce las manipulaciones que aprendió.",
        "disagree": "Los modelos no coinciden: el caso está en el límite de lo que aprendió alguno de ellos.",
        "test": "En el test detecta: DF {0} % · F2F {1} % · FS {2} % · NT {3} % · reales bien: {4} %",
        "reg": {"interior": "interior", "borde": "borde", "fuera": "fuera"},
        "pat": {"interior": "Mapa en el interior de la cara", "fuera": "Mapa disperso, fuera de la cara", "mixto": "Mapa mixto"},
        "coh_ok_fake": "patrón típico de una detección",
        "coh_ok_real": "patrón típico cuando no hay evidencia",
        "coh_no": "patrón atípico: interpretar con cautela",
        "find_title": "Resultados del proyecto y límites",
        "find": """
<div class="hallazgos">
  <div class="hz-card hz-find">
    <div class="hz-head">Fase B · solo Deepfakes</div>
    <ul>
      <li>Detecta Deepfakes (AUC 0.997), con la evidencia en el <b>interior</b> de la cara, no en el borde de la mezcla.</li>
      <li>No reconoce FaceSwap (AUC 0.26) y poco Face2Face (0.69).</li>
      <li>Focal Loss solo mejora NeuralTextures (+0.04).</li>
    </ul>
  </div>
  <div class="hz-card hz-fix">
    <div class="hz-head">Fase D · + falsos sintéticos (SBI)</div>
    <ul>
      <li>R3: FaceSwap 0.26 → <b>0.76</b> y Face2Face 0.69 → <b>0.85</b> (AUC, 3 semillas), sin perder Deepfakes.</li>
      <li>El mapa no pasa al borde: R3 encuentra la evidencia del interior en más falsos.</li>
      <li>Costo: más falsas alarmas (87 % de reales bien, frente a 95 %).</li>
    </ul>
  </div>
  <div class="hz-card hz-lim">
    <div class="hz-head">Límites de la demo</div>
    <ul>
      <li>Los modelos vieron frames de video comprimido: una foto de cámara o celular está fuera de esa distribución.</li>
      <li>Filtros, retoques o imágenes de IA generativa no se entrenaron: lo normal es que salgan «sin evidencia».</li>
      <li>Un caso aislado no prueba nada: las conclusiones salen de 3 semillas con IC 95 %.</li>
    </ul>
  </div>
</div>""",
        "guide_title": "Cómo leer los resultados",
        "guide": """
<div class="guia">
<p><span class="chip fake">EVIDENCIA DE FALSO</span> si p(falso) ≥ 0.5; si no, <span class="chip real">SIN EVIDENCIA</span>.
«Sin evidencia» no significa auténtica: cada tarjeta indica qué porcentaje de cada manipulación detecta ese modelo en el test
(DF Deepfakes, F2F Face2Face, FS FaceSwap, NT NeuralTextures).</p>
<p class="nota-auc">Las cifras del proyecto son <b>AUC</b> (si los falsos puntúan más alto que los reales, sin umbral). Con el umbral 0.5
un modelo puede ordenar bien y aun así detectar poco: BCE tiene AUC 0.76 en NeuralTextures pero detecta el 21 %.</p>
<p><b>Mapa de calor (Grad-CAM).</b> <span class="sw" style="background:#d7191c"></span> rojo = zonas que empujan hacia «falso»,
<span class="sw" style="background:#2c7bb6"></span> azul = poco. Muestra <i>dónde</i> mira el modelo, no <i>por qué</i>.</p>
<p><b>Regiones</b> (recuadros blancos, aproximadas):
<span class="sw" style="background:var(--c-int)"></span> interior (ojos, nariz, boca) ·
<span class="sw" style="background:var(--c-bor)"></span> borde (contorno) ·
<span class="sw" style="background:var(--c-out)"></span> fuera (pelo, fondo). En el test, cuando el modelo detecta,
la masa está en el interior (≥ 0.6); cuando no, se dispersa fuera.</p>
</div>""",
    },
    "en": {
        "lang_label": "Idioma / Language",
        "title": "# What does the deepfake detector look at?",
        "intro": ("Xception trained on FaceForensics++ faces (c23 video, seed 42). Uploaded images are not stored.\n\n"
                  "**1.** Upload a photo with a face · **2.** Press *Analyze* · "
                  "**3.** Switch the comparison to see the other models (no need to analyze again)."),
        "comp_label": "Compare",
        "comp": {"B": "Phase B · BCE vs. Focal", "D": "Phase D · R1 vs. R3", "BD": "Before and after · BCE vs. R3"},
        "comp_desc": {
            "B": ("**Phase B** · both are trained only on Deepfakes; the loss function changes.  \n"
                  "**BCE:** standard loss · **Focal (γ = 2):** gives more weight to hard examples."),
            "D": ("**Phase D** · Deepfakes + **SBI** synthetic fakes (a real frame blended with an altered copy of itself).  \n"
                  "**R1:** adds only SBI (effect of the data) · **R3:** SBI + focal + JPEG compression and blur during training (final model)."),
            "BD": ("**Before and after** · **BCE** (Phase B: Deepfakes only) against **R3** "
                   "(Phase D, final model: SBI + focal + JPEG)."),
        },
        "missing": "Phase D not available: {f} missing from `results/checkpoints/` (notebook 07 output).",
        "input": "Image with a face",
        "analyze": "Analyze",
        "clear": "Clear",
        "options": "Options",
        "precrop": "The image is already a face crop (skip detection)",
        "guides": "Draw the region boundaries",
        "crop": "Crop the model sees",
        "empty": "Results will appear here after pressing <b>Analyze</b>.",
        "no_image": "Upload an image first.",
        "msg_crop": "Face detected and cropped as in training.",
        "msg_precrop": "The whole image was used as the face crop.",
        "msg_noface": "No face detected. Try another image or tick «The image is already a face crop».",
        "msg_out": "The detected face falls outside the image.",
        "fake": "FAKE EVIDENCE", "real": "NO EVIDENCE",
        "pfake": "p(fake)",
        "conf": {"alta": "far from the threshold", "media": "", "baja": "near the threshold: unreliable"},
        "agree_fake": "Both models find evidence of a fake.",
        "agree_real": "Neither finds evidence of a fake. That does <b>not prove</b> the image is authentic: "
                      "each model only recognises the manipulations it learned.",
        "disagree": "The models disagree: the case is at the edge of what one of them learned.",
        "test": "On the test set it catches: DF {0}% · F2F {1}% · FS {2}% · NT {3}% · reals right: {4}%",
        "reg": {"interior": "interior", "borde": "border", "fuera": "outside"},
        "pat": {"interior": "Map on the face interior", "fuera": "Diffuse map, outside the face", "mixto": "Mixed map"},
        "coh_ok_fake": "typical pattern of a detection",
        "coh_ok_real": "typical pattern when there is no evidence",
        "coh_no": "atypical pattern: interpret with caution",
        "find_title": "Project results and limits",
        "find": """
<div class="hallazgos">
  <div class="hz-card hz-find">
    <div class="hz-head">Phase B · Deepfakes only</div>
    <ul>
      <li>Detects Deepfakes (AUC 0.997), with the evidence in the face <b>interior</b>, not on the blending border.</li>
      <li>Does not recognise FaceSwap (AUC 0.26) and barely Face2Face (0.69).</li>
      <li>Focal loss only improves NeuralTextures (+0.04).</li>
    </ul>
  </div>
  <div class="hz-card hz-fix">
    <div class="hz-head">Phase D · + synthetic fakes (SBI)</div>
    <ul>
      <li>R3: FaceSwap 0.26 → <b>0.76</b> and Face2Face 0.69 → <b>0.85</b> (AUC, 3 seeds), with no loss on Deepfakes.</li>
      <li>The map does not move to the border: R3 finds the interior evidence in more fakes.</li>
      <li>Cost: more false alarms (87% of reals right, versus 95%).</li>
    </ul>
  </div>
  <div class="hz-card hz-lim">
    <div class="hz-head">Demo limits</div>
    <ul>
      <li>The models saw compressed video frames: a camera or phone photo is outside that distribution.</li>
      <li>Filters, retouching or generative-AI images were not trained on: they will usually come out “no evidence”.</li>
      <li>A single case proves nothing: the conclusions come from 3 seeds with 95% CIs.</li>
    </ul>
  </div>
</div>""",
        "guide_title": "How to read the results",
        "guide": """
<div class="guia">
<p><span class="chip fake">FAKE EVIDENCE</span> if p(fake) ≥ 0.5; otherwise <span class="chip real">NO EVIDENCE</span>.
“No evidence” does not mean authentic: each card shows what share of each manipulation that model catches on the test set
(DF Deepfakes, F2F Face2Face, FS FaceSwap, NT NeuralTextures).</p>
<p class="nota-auc">The project's figures are <b>AUCs</b> (whether fakes score higher than reals, with no threshold). At the 0.5 threshold
a model can rank well and still catch little: BCE has AUC 0.76 on NeuralTextures but catches 21%.</p>
<p><b>Heat map (Grad-CAM).</b> <span class="sw" style="background:#d7191c"></span> red = areas pushing towards “fake”,
<span class="sw" style="background:#2c7bb6"></span> blue = little. It shows <i>where</i> the model looks, not <i>why</i>.</p>
<p><b>Regions</b> (white boxes, approximate):
<span class="sw" style="background:var(--c-int)"></span> interior (eyes, nose, mouth) ·
<span class="sw" style="background:var(--c-bor)"></span> border (contour) ·
<span class="sw" style="background:var(--c-out)"></span> outside (hair, background). On the test set, when the model detects,
the mass is in the interior (≥ 0.6); when it does not, it spreads outside.</p>
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
.res-acuerdo { padding:8px 12px; border-radius:8px; margin-bottom:10px; line-height:1.4;
               background:rgba(127,140,141,0.15); border-left:4px solid var(--c-mix); }
.res-cards { display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:12px; }
.res-card { border:1px solid rgba(127,127,127,0.35); border-radius:10px; padding:12px 14px; border-top:5px solid var(--c-mix); }
.res-card.fake { border-top-color:var(--c-fake); } .res-card.real { border-top-color:var(--c-real); }
.res-card h4 { margin:0 0 6px; font-size:1em; }
.res-card h4 .fase { font-weight:500; opacity:0.65; font-size:0.85em; margin-left:6px; }
.res-dec { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }
.res-dec .chip { font-size:0.95em; padding:3px 12px; }
.res-p { font-size:1.4em; font-weight:700; font-variant-numeric:tabular-nums; }
.res-conf { font-size:0.85em; opacity:0.8; }
.res-test { font-size:0.8em; opacity:0.8; margin:6px 0 8px; font-variant-numeric:tabular-nums; }
.res-bar { display:flex; height:20px; border-radius:6px; overflow:hidden; }
.res-bar div { color:#fff; font-size:0.75em; font-weight:700; display:flex; align-items:center; justify-content:center;
               white-space:nowrap; overflow:hidden; }
.res-pat { margin-top:8px; font-size:0.85em; }
.res-pat.ok::after { content:" ✓"; color:var(--c-real); font-weight:700; }
.res-pat.no::after { content:" ⚠"; color:var(--c-int); font-weight:700; }
.guia p { margin:0 0 8px; line-height:1.45; }
.nota-auc { padding:8px 12px; border-radius:8px; background:rgba(232,131,58,0.10); border-left:3px solid rgba(232,131,58,0.7); }
.hallazgos { display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:12px; }
.hz-card { border-radius:10px; padding:12px 16px; line-height:1.45; }
.hz-card ul { margin:0; padding-left:1.2em; } .hz-card li { margin-bottom:4px; }
.hz-head { font-weight:700; margin-bottom:6px; }
.hz-find { background:rgba(60,125,217,0.10); border:1px solid rgba(60,125,217,0.35); }
.hz-lim { background:rgba(232,131,58,0.10); border:1px solid rgba(232,131,58,0.40); }
.hz-fix { background:rgba(46,160,90,0.10); border:1px solid rgba(46,160,90,0.40); }
.cap-wrap { padding:0 !important; min-height:0 !important; margin-bottom:calc(-1 * var(--layout-gap, 12px) + 4px) !important; }
.cap-wrap > div { min-height:0 !important; }
.cap { font-size:0.85em; font-weight:600; opacity:0.85; margin:0; line-height:1.3; }
.cap3 { display:grid; grid-template-columns:repeat(3, 1fr); gap:var(--layout-gap, 12px); }
#comparar .wrap { gap:6px; }
#comp-desc p { margin:0; font-size:0.9em; opacity:0.85; }

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
def buscar_ckpt(nombre: str, ckpt_dir: str | None) -> Path | None:
    dirs = [ckpt_dir, os.environ.get("DEMO_CKPT_DIR"), ROOT / "results" / "checkpoints", "/kaggle/input"]
    for d in dirs:
        if d and Path(d).exists():
            hit = sorted(Path(d).rglob(f"{nombre}.pt"))
            if hit:
                return hit[0]
    return None


def cargar_modelos(ckpt_dir: str | None = None) -> dict:
    """Carga los checkpoints disponibles. Exige al menos los de la Fase B; los de la Fase D son opcionales."""
    out = {}
    for clave, (_, _, nombre, _) in MODELOS.items():
        ruta = buscar_ckpt(nombre, ckpt_dir)
        if ruta is None:
            continue
        m = TD.build_model(False, 1)
        m.load_state_dict(torch.load(ruta, map_location="cpu", weights_only=False)["model"])
        out[clave] = m.to(DEVICE).eval()
    if not {"bce", "focal"} <= set(out):
        raise FileNotFoundError("Faltan bce_s42.pt o focal_g2_s42.pt (output del notebook 04): copiarlos a "
                                "results/checkpoints/ o indicar la carpeta con --ckpt-dir.")
    return out


def comparaciones_disponibles(modelos: dict) -> list[str]:
    return [c for c, par in COMPARACIONES.items() if set(par) <= set(modelos)]


def faltantes(modelos: dict) -> list[str]:
    return [f"{MODELOS[k][2]}.pt" for k in MODELOS if k not in modelos]


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
    """Corre detección + todos los modelos cargados. Devuelve un dict sin texto (el texto depende del idioma)."""
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
    """¿La decisión coincide con el patrón de los notebooks 06 y 08? detecta ↔ interior, no detecta ↔ fuera."""
    return (r["p"] >= UMBRAL and patron(r) == "interior") or (r["p"] < UMBRAL and patron(r) == "fuera")


def tarjeta(clave: str, r: dict, t: dict) -> str:
    nombre, fase, _, det = MODELOS[clave]
    falso = r["p"] >= UMBRAL
    cls = "fake" if falso else "real"
    seg = "".join(
        f'<div style="width:{100 * r[k]:.1f}%;background:var({v})" title="{t["reg"][k]} {r[k]:.2f}">'
        f'{(t["reg"][k] + " ") if r[k] >= 0.22 else ""}{r[k]:.2f}</div>' if r[k] >= 0.08 else
        f'<div style="width:{100 * r[k]:.1f}%;background:var({v})" title="{t["reg"][k]} {r[k]:.2f}"></div>'
        for k, v in (("interior", "--c-int"), ("borde", "--c-bor"), ("fuera", "--c-out")))
    pat = patron(r)
    ok = coherente(r)
    lectura = t["coh_ok_fake" if falso else "coh_ok_real"] if ok else t["coh_no"]
    conf = t["conf"][confianza(r["p"])]
    return f"""
<div class="res-card {cls}">
  <h4>{html.escape(nombre)}<span class="fase">{"Fase" if t is T["es"] else "Phase"} {fase}</span></h4>
  <div class="res-dec"><span class="chip {cls}">{t["fake"] if falso else t["real"]}</span>
    <span class="res-p">{r["p"]:.3f}</span><span class="res-conf">{t["pfake"]}{" · " + conf if conf else ""}</span></div>
  <div class="res-test">{t["test"].format(*det)}</div>
  <div class="res-bar">{seg}</div>
  <div class="res-pat {"ok" if ok else "no"}">{t["pat"][pat]} · {lectura}</div>
</div>"""


def render(res: dict | None, lang: str, comp: str = "B") -> str:
    t = T[lang]
    if not res:
        return f'<div class="res-vacio">{t["empty"]}</div>'
    if "modelos" not in res:
        return f'<div class="res-acuerdo" style="border-left-color:var(--c-int)">{t[res["msg"]]}</div>'
    par = [k for k in COMPARACIONES[comp] if k in res["modelos"]]
    dec = {res["modelos"][k]["p"] >= UMBRAL for k in par}
    if dec == {True}:
        acuerdo = f'<div class="res-acuerdo" style="border-left-color:var(--c-fake)">{t["agree_fake"]}</div>'
    elif dec == {False}:
        acuerdo = f'<div class="res-acuerdo" style="border-left-color:var(--c-real)">{t["agree_real"]}</div>'
    else:
        acuerdo = f'<div class="res-acuerdo" style="border-left-color:var(--c-int)">{t["disagree"]}</div>'
    return (f'<div class="res-msg">{t[res["msg"]]}</div>{acuerdo}'
            f'<div class="res-cards">{"".join(tarjeta(k, res["modelos"][k], t) for k in par)}</div>')


def analizar(modelos: dict, imagen, ya_recortada: bool = False, guias: bool = True, lang: str = "es", comp: str = "B"):
    """Atajo para pruebas sin interfaz: (recorte, overlay A, overlay B, html) de la comparación `comp`."""
    res = procesar(modelos, imagen, ya_recortada, guias)
    ms = res.get("modelos", {})
    a, b = COMPARACIONES[comp]
    return res.get("crop"), ms.get(a, {}).get("overlay"), ms.get(b, {}).get("overlay"), render(res, lang, comp)


# --------------------------------------------------------------------------------------------- interfaz
def cabecera_entrada(t: dict) -> str:
    return f'<div class="cap">{t["input"]}</div>'


def cabecera_salidas(t: dict, comp: str) -> str:
    a, b = (MODELOS[k][0] for k in COMPARACIONES[comp])
    return (f'<div class="cap cap3"><span>{t["crop"]}</span><span>Grad-CAM · {html.escape(a)}</span>'
            f'<span>Grad-CAM · {html.escape(b)}</span></div>')


def desc_comparacion(t: dict, comp: str, modelos: dict) -> str:
    txt = t["comp_desc"][comp]
    falta = faltantes(modelos)
    if falta:
        txt += "  \n⚠ " + t["missing"].format(f=", ".join(f"`{f}`" for f in falta))
    return txt


def construir_ui(modelos: dict):
    import gradio as gr

    comps = comparaciones_disponibles(modelos)
    opciones_comp = lambda tt: [(tt["comp"][c], c) for c in comps]  # noqa: E731
    t = T["es"]
    with gr.Blocks(title="Deepfake detector · Grad-CAM", delete_cache=(600, 600)) as ui:
        estado = gr.State(None)   # resultados de todos los modelos (con overlays) para cambiar de comparación sin recalcular
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
                comp = gr.Radio(opciones_comp(t), value="B", label=t["comp_label"], elem_id="comparar")
                comp_desc = gr.Markdown(desc_comparacion(t, "B", modelos), elem_id="comp-desc")
                cap_out = gr.HTML(cabecera_salidas(t, "B"), elem_classes="cap-wrap")
                with gr.Row():  # sin etiqueta propia: cambiarla recarga la imagen y parpadea al cambiar de idioma
                    o_crop = gr.Image(show_label=False, interactive=False, height=230)
                    o_a = gr.Image(show_label=False, interactive=False, height=230)
                    o_b = gr.Image(show_label=False, interactive=False, height=230)
                resultado = gr.HTML(render(None, "es"), elem_id="resultado")
        with gr.Accordion(t["guide_title"], open=True) as guia_acc:
            guia = gr.HTML(t["guide"], elem_id="guia")
        with gr.Accordion(t["find_title"], open=True) as hall_acc:
            hallazgos = gr.HTML(t["find"], elem_id="hallazgos")

        def overlays(res, c):
            ms = (res or {}).get("modelos", {})
            return tuple(ms.get(k, {}).get("overlay") for k in COMPARACIONES[c])

        def correr(im, y, g, lang_nombre, c):
            res = procesar(modelos, im, y, g)
            return (res.get("crop"), *overlays(res, c), res, render(res, IDIOMAS[lang_nombre], c))

        def cambiar_comp(c, lang_nombre, res):
            tt = T[IDIOMAS[lang_nombre]]
            return (desc_comparacion(tt, c, modelos), cabecera_salidas(tt, c), *overlays(res, c),
                    render(res, IDIOMAS[lang_nombre], c))

        def cambiar_idioma(lang_nombre, c, res):
            tt = T[IDIOMAS[lang_nombre]]
            # Solo se actualizan textos; las imágenes no se tocan.
            return (tt["title"], tt["intro"], cabecera_entrada(tt), cabecera_salidas(tt, c),
                    gr.update(choices=opciones_comp(tt), value=c, label=tt["comp_label"]), desc_comparacion(tt, c, modelos),
                    gr.update(value=tt["analyze"]), gr.update(value=tt["clear"]), gr.update(label=tt["options"]),
                    gr.update(label=tt["precrop"]), gr.update(label=tt["guides"]),
                    render(res, IDIOMAS[lang_nombre], c), gr.update(label=tt["guide_title"]), tt["guide"],
                    gr.update(label=tt["find_title"]), tt["find"])

        def borrar(lang_nombre):
            return None, None, None, None, None, render(None, IDIOMAS[lang_nombre])

        boton.click(correr, [entrada, ya, guias, idioma, comp], [o_crop, o_a, o_b, estado, resultado],
                    api_name="analizar")
        comp.change(cambiar_comp, [comp, idioma, estado], [comp_desc, cap_out, o_a, o_b, resultado],
                    api_name=False, show_progress="hidden", queue=False)
        limpiar.click(borrar, [idioma], [entrada, o_crop, o_a, o_b, estado, resultado], api_name=False)
        idioma.change(cambiar_idioma, [idioma, comp, estado],
                      [titulo, intro, cap_in, cap_out, comp, comp_desc, boton, limpiar, opciones, ya, guias,
                       resultado, guia_acc, guia, hall_acc, hallazgos], api_name=False,
                      show_progress="hidden", queue=False)  # sin la animación de «cargando» en cada texto
    return ui


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt-dir", help="carpeta con los checkpoints *_s42.pt")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="enlace público temporal de Gradio (p. ej. desde Kaggle)")
    args = ap.parse_args()
    modelos = cargar_modelos(args.ckpt_dir)
    print(f"Modelos cargados en {DEVICE}: {', '.join(MODELOS[k][0] for k in modelos)}")
    if faltantes(modelos):
        print("Faltan (la Fase D queda desactivada):", ", ".join(faltantes(modelos)))
    construir_ui(modelos).launch(server_name="127.0.0.1", server_port=args.port, share=args.share, css=CSS, js=JS)


if __name__ == "__main__":
    main()

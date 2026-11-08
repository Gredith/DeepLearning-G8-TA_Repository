"""Genera los notebooks de Kaggle: 02 (preprocesamiento) y 03 (BCE vs Focal Loss)."""
import nbformat as nbf
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
NB = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle"
META = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
        "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}}


def caras(source):
    """Celda que muestra caras de FF++: tools/limpiar_caras.py borra su salida antes de versionar."""
    return code(source, metadata={"tags": ["caras"]})


CLONE = '''# Settings de Kaggle: Accelerator = GPU T4 x2 · Internet = On
import os, sys, subprocess
from pathlib import Path
REPO_URL = "https://github.com/Gredith/DeepLearning-G8-TA_Repository.git"
ROOT = Path("/kaggle/working/repo") if Path("/kaggle").exists() else Path.cwd().parents[1]
if Path("/kaggle").exists() and not ROOT.exists():
    subprocess.run(["git", "clone", "-q", REPO_URL, str(ROOT)], check=True)
os.chdir(ROOT); sys.path.insert(0, str(ROOT / "src" / "kaggle_e2"))
print("ROOT =", ROOT)'''

# ------------------------------------------------------------------ 02
nb02 = [
md("""# 02 · Datos — descarga de FaceForensics++ (c23) y recorte de caras
Genera los recortes de cara (299×299, 16 frames por video) **solo** para los videos que usan los experimentos y los entrega como un único archivo para crear el *Kaggle Dataset* privado `ffpp-crops-g8` (D-05, D-12).

| Split | Pares oficiales usados | Categorías |
|---|---|---|
| train | 100 de 360 | original + Deepfakes |
| val | 30 de 70 | original + Deepfakes |
| test | 70 de 70 | original + Deepfakes, Face2Face, FaceSwap, NeuralTextures |

**Output de este notebook (solo 3 archivos y la carpeta `privado/`):** `crops.tar` (~3 GB), `manifest.csv` (recortes y descartes por video) y `crops_meta.json`.

**Cómo usarlo**
1. Settings: Accelerator = GPU T4 · Internet = On · Input: el Dataset privado con el script oficial de descarga.
2. **Save Version → Save & Run All (Commit)**. El output solo se puede convertir en Dataset desde una versión guardada, no desde la sesión interactiva.
3. En la versión terminada: pestaña *Output* → **New Dataset** → nombre `ffpp-crops-g8`, visibilidad **Private**.
4. El 03 (y la sección 5 del 01) lo agregan como input y lo descomprimen solos.

> **Licencia FF++ y sincronización con GitHub.** Kaggle sube este notebook a GitHub *con sus salidas*. Por eso ninguna celda muestra caras ni el log del script de descarga: la inspección visual se guarda en `privado/` (output privado de Kaggle) y la descarga escribe en `privado/descarga.log`."""),
code(CLONE),
code('''!pip install -q --no-deps facenet-pytorch   # --no-deps: evita que reinstale otra versión de torch
import json, shutil, pandas as pd, torch
import data as D, preprocess as P, kaggle_io as K
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FFPP = Path("/tmp/ffpp")                 # videos: disco temporal, se borran tras recortar cada categoría
CROPS = Path("/tmp/crops_build/crops")    # recortes: se empaquetan en /kaggle/working/crops.tar
K.PRIVADO.mkdir(parents=True, exist_ok=True)
N_TRAIN, N_VAL, N_TEST, N_FRAMES, SEED = 100, 30, None, 16, 42'''),
md("## 1. Qué videos se necesitan (splits oficiales, partición por video)"),
code('''tr = D.subset_pairs(D.load_pairs("results/splits", "train"), N_TRAIN, SEED)
va = D.subset_pairs(D.load_pairs("results/splits", "val"), N_VAL, SEED)
te = D.subset_pairs(D.load_pairs("results/splits", "test"), N_TEST, SEED)
need, split_of = {}, {}
for split, pairs, cats in [("train", tr, ["Deepfakes"]), ("val", va, ["Deepfakes"]), ("test", te, D.METHODS)]:
    reals, fakes = D.video_ids(pairs)
    need.setdefault(D.REAL, set()).update(reals)
    split_of.update({v: split for v in reals + fakes})
    for c in cats: need.setdefault(c, set()).update(fakes)
need = {k: sorted(v) for k, v in need.items()}
# chequeo anti-fuga: ninguna identidad en dos splits
ids = {s: {v for p in ps for v in p} for s, ps in [("train", tr), ("val", va), ("test", te)]}
assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])
{k: len(v) for k, v in need.items()}'''),
md("""## 2. Descarga y recorte, una categoría a la vez
El script oficial de descarga llega con la aprobación del acceso. **No se versiona ni se enlaza** (términos de FF++): súbanlo a Kaggle como Dataset privado (*Add Input → Upload*); la celda lo encuentra sola.

Por cada categoría: descarga los 1000 videos c23 → recorta solo los necesarios (MTCNN, caja ×1.3, 16 frames equiespaciados) → **borra los videos**. Así el disco nunca guarda más de una categoría de videos. Si la sesión se corta, volver a correr: los videos ya recortados se saltan."""),
code('''import subprocess
def _es_script_ffpp(p):
    n = p.name.lower()
    return n.endswith(".py") and "faceforensics" in n and "download" in n
SCRIPT = next((p for p in Path("/kaggle/input").rglob("*.py") if _es_script_ffpp(p)), None) if Path("/kaggle/input").exists() else None
assert SCRIPT, "No se encontró el script de FF++: súbanlo como Kaggle Dataset privado (Add Input → Upload)"
print("Script de descarga encontrado en los inputs privados.")   # sin ruta ni URL: esta salida va a GitHub

LOG = K.PRIVADO / "descarga.log"
rows = []
for cat in [D.REAL, *D.METHODS]:
    with open(LOG, "a") as log:   # `echo |` acepta los términos de uso que pide el script
        r = subprocess.run(f'echo "" | python "{SCRIPT}" "{FFPP}" -d {cat} -c c23 -t videos --server EU2',
                           shell=True, stdout=log, stderr=subprocess.STDOUT)
    assert r.returncode == 0, f"Falló la descarga de {cat}: revisar privado/descarga.log en el output"
    vdir = P.video_path(FFPP, cat, "x").parent
    n_vid = len(list(vdir.glob("*.mp4")))
    rows += P.process_all(FFPP, CROPS, {cat: need[cat]}, n_frames=N_FRAMES, device=DEVICE, progress=False)
    shutil.rmtree(vdir, ignore_errors=True)
    print(f"{cat:15s} videos descargados={n_vid:4d}  recortados={len(need[cat]):4d}  "
          f"recortes={sum(x['recortes'] for x in rows if x['categoria'] == cat)}")'''),
md("## 3. Manifiesto, conteos y auditoría de descartes"),
code('''man = pd.DataFrame(rows)
man["split"] = man.video.map(split_of)
man.to_csv(CROPS / "manifest.csv", index=False)
resumen = man.groupby("categoria").agg(videos=("video", "count"), sin_video=("existe_video", lambda s: (~s).sum()),
                                      recortes=("recortes", "sum"), esperados=("video", lambda s: len(s) * N_FRAMES),
                                      videos_sin_cara=("recortes", lambda s: (s == 0).sum()),
                                      frames_sin_cara=("frames_sin_cara", "sum"), frames_no_leidos=("frames_no_leidos", "sum"))
print(resumen)
incompletos = man[man.recortes < N_FRAMES]
print(f"\\nVideos con menos de {N_FRAMES} recortes: {len(incompletos)}")
print(incompletos[["categoria", "video", "split", "recortes", "frames_sin_cara", "frames_no_leidos", "recortes_vacios"]].to_string(index=False))'''),
caras('''# Inspección visual: real vs. cada manipulación del mismo target → solo a privado/, no se muestra
import matplotlib.pyplot as plt
from PIL import Image
a, b = te[0]
paths = [CROPS / D.REAL / a] + [CROPS / m / f"{a}_{b}" for m in D.METHODS]
fig, axes = plt.subplots(1, 5, figsize=(14, 3))
for ax, p in zip(axes, paths):
    fs = sorted(p.glob("*.png"))
    if fs: ax.imshow(Image.open(fs[len(fs) // 2]))
    ax.set_title(p.parent.name, loc="left"); ax.axis("off")
K.save_private(fig, "ejemplo_recortes.png")'''),
md("""## 4. Empaquetar el output
Deja en `/kaggle/working` solo `crops.tar`, `manifest.csv`, `crops_meta.json` y `privado/`. Se borra la copia del repo para que el output sea liviano y fácil de convertir en Dataset."""),
code('''import os
commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
meta = {"commit": commit, "seed": SEED, "n_frames": N_FRAMES, "pares": {"train": N_TRAIN, "val": N_VAL, "test": len(te)},
        "recortes_por_categoria": {k: int(v) for k, v in man.groupby("categoria").recortes.sum().items()},
        "videos_incompletos": int(len(incompletos)), "detector": "MTCNN (facenet-pytorch), caja x1.3", "compresion": "c23"}
tar = K.pack_crops(CROPS, K.WORK)
shutil.copy(CROPS / "manifest.csv", K.WORK / "manifest.csv")
(K.WORK / "crops_meta.json").write_text(json.dumps(meta, indent=1))
os.chdir(K.WORK)
if K.KAGGLE:
    shutil.rmtree("/kaggle/working/repo", ignore_errors=True)
    shutil.rmtree(CROPS.parent, ignore_errors=True)
print(f"{tar.name}: {tar.stat().st_size / 1e9:.2f} GB")
print(sorted(p.name for p in K.WORK.iterdir()))'''),
md("""## 5. (Plan B) Crear el Dataset por API
Solo si *New Dataset* desde el output vuelve a fallar. Requiere un **Kaggle Secret** `KAGGLE_KEY` (Add-ons → Secrets) del integrante dueño del Dataset. Cambien `CREAR_POR_API = True` y vuelvan a guardar la versión. El Dataset se crea **privado** (es el valor por defecto de `kaggle datasets create`)."""),
code('''CREAR_POR_API = False
KAGGLE_USER = "ronaldosegura"   # dueño del Dataset
if CREAR_POR_API:
    from kaggle_secrets import UserSecretsClient
    os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = KAGGLE_USER, UserSecretsClient().get_secret("KAGGLE_KEY")
    up = Path("/tmp/ds_upload"); up.mkdir(exist_ok=True)
    for f in ["crops.tar", "manifest.csv", "crops_meta.json"]:
        shutil.copy(K.WORK / f, up / f)
    (up / "dataset-metadata.json").write_text(json.dumps({"title": "ffpp-crops-g8", "id": f"{KAGGLE_USER}/ffpp-crops-g8",
                                                          "licenses": [{"name": "unknown"}]}))
    r = subprocess.run(["kaggle", "datasets", "create", "-p", str(up), "--dir-mode", "skip", "-q"], capture_output=True, text=True)
    print(r.returncode, r.stdout[-500:], r.stderr[-500:])
    shutil.rmtree(up)'''),
]

# ------------------------------------------------------------------ 03
nb03 = [
md("""# 03 · Fase B (preliminar) — BCE vs. Focal Loss en Xception
**Un cambio:** la pérdida, BCE (γ = 0) → Focal Loss (γ = 2). Todo lo demás es idéntico: arquitectura, pesos iniciales, datos, splits, semilla, épocas, optimizador y augmentación.

### Hipótesis previa (escrita antes de correr; copiada del E1, ver D-08)
Con BCE, el modelo aprende rápido los ejemplos fáciles, que suelen ser los que muestran los artefactos más obvios del generador de entrenamiento, y deja de aprender de ellos. Focal Loss reduce el peso de esos ejemplos fáciles y concentra el gradiente en los difíciles (falsificaciones sutiles, frames comprimidos, caras de perfil).

**Predicciones concretas:**
1. AUC *in-domain* (Deepfakes) casi igual (|Δ| < 0.01).
2. AUC cross-manipulation un poco mayor con Focal Loss, más en FaceSwap (comparte artefactos de mezcla con Deepfakes) que en NeuralTextures (artefactos de otra naturaleza).
3. Diagnóstico del mecanismo: con γ = 2, el % de ejemplos "fáciles" (p_t > 0.9) en entrenamiento crecerá **más lento** que con BCE.

**Hipótesis alternativa:** no mejora, porque los ejemplos difíciles tienen la misma huella del generador; Focal Loss solo afinaría esa huella."""),
code(CLONE),
md("""## 0. Datos
Input: el Dataset privado **`ffpp-crops-g8`** (output del notebook 02). Esta celda encuentra `crops.tar`, lo descomprime en `/tmp` y compara los recortes con `manifest.csv`.

> **Licencia FF++:** Kaggle sube este notebook a GitHub con sus salidas. Ninguna celda muestra caras: el Grad-CAM se guarda en `privado/` (output privado de Kaggle). Las métricas y curvas, que no contienen caras, sí se muestran."""),
code('''import json, pandas as pd, numpy as np, torch, matplotlib.pyplot as plt
from train import Config, run
import plotstyle as S, kaggle_io as K; S.apply()
CROPS = K.find_crops()
MAN = K.find_manifest(CROPS)
if MAN is not None:
    conteo = K.check_counts(CROPS, MAN); print(conteo); assert conteo.ok.all(), "Los recortes no coinciden con el manifiesto"
OUT = K.WORK / "resultados"; CKPT = K.WORK / "checkpoints"
BASE = dict(crops_dir=str(CROPS), out_dir=str(OUT / "runs"), ckpt_dir=str(CKPT), n_train_pairs=100, n_val_pairs=30, n_test_pairs=None,
            frames_per_video=16, epochs=6, warmup_epochs=1, batch_size=32, lr=2e-4, seed=42)
if os.environ.get("SMOKE"):  # prueba rápida del pipeline en CPU con datos sintéticos
    BASE.update(n_train_pairs=4, n_val_pairs=2, n_test_pairs=2, frames_per_video=3, epochs=2, batch_size=8, img_size=96, num_workers=0)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print(K.snapshot_env(ROOT, OUT / "entorno.json"))'''),
md("## 1. Línea base: BCE (γ = 0)"),
code('''res_bce = run(Config(run_name="bce_s42", gamma=0.0, **BASE))'''),
md("## 2. Intervención: Focal Loss (γ = 2)"),
code('''res_fl = run(Config(run_name="focal_g2_s42", gamma=2.0, **BASE))'''),
md("## 3. Comparación"),
code('''runs = {"BCE (γ=0)": res_bce, "Focal (γ=2)": res_fl}
rows = []
for name, r in runs.items():
    for m, t in r["test"].items():
        rows.append({"pérdida": name, "test": m, "AUC frame": t.get("auc_frame"), "AUC video": t.get("auc_video"),
                     "acc video": t.get("acc_video@0.5"), "videos": t.get("n_videos")})
tab = pd.DataFrame(rows)
indom = tab[tab.test == "Deepfakes"].set_index("pérdida")["AUC frame"]
tab["caída rel. vs in-domain (%)"] = tab.apply(lambda r: 100 * (indom[r["pérdida"]] - r["AUC frame"]) / indom[r["pérdida"]], axis=1)
tab.to_csv(OUT / "comparacion_bce_vs_focal.csv", index=False)
pivot = tab.pivot(index="test", columns="pérdida", values="AUC frame")
pivot["Δ (focal − BCE)"] = pivot["Focal (γ=2)"] - pivot["BCE (γ=0)"]
pivot.round(4)'''),
code('''order = ["Deepfakes", "FaceSwap", "Face2Face", "NeuralTextures"]
x = np.arange(len(order)); w = 0.38
fig, ax = plt.subplots(figsize=(8, 4))
for i, (name, col) in enumerate(zip(runs, S.SERIES)):
    v = pivot.loc[order, name].values
    ax.bar(x + (i - 0.5) * w, v, w * 0.95, color=col, label=name)
    for xi, vi in zip(x, v): ax.text(xi + (i - 0.5) * w, vi + 0.01, f"{vi:.3f}", ha="center", fontsize=8, color=S.MUTED)
ax.axhline(0.5, color=S.MUTED, lw=1, ls="--"); ax.text(3.45, 0.51, "azar", color=S.MUTED, fontsize=8, ha="right")
ax.set_xticks(x, ["Deepfakes\\n(in-domain)", "FaceSwap", "Face2Face", "NeuralTextures"])
ax.set_ylim(0, 1.05); ax.set_ylabel("AUC (frame)"); ax.grid(axis="x", visible=False)
ax.set_title("Entrenado solo en Deepfakes: AUC por manipulación de test", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
plt.tight_layout(); plt.savefig(OUT / "fig_auc_cross_manipulation.png", dpi=200); plt.show()'''),
code('''fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for (name, r), col in zip(runs.items(), S.SERIES):
    h = pd.DataFrame(r["history"])
    axes[0].plot(h.epoch, h.val_auc_frame, marker="o", ms=6, color=col, label=name)
    axes[1].plot(h.epoch, h.train_pct_faciles, marker="o", ms=6, color=col, label=name)
axes[0].set_title("AUC de validación (Deepfakes)", loc="left"); axes[0].set_xlabel("época")
axes[1].set_title("% de ejemplos fáciles en train (p_t > 0.9)", loc="left"); axes[1].set_xlabel("época")
for a in axes: a.axvline(1.5, color=S.GRID, lw=1); a.legend()
plt.tight_layout(); plt.savefig(OUT / "fig_curvas_entrenamiento.png", dpi=200); plt.show()'''),
md("""## 4. Grad-CAM: ¿dónde mira cada modelo?
Las grillas se guardan en `privado/fig_gradcam_<corrida>.png` (output privado de Kaggle) y **no se muestran** en la celda, porque contienen caras de FF++."""),
caras('''import glob, gradcam, data as D, teardown as T
from PIL import Image
tf = D.eval_transform(299)
te = D.load_pairs("results/splits", "test")
a, b = next(p for p in te if (CROPS / "FaceSwap" / f"{p[0]}_{p[1]}").exists()); pick = {}
for cat, vid in [("real", a), ("Deepfakes", f"{a}_{b}"), ("FaceSwap", f"{a}_{b}"), ("NeuralTextures", f"{a}_{b}")]:
    folder = CROPS / (D.REAL if cat == "real" else cat) / vid
    fs = sorted(folder.glob("*.png"))[:3]
    if fs: pick[cat] = torch.stack([tf(Image.open(f).convert("RGB")) for f in fs])
for run_name in ["bce_s42", "focal_g2_s42"]:
    m = T.build_model(False, 1); m.load_state_dict(torch.load(CKPT / f"{run_name}.pt", map_location="cpu")["model"])
    K.save_private(gradcam.grid(m, pick, title=run_name), f"fig_gradcam_{run_name}.png")'''),
md("""### 5. Contraste con la hipótesis
Valores de la corrida oficial (notebook 03 v3, semilla 42, Kaggle T4, 07/11/2026; `results/metrics/e3_*.json`). La primera corrida (v5 del 02, 28/10) se conserva en `results/metrics/b0*_*.json`.

| Predicción | Resultado observado | ¿Se cumplió? |
|---|---|---|
| 1. AUC in-domain casi igual | Deepfakes, AUC por video: 0.9971 (BCE) vs. 0.9980 (Focal); Δ = +0.0009 | Sí |
| 2. Mejora cross-manip., más en FaceSwap que en NeuralTextures | AUC por video con Focal: Face2Face 0.666 vs. 0.688 (−0.022), FaceSwap 0.213 vs. 0.299 (−0.086), NeuralTextures 0.823 vs. 0.753 (+0.070) | No (al revés: empeora FaceSwap, mejora NeuralTextures) |
| 3. % de fáciles crece más lento con γ = 2 | Época 6: 93.8 % (Focal) vs. 98.1 % (BCE); por debajo de BCE en todas las épocas tras el warm-up | Sí |

**Lectura.** El mecanismo actúa (predicción 3), pero no da la robustez predicha (predicción 2). En FaceSwap el AUC queda **por debajo de 0.5** con ambas pérdidas: el detector puntúa los FaceSwap como *más reales* que los originales, de forma sistemática (no es un error de etiquetas: `data.build_items` asigna 1 a toda manipulación).

**Advertencia de validez:** la misma configuración y semilla dio en la primera corrida NeuralTextures 0.826 (BCE) y 0.820 (Focal), y aquí 0.753 y 0.823: la variación entre ejecuciones (cuDNN, versión de librerías) es del orden de los Δ observados. Una sola corrida no basta; el barrido de semillas del notebook 04 es el que decide."""),
]

for name, cells in [("02_preprocesamiento_ffpp.ipynb", nb02), ("03_entrenamiento_bce_vs_focal.ipynb", nb03)]:
    nbf.write(nbf.v4.new_notebook(cells=cells, metadata=META), NB / name)
print("ok")

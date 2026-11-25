"""Genera el notebook de Kaggle 07 (Fase D: Deepfakes + SBI, focal y aumentos de robustez; D-14)."""
import nbformat as nbf
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
NB = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle"
META_GPU = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
            "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}}


def caras(source):
    """Celda que genera figuras con caras de FF++: se guardan en privado/ y no se muestran."""
    return code(source, metadata={"tags": ["caras"]})


CLONE = '''# Kaggle: clona el repo (Internet = On). En local: usa la copia del repo desde notebooks/kaggle/
import os, sys, subprocess
from pathlib import Path
REPO_URL = "https://github.com/Gredith/DeepLearning-G8-TA_Repository.git"
BRANCH = os.environ.get("REPO_BRANCH", "main")
ROOT = Path("/kaggle/working/repo") if Path("/kaggle").exists() else Path.cwd().parents[1]
if Path("/kaggle").exists() and not ROOT.exists():
    subprocess.run(["git", "clone", "-q", "-b", BRANCH, REPO_URL, str(ROOT)], check=True)
os.chdir(ROOT); sys.path.insert(0, str(ROOT / "src" / "kaggle_e2"))
print("ROOT =", ROOT, "| rama:", subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True).stdout.strip())'''

LICENCIA = ("> **Licencia FF++:** Kaggle sube este notebook a GitHub con sus salidas. Ninguna celda muestra caras; "
            "las figuras con caras (incluidos los SBI, que son derivados de FF++) van a `privado/`. "
            "Tablas y gráficos numéricos sí se muestran.")

cells = [
md(f"""# 07 · Fase D — Deepfakes + Self-Blended Images (SBI): ¿el detector generaliza?
**Problema (Fases B y C):** entrenado solo con Deepfakes, Xception busca una huella del interior de la cara propia de ese generador. FaceSwap no la tiene y queda por debajo del azar (AUC 0.26) en las 12 corridas; Focal Loss solo mejora NeuralTextures (+0.04).

**Intervención (D-14):** además de Deepfakes, el modelo ve **SBI**: falsos sintéticos fabricados con un solo frame real (copia alterada en color y nitidez, un poco desalineada y mezclada con el original dentro de una máscara con forma de cara). Tienen el borde de mezcla y las diferencias de color que comparten casi todas las falsificaciones, pero no la huella de ningún generador. **El modelo sigue sin ver Face2Face, FaceSwap ni NeuralTextures**: el protocolo «entrenar en una manipulación y evaluar en otra» se mantiene.

**Hipótesis H-D1 (escrita antes de correr, `docs/DECISIONS.md`):**
1. FaceSwap supera 0.5 y el IC 95 % de Δ AUC (R1 − R0) excluye 0.
2. Face2Face y NeuralTextures mejoran o se mantienen (IC que no queda por debajo de 0).
3. Deepfakes (in-domain) no baja más de 0.01.
4. *(notebook 06 con el checkpoint R1)* la masa de Grad-CAM en el borde de la cara sube en los falsos.

**Corridas (protocolo D-06 sin cambios: 6 épocas, lr 2×10⁻⁴, batch 32, mismos datos y semillas 42–44).** Ablación acumulativa fijada **antes** de ver el test, para no elegir la variante con el test:

| Grupo | Falsos de train | Pérdida | Aumentos | Para qué |
|---|---|---|---|---|
| R0 | Deepfakes | BCE | flip | línea base en el mismo entorno (repite B, ahora con estrés JPEG) |
| R1 | Deepfakes + SBI (50 %) | BCE | flip | efecto de los SBI |
| R2 | Deepfakes + SBI (50 %) | Focal γ = 2 | flip | ¿se suma la ganancia de Focal en NeuralTextures? |
| R3 | Deepfakes + SBI (50 %) | Focal γ = 2 | flip + JPEG + desenfoque | modelo final: robustez a la compresión |

En R1–R3, en cada época la mitad de los falsos de train son Deepfakes y la otra mitad SBI: el tamaño de la época y el balance 1:1 son los de D-06. El checkpoint se elige con el AUC de val sobre reales + Deepfakes + SBI deterministas de val. Todas las corridas se evalúan también con el test **recomprimido en JPEG calidad 50** (`<manipulación>@jpeg50`): es compresión simulada, no c40.

**Inputs en Kaggle:** `ffpp-crops-g8` (obligatorio). *(Opcional)* el output de una versión anterior de este 07: las corridas terminadas se copian y no se repiten.

**Tiempo:** ~10–12 min por corrida en una T4 (los SBI se fabrican en la CPU, ~8 ms por imagen): unas 2.5 h para las 12. Si se corta, agregar el output como input y volver a guardar.

{LICENCIA}"""),
code(CLONE),
code('''import json, shutil, time, numpy as np, pandas as pd, torch
from train import Config, run
import kaggle_io as K, analysis as A, data as D, sbi
CROPS = K.find_crops()
MAN = K.find_manifest(CROPS)
if MAN is not None:
    conteo = K.check_counts(CROPS, MAN); print(conteo); assert conteo.ok.all(), "Los recortes no coinciden con el manifiesto"
OUT = K.WORK / "resultados"; RUNS = OUT / "runs"; CKPT = K.WORK / "checkpoints"
RUNS.mkdir(parents=True, exist_ok=True)
BASE = dict(crops_dir=str(CROPS), out_dir=str(RUNS), ckpt_dir=str(CKPT), n_train_pairs=100, n_val_pairs=30,
            n_test_pairs=None, frames_per_video=16, epochs=6, warmup_epochs=1, batch_size=32, lr=2e-4,
            data_seed=42, stress_jpeg=(50,))
if os.environ.get("SMOKE"):  # prueba rápida en CPU con datos sintéticos
    BASE.update(n_train_pairs=4, n_val_pairs=2, n_test_pairs=2, frames_per_video=3, epochs=2, batch_size=8, img_size=96, num_workers=0)
VARIANTES = {   # grupo: (prefijo del nombre, parámetros de Config)
    "R0": ("r0_bce",              dict(train_fakes="deepfakes",     gamma=0.0, robust_aug=False)),
    "R1": ("r1_sbi_bce",          dict(train_fakes="deepfakes+sbi", gamma=0.0, robust_aug=False)),
    "R2": ("r2_sbi_focal_g2",     dict(train_fakes="deepfakes+sbi", gamma=2.0, robust_aug=False)),
    "R3": ("r3_sbi_focal_g2_aug", dict(train_fakes="deepfakes+sbi", gamma=2.0, robust_aug=True)),
}
# prioridad: con la semilla 42 ya se ve si R1 funciona; luego se completan las semillas 43 y 44
PRIORIDAD = [(v, s) for s in (42, 43, 44) for v in ("R1", "R0", "R2", "R3")]
if os.environ.get("SMOKE"):
    PRIORIDAD = [(v, 42) for v in VARIANTES]
nombre = lambda v, s: f"{VARIANTES[v][0]}_s{s}"
CKPT_GUARDAR = {nombre(v, 42) for v in VARIANTES}   # para la demo y el Grad-CAM (06); el resto se borra
MAX_HORAS = float(os.environ.get("MAX_HORAS", 10.5))
print(K.snapshot_env(ROOT, OUT / "entorno.json"))'''),
md("""## 0. Revisión visual de los SBI (privado)
Antes de entrenar, mirar `privado/sbi_ejemplos.png` en el output: cada columna es un real de train, su SBI y la máscara. El SBI debe verse casi igual al real (falsificación sutil), sin marcos, bloques ni bordes duros. La figura no se muestra aquí (licencia FF++). Abajo sí se imprimen números sin caras: área de la máscara y cuánto cambia el SBI."""),
caras('''import matplotlib.pyplot as plt
from PIL import Image
tr_pairs = D.subset_pairs(D.load_pairs("results/splits", "train"), BASE["n_train_pairs"], 42)
reales, _ = D.build_items(CROPS, tr_pairs, [], include_real=True, max_frames=BASE["frames_per_video"])
rng = np.random.default_rng(0)
muestra = [reales[i][0] for i in rng.choice(len(reales), 6, replace=False)]
fig, axes = plt.subplots(3, 6, figsize=(15, 7.5))
for j, p in enumerate(muestra):
    img = Image.open(p).convert("RGB")
    out, m = sbi.self_blend(img, np.random.default_rng(j), return_mask=True)
    for i, (x, t) in enumerate([(img, "real"), (out, "SBI"), (m, "máscara")]):
        axes[i, j].imshow(x, cmap="gray" if t == "máscara" else None, vmin=0, vmax=1 if t == "máscara" else None)
        axes[i, j].set_title(t, loc="left"); axes[i, j].axis("off")
K.save_private(fig, "sbi_ejemplos.png")'''),
code('''# Números sin caras, en 200 frames de train: área de la máscara (fracción del recorte con más de la mitad de su
# valor máximo), proporción de mezcla y diferencia media entre el SBI y su real (escala 0–255)
stats = []
for k, i in enumerate(np.random.default_rng(1).choice(len(reales), min(200, len(reales)), replace=False)):
    img = Image.open(reales[i][0]).convert("RGB")
    out, m = sbi.self_blend(img, np.random.default_rng(k), return_mask=True)
    d = np.abs(np.asarray(out, np.float32) - np.asarray(img, np.float32)).mean()
    stats.append({"area_mascara": float((m > 0.5 * m.max()).mean()), "mezcla_max": float(m.max()), "dif_media_0a255": float(d)})
pd.DataFrame(stats).describe().round(3)'''),
md("## 1. Reutilizar corridas ya hechas (inputs opcionales)"),
code('''prefijos = {p for p, _ in VARIANTES.values()}
reusadas = []
if Path("/kaggle/input").exists():
    for js in Path("/kaggle/input").rglob("*.json"):
        if js.stem.rsplit("_s", 1)[0] not in prefijos or (RUNS / js.name).exists():
            continue
        preds = js.parent / f"preds_{js.stem}.csv.gz"
        if not preds.exists():
            continue
        shutil.copy(js, RUNS / js.name); shutil.copy(preds, RUNS / preds.name); reusadas.append(js.stem)
        ck = next(Path("/kaggle/input").rglob(f"{js.stem}.pt"), None)
        if js.stem in CKPT_GUARDAR and ck is not None:
            CKPT.mkdir(parents=True, exist_ok=True); shutil.copy(ck, CKPT / ck.name)
print("reutilizadas:", sorted(reusadas) or "ninguna")'''),
md("## 2. Corridas R0–R3 × semillas"),
code('''t0 = time.time()
for v, s in PRIORIDAD:
    name = nombre(v, s)
    if (RUNS / f"{name}.json").exists() and (RUNS / f"preds_{name}.csv.gz").exists():
        print(f"[{name}] ya existe, se salta"); continue
    horas = (time.time() - t0) / 3600
    if horas > MAX_HORAS - 0.4:
        print(f"Se detiene antes de {name}: {horas:.1f} h usadas (MAX_HORAS={MAX_HORAS}). Agregar este output como input y volver a guardar."); break
    r = run(Config(run_name=name, variant=v, seed=s, **VARIANTES[v][1], **BASE), verbose=True)
    assert os.environ.get("SMOKE") or r["data"]["train"]["videos_faltantes"] == 0, f"{name}: faltan videos de train"
    if name not in CKPT_GUARDAR:
        (CKPT / f"{name}.pt").unlink(missing_ok=True)
print(f"total: {(time.time() - t0) / 60:.0f} min")'''),
md("## 3. Resultados (sin caras)\nAUC por video, media ± desviación entre semillas. Las columnas `@jpeg50` son el test recomprimido."),
code('''ORDEN = ["R0", "R1", "R2", "R3"]
runs = A.load_runs(RUNS)
tabla = A.metrics_table(runs); tabla.to_csv(OUT / "fase_d_metricas_por_corrida.csv", index=False)
res = A.summary_by_group(tabla, "auc_video", order=ORDEN); res.to_csv(OUT / "fase_d_resumen_auc_video.csv", index=False)
print(f"{len(runs)} corridas:", sorted(runs))
res.pivot(index="test", columns="grupo", values="texto")'''),
md("""## 4. ¿El efecto es real? Δ AUC por video frente a R0 con IC 95 % (bootstrap pareado, 2000 réplicas)
Mismo método que el notebook 05 (F-01): se remuestrean los videos de test una vez por réplica y se usa la misma muestra para todas las semillas y ambos grupos."""),
code('''tests = [t for t in sorted(set(tabla.test), key=str)]
filas = []
for g in ["R1", "R2", "R3"]:
    for t in tests:
        r = A.paired_bootstrap_groups(runs, g, "R0", test=t, n_boot=200 if os.environ.get("SMOKE") else 2000)
        if r: filas.append(r)
boot = pd.DataFrame(filas)
if len(boot):
    boot.to_csv(OUT / "fase_d_bootstrap_vs_r0.csv", index=False)
    boot["texto"] = boot.apply(lambda r: f"{r.delta_auc_video:+.3f} [{r.ic95_bajo:+.3f}, {r.ic95_alto:+.3f}]" + (" *" if r.excluye_0 else ""), axis=1)
    display(boot.pivot(index="test", columns="grupo_a", values="texto"))
print("* = el IC 95 % excluye 0")'''),
md("## 5. Contraste con H-D1"),
code('''def media(g, t):
    x = res[(res.grupo == g) & (res.test == t)]
    return float(x.media.iloc[0]) if len(x) else float("nan")
def ic(g, t):
    x = boot[(boot.grupo_a == g) & (boot.test == t)] if len(boot) else boot
    return (float(x.ic95_bajo.iloc[0]), float(x.ic95_alto.iloc[0])) if len(x) else (float("nan"), float("nan"))
chequeo = [
    ("1. FaceSwap R1 > 0.5", media("R1", "FaceSwap") > 0.5, f"{media('R1', 'FaceSwap'):.3f}"),
    ("1. IC Δ FaceSwap (R1 − R0) excluye 0, positivo", ic("R1", "FaceSwap")[0] > 0, "[%.3f, %.3f]" % ic("R1", "FaceSwap")),
    ("2. Face2Face: IC no queda por debajo de 0", ic("R1", "Face2Face")[1] > 0, "[%.3f, %.3f]" % ic("R1", "Face2Face")),
    ("2. NeuralTextures: IC no queda por debajo de 0", ic("R1", "NeuralTextures")[1] > 0, "[%.3f, %.3f]" % ic("R1", "NeuralTextures")),
    ("3. Deepfakes: caída ≤ 0.01", media("R1", "Deepfakes") >= media("R0", "Deepfakes") - 0.01,
     f"R0 {media('R0', 'Deepfakes'):.3f} → R1 {media('R1', 'Deepfakes'):.3f}"),
]
pd.DataFrame(chequeo, columns=["predicción", "¿se cumple?", "valor"])'''),
md("## 6. Diagnóstico del mecanismo: % de ejemplos fáciles por época\nCon SBI se espera que los fáciles crezcan más lento que en R0: los SBI son falsificaciones sutiles."),
code('''hist = A.history_table(runs)
if len(hist):
    hist.to_csv(OUT / "fase_d_historial_por_epoca.csv", index=False)
    display(hist.groupby(["grupo", "epoch"])["train_pct_faciles"].mean().unstack().round(1))'''),
md("""## 7. Llevar los resultados al repo
Descargar `resultados/` del output y copiarlo a **`results/metrics/fase_d/`** (subcarpeta: así el notebook 05 de la Fase B no mezcla estas corridas con el barrido de γ). Los checkpoints `*_s42.pt` quedan en Kaggle: el notebook 06 (Grad-CAM) y la demo los usan desde este output. Después, registrar los resultados en `docs/DECISIONS.md` (filas D-01… del registro de experimentos) y contrastarlos con H-D1."""),
]

nbf.write(nbf.v4.new_notebook(cells=cells, metadata=META_GPU), NB / "07_sbi_generalizacion.ipynb")
print("ok")

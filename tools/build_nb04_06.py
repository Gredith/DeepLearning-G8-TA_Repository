"""Genera los notebooks de Kaggle 04 (barrido de γ y semillas), 05 (análisis final) y 06 (diagnóstico Grad-CAM)."""
import sys
import nbformat as nbf
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
NB = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle"
META_GPU = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
            "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}}
META_CPU = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
            "kaggle": {"accelerator": "none", "isInternetEnabled": True}}


def caras(source):
    """Celda que genera figuras con caras de FF++: se guardan en privado/ y no se muestran."""
    return code(source, metadata={"tags": ["caras"]})


CLONE = '''# Kaggle: clona el repo (Internet = On). En local: usa la copia del repo desde notebooks/kaggle/
import os, sys, subprocess
from pathlib import Path
REPO_URL = "https://github.com/Gredith/DeepLearning-G8-TA_Repository.git"
ROOT = Path("/kaggle/working/repo") if Path("/kaggle").exists() else Path.cwd().parents[1]
if Path("/kaggle").exists() and not ROOT.exists():
    subprocess.run(["git", "clone", "-q", REPO_URL, str(ROOT)], check=True)
os.chdir(ROOT); sys.path.insert(0, str(ROOT / "src" / "kaggle_e2"))
print("ROOT =", ROOT)'''

LICENCIA = ("> **Licencia FF++:** Kaggle sube este notebook a GitHub con sus salidas. Ninguna celda muestra caras; "
            "las figuras con caras van a `privado/` (output privado de Kaggle). Tablas y gráficos numéricos sí se muestran.")

# ------------------------------------------------------------------ 04
nb04 = [
md(f"""# 04 · Fase B completa — barrido de γ ∈ {{0, 1, 2, 5}} × semillas {{42, 43, 44}}
**Hipótesis H-B2 (escrita antes de correr):** la relación entre γ y la robustez cross-manipulation es no monótona; con γ = 5 el modelo ignora casi todo el train y el AUC in-domain cae. Además, este barrido dice si el resultado nulo de B-01/B-02 (Focal γ = 2 no mejora el cross-manipulation) se mantiene entre semillas.

**Protocolo:** idéntico a B-01/B-02 (D-06). Solo cambian γ y la semilla. Cada corrida guarda su JSON y `preds_<corrida>.csv.gz` (probabilidades por frame de test: solo IDs y números) en `resultados/runs/`.

**Inputs en Kaggle**
- `ffpp-crops-g8` (obligatorio).
- *(opcional)* el output del 03 o de una versión anterior de este 04: las corridas que ya tengan JSON y predicciones se copian y no se repiten. Así, si la sesión se corta, se agrega el output parcial como input y se vuelve a guardar la versión.

**Tiempo:** ~10 min por corrida en una T4 (unas 2 h para las 12). Se corre por orden de prioridad y se deja de lanzar corridas nuevas al acercarse a `MAX_HORAS`.

{LICENCIA}"""),
code(CLONE),
code('''import json, shutil, time, pandas as pd, torch
from train import Config, run
import kaggle_io as K, analysis as A
CROPS = K.find_crops()
MAN = K.find_manifest(CROPS)
if MAN is not None:
    conteo = K.check_counts(CROPS, MAN); print(conteo); assert conteo.ok.all(), "Los recortes no coinciden con el manifiesto"
OUT = K.WORK / "resultados"; RUNS = OUT / "runs"; CKPT = K.WORK / "checkpoints"
RUNS.mkdir(parents=True, exist_ok=True)
BASE = dict(crops_dir=str(CROPS), out_dir=str(RUNS), ckpt_dir=str(CKPT), n_train_pairs=100, n_val_pairs=30,
            n_test_pairs=None, frames_per_video=16, epochs=6, warmup_epochs=1, batch_size=32, lr=2e-4)
if os.environ.get("SMOKE"):  # prueba rápida en CPU con datos sintéticos
    BASE.update(n_train_pairs=4, n_val_pairs=2, n_test_pairs=2, frames_per_video=3, epochs=2, batch_size=8, img_size=96, num_workers=0)
# orden de prioridad: primero confirmar BCE vs γ=2 con más semillas, luego γ=1 y γ=5
PRIORIDAD = [(0, 42), (2, 42), (0, 43), (2, 43), (0, 44), (2, 44), (1, 42), (5, 42), (1, 43), (5, 43), (1, 44), (5, 44)]
if os.environ.get("SMOKE"):
    PRIORIDAD = [(0, 42), (2, 42), (0, 43), (2, 43)]
CKPT_GUARDAR = {"bce_s42", "focal_g2_s42"}   # los que usa el notebook 06; el resto se borra para aligerar el output
MAX_HORAS = float(os.environ.get("MAX_HORAS", 10.5))   # el commit de Kaggle se corta a las 12 h
nombre = lambda g, s: "bce_s%d" % s if g == 0 else "focal_g%g_s%d" % (g, s)
print(K.snapshot_env(ROOT, OUT / "entorno.json"))'''),
md("## 1. Reutilizar corridas ya hechas (inputs opcionales)"),
code('''reusadas = []
if Path("/kaggle/input").exists():
    for js in Path("/kaggle/input").rglob("*.json"):
        g, s = A.parse_run_name(js.stem)
        preds = js.parent / f"preds_{js.stem}.csv.gz"
        if g is None or not preds.exists() or (RUNS / js.name).exists():
            continue
        info = json.loads(js.read_text())
        if info.get("data", {}).get("train", {}).get("videos_faltantes", 0) > 0 and not os.environ.get("SMOKE"):
            print(f"[{js.stem}] no se reutiliza: entrenó sin todos sus videos (corrida anterior a data_seed)"); continue
        shutil.copy(js, RUNS / js.name); shutil.copy(preds, RUNS / preds.name); reusadas.append(js.stem)
        ck = next(Path("/kaggle/input").rglob(f"{js.stem}.pt"), None)
        if js.stem in CKPT_GUARDAR and ck is not None:
            CKPT.mkdir(parents=True, exist_ok=True); shutil.copy(ck, CKPT / ck.name)
print("reutilizadas:", sorted(reusadas) or "ninguna")'''),
md("## 2. Barrido"),
code('''t0 = time.time()
for g, s in PRIORIDAD:
    name = nombre(g, s)
    if (RUNS / f"{name}.json").exists() and (RUNS / f"preds_{name}.csv.gz").exists():
        print(f"[{name}] ya existe, se salta"); continue
    horas = (time.time() - t0) / 3600
    if horas > MAX_HORAS - 0.3:
        print(f"Se detiene antes de {name}: {horas:.1f} h usadas (MAX_HORAS={MAX_HORAS}). Agregar este output como input y volver a guardar."); break
    r = run(Config(run_name=name, gamma=float(g), seed=s, data_seed=42, **BASE), verbose=True)
    assert os.environ.get("SMOKE") or r["data"]["train"]["videos_faltantes"] == 0, f"{name}: faltan videos de train"
    if name not in CKPT_GUARDAR:
        (CKPT / f"{name}.pt").unlink(missing_ok=True)
print(f"total: {(time.time() - t0) / 60:.0f} min")'''),
md("## 3. Resumen (sin caras)"),
code('''runs = A.load_runs(RUNS)
tabla = A.metrics_table(runs); tabla.to_csv(OUT / "barrido_metricas.csv", index=False)
resumen = A.summary_by_gamma(tabla, "auc_video"); resumen.to_csv(OUT / "barrido_resumen_auc_video.csv", index=False)
print(f"{len(runs)} corridas:", sorted(runs))
resumen.pivot(index="test", columns="gamma", values="texto")'''),
md("""Para llevar los resultados al repo: descargar `resultados/` del output (JSON, `preds_*.csv.gz`, CSV, `entorno.json`, `pip_freeze.txt`) y copiarlos a `results/metrics/`. Los checkpoints quedan en Kaggle; el notebook 06 usa este output como input."""),
]

# ------------------------------------------------------------------ 05
nb05 = [
md(f"""# 05 · Análisis final — tablas, intervalos de confianza y figuras
Junta todas las corridas (B-01/B-02 y el barrido del 04) y produce lo que va al informe y al póster:

1. Tabla final de AUC por video (media ± desviación entre semillas) por γ y manipulación, con la caída relativa frente a in-domain.
2. **Δ AUC (γ − BCE) con IC 95 % por bootstrap emparejado de videos.** Así se separa un efecto real de la variación entre semillas y entre videos de test.
3. Curvas del % de ejemplos fáciles (p_t > 0.9) por época y γ: el diagnóstico del mecanismo.

**No necesita GPU ni datos de FF++**: solo los JSON y las predicciones (IDs y probabilidades). Busca corridas en `/kaggle/input` (output del 03 y del 04) y en `results/metrics/` del repo. También corre en local, desde `notebooks/kaggle/`.

Sin predicciones (por ejemplo, B-01/B-02 transcritos de la v5), la corrida entra en la tabla pero no en el bootstrap."""),
code(CLONE),
code('''import numpy as np, pandas as pd, matplotlib.pyplot as plt
import analysis as A, plotstyle as S; S.apply()
fuentes = [ROOT / "results" / "metrics"] + [Path(d) for d in os.environ.get("RUNS_DIRS", "").split(os.pathsep) if d]
if Path("/kaggle/input").exists():
    fuentes += sorted({p.parent for p in Path("/kaggle/input").rglob("*.json") if A.parse_run_name(p.stem)[0] is not None})
runs = A.load_runs(fuentes)
OUT = Path("/kaggle/working/resultados") if Path("/kaggle").exists() else ROOT / "results" / "metrics" / "final"
OUT.mkdir(parents=True, exist_ok=True)
print(f"{len(runs)} corridas;", sum(r["_preds"] is not None for r in runs.values()), "con predicciones")
pd.DataFrame([{"run": n, "gamma": r["gamma"], "seed": r["seed"], "preds": r["_preds"] is not None} for n, r in runs.items()]).sort_values(["gamma", "seed"])'''),
md("## 1. Tabla final"),
code('''tabla = A.metrics_table(runs); tabla.to_csv(OUT / "tabla_metricas_por_corrida.csv", index=False)
res = A.summary_by_gamma(tabla, "auc_video"); res.to_csv(OUT / "tabla_final_auc_video.csv", index=False)
caida = A.summary_by_gamma(tabla, "caida_rel_video_pct"); caida.to_csv(OUT / "tabla_final_caida_rel_video.csv", index=False)
print("AUC por video (media ± desv. entre semillas)")
display(res.pivot(index="test", columns="gamma", values="texto"))
print("Caída relativa frente a in-domain (%)")
caida.pivot(index="test", columns="gamma", values="media").round(1)'''),
code('''tests = [t for t in A.TEST_ORDER if t in set(res.test)]
gammas = sorted(res.gamma.unique())
fig, ax = plt.subplots(figsize=(8.5, 4))
w = 0.8 / max(len(gammas), 1); x = np.arange(len(tests))
for i, (g, col) in enumerate(zip(gammas, S.SERIES)):
    sub = res[res.gamma == g].set_index("test").reindex(tests)
    ax.bar(x + (i - (len(gammas) - 1) / 2) * w, sub.media, w * 0.95, yerr=sub.desv.fillna(0), capsize=3, color=col,
           label=("BCE (γ=0)" if g == 0 else f"Focal γ={g:g}") + f" · n={int(sub.n_semillas.max())}")
ax.axhline(0.5, color=S.MUTED, lw=1, ls="--"); ax.text(len(tests) - 0.5, 0.51, "azar", color=S.MUTED, fontsize=8, ha="right")
ax.set_xticks(x, [t + ("\\n(in-domain)" if t == "Deepfakes" else "") for t in tests]); ax.set_ylim(0, 1.05)
ax.set_ylabel("AUC por video"); ax.grid(axis="x", visible=False)
ax.set_title("Entrenado solo en Deepfakes c23: AUC por manipulación de test", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(len(gammas), 4))
plt.tight_layout(); plt.savefig(OUT / "fig_final_auc_por_gamma.png", dpi=200); plt.show()'''),
md("## 2. ¿La diferencia con BCE es real? Bootstrap emparejado (2000 réplicas)"),
code('''boot = []
for g in [g for g in gammas if g != 0]:
    for t in tests:
        r = A.paired_bootstrap_delta(runs, g, 0.0, test=t, n_boot=int(os.environ.get("N_BOOT", 2000)))
        if r: boot.append(r)
boot = pd.DataFrame(boot)
if len(boot):
    boot.to_csv(OUT / "bootstrap_delta_auc_video.csv", index=False)
    display(boot.round(4))
    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.35 * len(boot)))
    lab = boot.apply(lambda r: f"γ={r.gamma_a:g} · {r.test} (n={r.semillas})", axis=1)
    ax.errorbar(boot.delta_auc_video, np.arange(len(boot)), xerr=[boot.delta_auc_video - boot.ic95_bajo, boot.ic95_alto - boot.delta_auc_video],
                fmt="o", color=S.SERIES[1], capsize=3)
    ax.axvline(0, color=S.MUTED, lw=1); ax.set_yticks(np.arange(len(boot)), lab); ax.invert_yaxis()
    ax.set_xlabel("Δ AUC por video frente a BCE (IC 95 %)"); ax.set_title("Efecto de Focal Loss sobre BCE", loc="left")
    plt.tight_layout(); plt.savefig(OUT / "fig_final_bootstrap_delta.png", dpi=200); plt.show()
else:
    print("No hay pares de corridas con predicciones (preds_*.csv.gz): correr el 03 o el 04 con la versión actual de train.py.")'''),
md("## 3. Mecanismo: % de ejemplos fáciles y AUC de validación por época"),
code('''hist = A.history_table(runs)
if len(hist):
    hist.to_csv(OUT / "historial_por_epoca.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for g, col in zip(gammas, S.SERIES):
        h = hist[hist.gamma == g].groupby("epoch")
        lab = "BCE (γ=0)" if g == 0 else f"Focal γ={g:g}"
        for ax, m in zip(axes, ["train_pct_faciles", "val_auc_frame"]):
            mu, sd = h[m].mean(), h[m].std(ddof=1).fillna(0)
            ax.plot(mu.index, mu, marker="o", ms=5, color=col, label=lab); ax.fill_between(mu.index, mu - sd, mu + sd, color=col, alpha=0.15)
    axes[0].set_title("% de ejemplos fáciles en train (p_t > 0.9)", loc="left"); axes[1].set_title("AUC de validación (Deepfakes)", loc="left")
    for a in axes: a.set_xlabel("época"); a.axvline(1.5, color=S.GRID, lw=1); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "fig_final_mecanismo.png", dpi=200); plt.show()'''),
md("""## 4. Lectura (completar con los números de arriba)
| Pregunta | Evidencia | Conclusión |
|---|---|---|
| ¿Focal γ=2 mejora el cross-manipulation frente a BCE? | Δ AUC e IC 95 % de la sección 2 | |
| ¿La relación con γ es no monótona (H-B2)? ¿γ=5 baja el in-domain? | Tabla de la sección 1 | |
| ¿El mecanismo ocurre (menos fáciles con γ mayor)? | Sección 3 | |
| ¿FaceSwap < 0.5 se repite entre semillas y pérdidas? | Tabla de la sección 1 | Diagnóstico en el notebook 06 |

Regla: si el IC 95 % incluye 0, no se afirma un efecto, se reporta como nulo."""),
]

# ------------------------------------------------------------------ 06
nb06 = [
md(f"""# 06 · Fase C — diagnóstico Grad-CAM: ¿dónde mira cada modelo y por qué falla en FaceSwap?
**Pregunta:** el detector entrenado en Deepfakes puntúa los FaceSwap como *más reales* que los originales (AUC < 0.5). ¿En qué regiones del recorte se apoya cada modelo (BCE vs. Focal γ = 2), cuando acierta y cuando falla?

**Hipótesis (escrita antes de correr):** el modelo se apoya en el contorno de la cara, donde está el borde de mezcla característico del generador Deepfakes. En FaceSwap ese artefacto es distinto, así que la evidencia de «falso» desaparece. Predicción: la masa de Grad-CAM en el anillo del **borde** es mayor en Deepfakes bien detectados que en los FaceSwap no detectados, y Focal no la desplaza hacia el interior. *Qué la refutaría:* masas de borde iguales entre aciertos y errores, o que Focal mueva la masa al interior sin mejorar el AUC.

**Método (D-10):** capa `block12`; selección emparejada desde las predicciones de BCE, por manipulación: 8 reales bien clasificadas, 8 falsas bien clasificadas, 8 falsos negativos y 8 falsos positivos (frame central de cada video). Mismos frames para BCE y Focal. Las métricas por región (interior / borde / fuera, aproximación geométrica del recorte 1.3×) se calculan además sobre **todos** los videos de test (2 frames por video).

**Inputs:** `ffpp-crops-g8` y el output del 04 (o del 03), que tiene `checkpoints/bce_s42.pt`, `checkpoints/focal_g2_s42.pt` y `preds_bce_s42.csv.gz`.

{LICENCIA} Lo que se publica son los IDs de los videos elegidos y las tablas numéricas."""),
code(CLONE),
code('''import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from PIL import Image
import data as D, gradcam as G, teardown as T, analysis as A, kaggle_io as K, plotstyle as S; S.apply()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CROPS = K.find_crops()
OUT = K.WORK / "resultados"; OUT.mkdir(parents=True, exist_ok=True)
busca = [Path("/kaggle/input")] if Path("/kaggle/input").exists() else []
busca += [K.WORK, ROOT / "results"]
def encontrar(nombre):
    for b in busca:
        hit = sorted(Path(b).rglob(nombre)) if Path(b).exists() else []
        if hit: return hit[0]
    raise FileNotFoundError(f"No se encontró {nombre}: agregar el output del 04 (o del 03) como input")
RUNS = {"BCE (γ=0)": "bce_s42", "Focal (γ=2)": "focal_g2_s42"}
modelos = {}
for lab, name in RUNS.items():
    m = T.build_model(False, 1); m.load_state_dict(torch.load(encontrar(f"{name}.pt"), map_location="cpu")["model"])
    modelos[lab] = m.to(DEVICE).eval()
PREDS_REF = encontrar("preds_bce_s42.csv.gz")
LAYER, N_POR_GRUPO, FRAMES_POR_VIDEO = "block12", 8, 2
if os.environ.get("SMOKE"):
    N_POR_GRUPO, FRAMES_POR_VIDEO = 2, 1
tf = D.eval_transform(299 if not os.environ.get("SMOKE") else 96)
print(DEVICE, PREDS_REF.name)'''),
code('''def frames_de(video_key, k):
    """k frames equiespaciados del video (clave 'FaceSwap/071_054'); devuelve rutas."""
    fs = sorted((CROPS / video_key).glob("*.png"))
    if not fs: return []
    idx = np.linspace(0, len(fs) - 1, k + 2)[1:-1].round().astype(int) if k > 1 else [len(fs) // 2]
    return [fs[i] for i in idx]

def cams_de(model, paths, bs=16):
    cams, probs = [], []
    for i in range(0, len(paths), bs):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to(DEVICE)
        c, p = G.gradcam(model, x, LAYER); cams.append(c); probs.append(p)
    return np.concatenate(cams), np.concatenate(probs)'''),
md("## 1. Selección emparejada y grillas privadas"),
caras('''seleccion = []
for t in A.TEST_ORDER:
    sel = A.select_cases(PREDS_REF, t, n=N_POR_GRUPO).assign(test=t)
    seleccion.append(sel)
    paths = [frames_de(v, 1)[0] for v in sel.video]
    imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])
    fig, axes = plt.subplots(len(sel), 3, figsize=(6.6, 2.2 * len(sel)), squeeze=False)
    for j, (lab, m) in enumerate(modelos.items()):
        cams, probs = cams_de(m, paths)
        for i in range(len(sel)):
            axes[i, 0].imshow(G.denorm(imgs[i])); axes[i, 0].set_title(f"{sel.grupo.iloc[i]} · {sel.video.iloc[i]}", fontsize=6, loc="left")
            axes[i, j + 1].imshow(G.denorm(imgs[i])); axes[i, j + 1].imshow(cams[i], cmap="jet", alpha=0.4)
            axes[i, j + 1].set_title(f"{lab} p={probs[i]:.2f}", fontsize=6, loc="left")
    for a in axes.flat: a.axis("off")
    K.save_private(fig, f"fig_gradcam_seleccion_{t}.png", dpi=110)
seleccion = pd.concat(seleccion, ignore_index=True)
seleccion.to_csv(OUT / "gradcam_seleccion_ids.csv", index=False)   # solo IDs y probabilidades: se puede publicar'''),
code('''# casos elegidos por manipulación y grupo (solo conteos; las grillas están en privado/)
seleccion.groupby(["test", "grupo"]).size().unstack(fill_value=0)'''),
md("## 2. Métricas por región sobre todos los videos de test (sin caras)"),
code('''vp = A.video_preds(PREDS_REF)
filas = []
for t in A.TEST_ORDER:
    v = vp[vp.test == t]
    paths, meta = [], []
    for _, r in v.iterrows():
        for p in frames_de(r.video, FRAMES_POR_VIDEO):
            paths.append(p); meta.append({"test": t, "video": r.video, "label": r.label})
    meta = pd.DataFrame(meta)
    for lab, m in modelos.items():
        cams, probs = cams_de(m, paths)
        st = A.cam_region_stats(cams)
        filas.append(pd.concat([meta, st], axis=1).assign(modelo=lab, p_fake=probs))
regiones = pd.concat(filas, ignore_index=True)
regiones["acierto"] = (regiones.p_fake >= 0.5) == (regiones.label == 1)
regiones["clase"] = np.where(regiones.label == 1, "falso", "real")
regiones.to_csv(OUT / "gradcam_regiones_por_frame.csv.gz", index=False)
res = (regiones.groupby(["modelo", "test", "clase", "acierto"])[["masa_interior", "masa_borde", "masa_fuera", "area_50"]]
       .mean().round(3)); res.to_csv(OUT / "gradcam_regiones_resumen.csv"); res'''),
code('''falsos = regiones[regiones.clase == "falso"].groupby(["test", "modelo"], observed=True)["masa_borde"].mean().unstack()
falsos = falsos.reindex([t for t in A.TEST_ORDER if t in falsos.index])
ax = falsos.plot.bar(figsize=(8, 3.6), color=S.SERIES[:falsos.shape[1]], width=0.75, rot=0)
ax.set_ylabel("fracción de la masa Grad-CAM"); ax.set_xlabel("")
ax.set_title("Masa de Grad-CAM en el anillo del borde de la cara (videos falsos)", loc="left"); ax.grid(axis="x", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
plt.tight_layout(); plt.savefig(OUT / "fig_gradcam_masa_borde.png", dpi=200); plt.show()'''),
md("""### 3. Contraste con la hipótesis
Resultados de la v2 (07/11/2026; `results/metrics/gradcam_masa_por_region.csv`), 2 frames por video de test, regiones concéntricas aproximadas.

| Predicción | Resultado | ¿Se cumplió? |
|---|---|---|
| Masa en el borde: Deepfakes bien detectados > FaceSwap no detectados | BCE: 0.126 (Deepfakes detectados) vs. 0.319 (FaceSwap no detectados) | No (al revés) |
| Focal no desplaza la masa hacia el interior | Falsos detectados en Deepfakes: interior 0.946 (Focal) vs. 0.866 (BCE) | No: Focal la concentra aún más en el interior |

**Lectura.** El patrón no es «borde de mezcla»: cuando el modelo dice *falso* (falsos detectados y también reales mal clasificados), la masa está en el **interior** de la cara (0.67–0.95 según manipulación y pérdida, con `area_50` ≈ 0.04–0.07, es decir, muy concentrada); cuando dice *real* (reales bien clasificados y falsos no detectados, incluido FaceSwap), la masa se va sobre todo **fuera** de la cara (0.31–0.62) y se dispersa (`area_50` ≈ 0.22–0.28). Una lectura compatible: el detector aprendió una evidencia del interior propia de Deepfakes; FaceSwap no la presenta y el modelo responde *real*. En la tabla de regiones no aparece ningún video FaceSwap acertado con Focal.

Límites: las regiones son concéntricas y aproximadas (no hay segmentación de la cara), es una sola semilla y Grad-CAM muestra correlación, no causalidad."""),
]

for name, cells, meta in [("04_barrido_gamma_semillas.ipynb", nb04, META_GPU), ("05_analisis_final.ipynb", nb05, META_CPU),
                          ("06_diagnostico_gradcam.ipynb", nb06, META_GPU)]:
    # `python tools/build_nb04_06.py 04` regenera solo el 04 (no borra las salidas de Kaggle de los demás)
    if len(sys.argv) > 1 and not any(name.startswith(a) for a in sys.argv[1:]):
        continue
    nbf.write(nbf.v4.new_notebook(cells=cells, metadata=meta), NB / name)
print("ok")

"""Genera el notebook de Kaggle 08 (Fase D, predicción 4 de H-D1): Grad-CAM por regiones de R0–R3.

Es el notebook 06 adaptado a los checkpoints del 07. El 06 no se regenera: conserva las salidas de su
versión 2, que respaldan C-01.
"""
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
BRANCH = os.environ.get("REPO_BRANCH", "feature/sbi-ajustes")   # cambiar a "main" cuando se haga el merge
ROOT = Path("/kaggle/working/repo") if Path("/kaggle").exists() else Path.cwd().parents[1]
if Path("/kaggle").exists() and not ROOT.exists():
    subprocess.run(["git", "clone", "-q", "-b", BRANCH, REPO_URL, str(ROOT)], check=True)
os.chdir(ROOT); sys.path.insert(0, str(ROOT / "src" / "kaggle_e2"))
print("ROOT =", ROOT, "| rama:", subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True).stdout.strip())'''

LICENCIA = ("> **Licencia FF++:** Kaggle sube este notebook a GitHub con sus salidas. Ninguna celda muestra caras; "
            "las figuras con caras van a `privado/` (output privado de Kaggle). Tablas y gráficos numéricos sí se muestran.")

nb08 = [
md(f"""# 08 · Fase D — Grad-CAM de R0–R3: ¿los SBI llevan la atención al borde de la mezcla?
Es el diagnóstico del notebook 06 aplicado a los modelos de la Fase D (notebook 07, semilla 42). El 06 queda como está: sus salidas respaldan C-01.

**Predicción 4 de H-D1 (escrita antes de correr el 07, D-14):** con el Grad-CAM del notebook 06 sobre el checkpoint **R1**, la masa en el anillo del **borde** de la cara sube en los falsos frente a R0. Los SBI solo tienen el artefacto de mezcla, así que el modelo debería aprender a mirarlo.

*Qué la refutaría:* que la masa de borde en los falsos de R1 sea igual o menor que en R0 (IC 95 % del Δ que incluya 0 o sea negativo). En ese caso la mejora en FaceSwap y Face2Face se explicaría por otra señal (por ejemplo, del interior o de frecuencia) y habría que decirlo.

**Complementos (descriptivos, no son predicciones):** lo mismo con R2 y R3; y en FaceSwap, dónde mira R3 en los videos que ahora detecta frente a los que sigue sin detectar.

**Método:** igual que el 06 (D-10): capa `block12`, regiones concéntricas aproximadas del recorte 1.3× (interior / borde / fuera), 2 frames por video de test. Δ de masa de borde por video (R1 − R0, falsos), con bootstrap pareado por video (2000 réplicas). La selección emparejada para las grillas privadas sale de las predicciones de R0.

**Inputs:** `ffpp-crops-g8` y el output del notebook 07 (`checkpoints/*_s42.pt` y `resultados/runs/preds_r0_bce_s42.csv.gz`).

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
    raise FileNotFoundError(f"No se encontró {nombre}: agregar el output del notebook 07 como input")
# checkpoints que guarda el 07 (CKPT_GUARDAR: semilla 42 de cada variante)
RUNS = {"R0 (BCE)": "r0_bce_s42", "R1 (+SBI)": "r1_sbi_bce_s42",
        "R2 (+Focal)": "r2_sbi_focal_g2_s42", "R3 (+JPEG/blur)": "r3_sbi_focal_g2_aug_s42"}
modelos = {}
for lab, name in RUNS.items():
    m = T.build_model(False, 1); m.load_state_dict(torch.load(encontrar(f"{name}.pt"), map_location="cpu")["model"])
    modelos[lab] = m.to(DEVICE).eval()
PREDS_REF = encontrar("preds_r0_bce_s42.csv.gz")   # R0 = referencia de la selección y de los videos
LAYER, N_POR_GRUPO, FRAMES_POR_VIDEO, N_BOOT = "block12", 8, 2, 2000
if os.environ.get("SMOKE"):
    N_POR_GRUPO, FRAMES_POR_VIDEO, N_BOOT = 2, 1, 200
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
    if sel.empty: continue
    paths = [frames_de(v, 1)[0] for v in sel.video]
    imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])
    fig, axes = plt.subplots(len(sel), len(modelos) + 1, figsize=(2.2 * (len(modelos) + 1), 2.2 * len(sel)), squeeze=False)
    for j, (lab, m) in enumerate(modelos.items()):
        cams, probs = cams_de(m, paths)
        for i in range(len(sel)):
            axes[i, 0].imshow(G.denorm(imgs[i])); axes[i, 0].set_title(f"{sel.grupo.iloc[i]} · {sel.video.iloc[i]}", fontsize=6, loc="left")
            axes[i, j + 1].imshow(G.denorm(imgs[i])); axes[i, j + 1].imshow(cams[i], cmap="jet", alpha=0.4)
            axes[i, j + 1].set_title(f"{lab} p={probs[i]:.2f}", fontsize=6, loc="left")
    for a in axes.flat: a.axis("off")
    K.save_private(fig, f"fig_gradcam_fase_d_{t}.png", dpi=110)
seleccion = pd.concat(seleccion, ignore_index=True)
seleccion.to_csv(OUT / "gradcam_fase_d_seleccion_ids.csv", index=False)   # solo IDs y probabilidades: se puede publicar'''),
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
regiones["modelo"] = pd.Categorical(regiones.modelo, categories=list(RUNS), ordered=True)
regiones["acierto"] = (regiones.p_fake >= 0.5) == (regiones.label == 1)
regiones["clase"] = np.where(regiones.label == 1, "falso", "real")
regiones.to_csv(OUT / "gradcam_fase_d_regiones_por_frame.csv.gz", index=False)
res = (regiones.groupby(["modelo", "test", "clase", "acierto"], observed=True)[["masa_interior", "masa_borde", "masa_fuera", "area_50"]]
       .mean().round(3)); res.to_csv(OUT / "gradcam_fase_d_regiones_resumen.csv"); res'''),
md("## 3. Predicción 4: masa de borde en los falsos, R1 frente a R0 (y R2, R3)"),
code('''# media por video (falsos), luego Δ pareado por video frente a R0 con bootstrap
por_video = (regiones[regiones.clase == "falso"]
             .groupby(["modelo", "test", "video"], observed=True)[["masa_borde", "masa_interior", "masa_fuera"]].mean())
rng = np.random.default_rng(0)
filas = []
for lab in list(RUNS)[1:]:
    for t in A.TEST_ORDER:
        try:
            a, b = por_video.loc[(lab, t)], por_video.loc[("R0 (BCE)", t)]
        except KeyError:
            continue
        comunes = a.index.intersection(b.index)
        for col in ("masa_borde", "masa_interior", "masa_fuera"):
            d = (a.loc[comunes, col] - b.loc[comunes, col]).to_numpy()
            boot = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            filas.append({"modelo": lab, "test": t, "region": col, "videos": len(d), "media_R0": b.loc[comunes, col].mean(),
                          "media_modelo": a.loc[comunes, col].mean(), "delta": d.mean(), "ic95_bajo": lo, "ic95_alto": hi,
                          "excluye_0": bool(lo > 0 or hi < 0)})
delta = pd.DataFrame(filas)
delta.to_csv(OUT / "gradcam_fase_d_delta_vs_r0.csv", index=False)
delta[delta.region == "masa_borde"].round(3)'''),
code('''borde = (regiones[regiones.clase == "falso"].groupby(["test", "modelo"], observed=True)["masa_borde"].mean().unstack())
borde = borde.reindex([t for t in A.TEST_ORDER if t in borde.index])
ax = borde.plot.bar(figsize=(8, 3.6), color=S.SERIES[:borde.shape[1]], width=0.8, rot=0)
ax.set_ylabel("fracción de la masa Grad-CAM"); ax.set_xlabel("")
ax.set_title("Masa de Grad-CAM en el anillo del borde de la cara (videos falsos), R0–R3", loc="left"); ax.grid(axis="x", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
plt.tight_layout(); plt.savefig(OUT / "fig_gradcam_fase_d_masa_borde.png", dpi=200); plt.show()'''),
md("## 4. FaceSwap con R3: videos detectados frente a no detectados"),
code('''fs = regiones[(regiones.test == "FaceSwap") & (regiones.clase == "falso") & (regiones.modelo == "R3 (+JPEG/blur)")]
fs.groupby("acierto")[["masa_interior", "masa_borde", "masa_fuera", "area_50"]].agg(["mean", "count"]).round(3)'''),
md("""## 5. Contraste con la predicción 4 de H-D1
*Completar después de correr, con la tabla de la sección 3 (R1 frente a R0, masa de borde en falsos) y registrarlo como C-02 en `docs/DECISIONS.md`.*

| Predicción | Resultado | ¿Se cumplió? |
|---|---|---|
| Con R1, la masa de borde sube en los falsos frente a R0 (IC 95 % del Δ por encima de 0) | *pendiente* | *pendiente* |

Límites (los mismos del 06): regiones concéntricas aproximadas, una sola semilla y Grad-CAM muestra correlación, no causalidad.

Para llevar los resultados al repo: descargar `resultados/` del output (`gradcam_fase_d_*.csv`, `fig_gradcam_fase_d_masa_borde.png`) a `results/metrics/fase_d/`. La carpeta `privado/` no se descarga."""),
]

nbf.write(nbf.v4.new_notebook(cells=nb08, metadata=META_GPU), NB / "08_gradcam_fase_d.ipynb")
print("ok")

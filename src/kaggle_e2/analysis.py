"""Análisis final de las corridas (notebook 05) y selección/medición Grad-CAM (notebook 06).

Solo usa métricas y predicciones (IDs de video y probabilidades), nunca imágenes: todo lo que
produce este módulo se puede publicar en GitHub.

Corridas: `<dir>/<run_name>.json` (salida de train.run) y, si existe, `<dir>/preds_<run_name>.csv.gz`
con una fila por frame de test: test, video ("Deepfakes/071_054" u "original/071"), frame, label, p_fake.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TEST_ORDER = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
_RUN_RE = re.compile(r"^(?P<loss>bce|focal_g(?P<g>[\d.]+))_s(?P<seed>\d+)$")


# ------------------------------------------------------------------ carga
def parse_run_name(name):
    """'bce_s42' → (0.0, 42); 'focal_g2_s43' → (2.0, 43)."""
    m = _RUN_RE.match(name)
    if not m:
        return None, None
    return (0.0 if m["loss"] == "bce" else float(m["g"])), int(m["seed"])


def load_runs(dirs):
    """Lee todos los JSON de corridas en `dirs` (rutas o lista). Devuelve {run_name: dict}; la última gana."""
    dirs = [dirs] if isinstance(dirs, (str, Path)) else dirs
    runs = {}
    for d in dirs:
        for p in sorted(Path(d).glob("*.json")):
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            name = r.get("run_name") or r.get("config", {}).get("run_name") or p.stem
            if "test" not in r:
                continue
            gamma = r.get("gamma", r.get("config", {}).get("gamma"))
            g2, seed = parse_run_name(name)
            r.update(run_name=name, gamma=float(gamma if gamma is not None else g2), seed=r.get("config", {}).get("seed", seed),
                     variant=r.get("config", {}).get("variant", "B"))
            r["grupo"] = group_label(r)
            preds = p.parent / f"preds_{name}.csv.gz"
            r["_preds"] = preds if preds.exists() else None
            runs[name] = r
    return runs


def group_label(r):
    """Etiqueta del grupo de corridas que se promedia entre semillas.
    Fase B: 'BCE' o 'Focal γ=2'. Fase D (D-14): la variante del notebook 07 ('R0', 'R1', 'R2', 'R3')."""
    v = r.get("variant") or r.get("config", {}).get("variant", "B")
    if v != "B":
        return v
    return "BCE" if r["gamma"] == 0 else f"Focal γ={r['gamma']:g}"


def metrics_table(runs):
    """Una fila por (corrida, test) con AUC por frame/video y caída relativa frente a in-domain."""
    rows = []
    for name, r in runs.items():
        indom = r["test"].get("Deepfakes", {})
        for t, v in r["test"].items():
            if "auc_frame" not in v:
                continue
            row = {"run": name, "grupo": r.get("grupo", group_label(r)), "gamma": r["gamma"], "seed": r["seed"], "test": t,
                   "auc_frame": v.get("auc_frame"), "auc_video": v.get("auc_video")}
            for lvl in ("frame", "video"):
                a_in, a = indom.get(f"auc_{lvl}"), v.get(f"auc_{lvl}")
                row[f"caida_rel_{lvl}_pct"] = 100 * (a_in - a) / a_in if a_in and a is not None else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df["test"] = pd.Categorical(df.test, TEST_ORDER + sorted(set(df.test) - set(TEST_ORDER)), ordered=True)
    return df.sort_values(["gamma", "seed", "test"]).reset_index(drop=True)


def summary_by_gamma(table, metric="auc_video"):
    """Media ± desviación estándar entre semillas, por γ y conjunto de test."""
    g = table.groupby(["gamma", "test"], observed=True)[metric]
    out = pd.DataFrame({"media": g.mean(), "desv": g.std(ddof=1), "n_semillas": g.count()}).reset_index()
    out["texto"] = out.apply(lambda r: f"{r.media:.3f}" + (f" ± {r.desv:.3f}" if r.n_semillas > 1 else " (1 semilla)"), axis=1)
    return out


def summary_by_group(table, metric="auc_video", order=None):
    """Como summary_by_gamma, pero por grupo de corridas (sirve para comparar las variantes R0–R3 de la Fase D)."""
    g = table.groupby(["grupo", "test"], observed=True)[metric]
    out = pd.DataFrame({"media": g.mean(), "desv": g.std(ddof=1), "n_semillas": g.count()}).reset_index()
    out["texto"] = out.apply(lambda r: f"{r.media:.3f}" + (f" ± {r.desv:.3f}" if r.n_semillas > 1 else " (1 semilla)"), axis=1)
    if order:
        out["grupo"] = pd.Categorical(out.grupo, [o for o in order if o in set(out.grupo)], ordered=True)
        out = out.sort_values(["grupo", "test"]).reset_index(drop=True)
    return out


def history_table(runs):
    rows = []
    for name, r in runs.items():
        for h in r.get("history", []):
            rows.append({"run": name, "grupo": r.get("grupo", group_label(r)), "gamma": r["gamma"], "seed": r["seed"], **h})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ predicciones y bootstrap
def video_preds(preds_path):
    """Probabilidad media por video y conjunto de test (nivel video de D-09)."""
    df = pd.read_csv(preds_path)
    return df.groupby(["test", "video"], as_index=False).agg(label=("label", "first"), p_fake=("p_fake", "mean"))


def _auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def paired_bootstrap_delta(runs, gamma_a, gamma_b=0.0, test="FaceSwap", n_boot=2000, seed=0):
    """IC 95 % de Δ AUC por video = AUC(γ_a) − AUC(γ_b), promediado sobre las semillas que tienen ambas corridas.

    En cada réplica se remuestrean los videos (con reemplazo, estratificado por clase) una sola vez y se
    usa la misma muestra para todas las semillas y ambas pérdidas, así el IC refleja la variabilidad
    de los datos de test y no solo la de la inicialización.
    """
    by = {}
    for r in runs.values():
        if r["_preds"] is not None and r.get("variant", "B") == "B":   # solo la Fase B: γ identifica la corrida
            by[(r["gamma"], r["seed"])] = r["_preds"]
    a = {s: p for (g, s), p in by.items() if g == gamma_a}
    b = {s: p for (g, s), p in by.items() if g == gamma_b}
    out = _paired_bootstrap(a, b, test, n_boot, seed)
    return None if out is None else {"test": test, "gamma_a": gamma_a, "gamma_b": gamma_b, **out}


def paired_bootstrap_groups(runs, group_a, group_b, test="FaceSwap", n_boot=2000, seed=0):
    """IC 95 % de Δ AUC por video = AUC(group_a) − AUC(group_b), emparejado por semilla (p. ej. 'R1' frente a 'R0')."""
    by = {}
    for r in runs.values():
        if r["_preds"] is not None:
            by.setdefault(r.get("grupo", group_label(r)), {})[r["seed"]] = r["_preds"]
    out = _paired_bootstrap(by.get(group_a, {}), by.get(group_b, {}), test, n_boot, seed)
    return None if out is None else {"test": test, "grupo_a": group_a, "grupo_b": group_b, **out}


def _paired_bootstrap(preds_a, preds_b, test, n_boot=2000, seed=0):
    """Núcleo del bootstrap pareado. preds_a / preds_b: {semilla: ruta de preds_*.csv.gz}."""
    seeds = sorted(set(preds_a) & set(preds_b))
    if not seeds:
        return None
    # por semilla: tabla emparejada (mismos videos en a y b)
    pares = {}
    for s in seeds:
        a, b = video_preds(preds_a[s]), video_preds(preds_b[s])
        a, b = a[a.test == test], b[b.test == test]
        m = a.merge(b, on=["video", "label"], suffixes=("_a", "_b"))
        if m.label.nunique() == 2:
            pares[s] = m.set_index("video")
    if not pares:
        return None
    comunes = sorted(set.intersection(*(set(m.index) for m in pares.values())))
    conjunto = len(comunes) > 0 and all(set(m.index) == set(comunes) for m in pares.values())

    def delta_seed(m, idx):
        y, pa, pb = m.label.to_numpy()[idx], m.p_fake_a.to_numpy()[idx], m.p_fake_b.to_numpy()[idx]
        return _auc(y, pa) - _auc(y, pb)

    def strat(rng, y):
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])

    rng = np.random.default_rng(seed)
    if conjunto:   # mismo test en todas las semillas (caso normal: test oficial completo) → misma muestra para todas
        pares = {s: m.loc[comunes] for s, m in pares.items()}
        y = next(iter(pares.values())).label.to_numpy()
        boots = [np.mean([delta_seed(m, idx) for m in pares.values()]) for idx in (strat(rng, y) for _ in range(n_boot))]
    else:          # test distinto por semilla (p. ej. SMOKE): se remuestrea cada semilla por separado
        boots = [np.mean([delta_seed(m, strat(rng, m.label.to_numpy())) for m in pares.values()]) for _ in range(n_boot)]
    obs = np.mean([delta_seed(m, np.arange(len(m))) for m in pares.values()])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"semillas": len(pares),
            "videos": int(np.mean([len(m) for m in pares.values()])), "delta_auc_video": obs,
            "ic95_bajo": lo, "ic95_alto": hi, "excluye_0": bool(lo > 0 or hi < 0)}


# ------------------------------------------------------------------ Grad-CAM (notebook 06)
def select_cases(preds_path, test, n=8, seed=0):
    """Selección D-10 a nivel video con las predicciones de UNA corrida (la de referencia, BCE):
    n reales bien clasificadas, n falsas bien clasificadas, n falsos negativos y n falsos positivos.
    Devuelve DataFrame (grupo, video, label, p_fake); si un grupo tiene menos de n, se toman los que haya."""
    v = video_preds(preds_path)
    v = v[v.test == test]
    grupos = {"real_ok": (v.label == 0) & (v.p_fake < 0.5), "falso_ok": (v.label == 1) & (v.p_fake >= 0.5),
              "falso_negativo": (v.label == 1) & (v.p_fake < 0.5), "falso_positivo": (v.label == 0) & (v.p_fake >= 0.5)}
    out = []
    for name, mask in grupos.items():
        sub = v[mask]
        # los más confiados de cada grupo: los casos más claros de acierto o de error
        sub = sub.sort_values("p_fake", ascending=name in ("real_ok", "falso_negativo")).head(n)
        out.append(sub.assign(grupo=name))
    return pd.concat(out, ignore_index=True)[["grupo", "video", "label", "p_fake"]]


def region_masks(size=299, inner=0.5, face=1 / 1.3):
    """Máscaras concéntricas del recorte (la caja de MTCNN se agrandó 1.3×, así que la cara ocupa ~77 % del lado).

    - interior: cuadrado central de lado `inner` (ojos, nariz, boca);
    - borde: anillo entre `inner` y `face` (contorno de la cara, donde está el borde de mezcla);
    - fuera: el resto (pelo, fondo, cuello).
    Es una aproximación geométrica, no una segmentación: se reporta como tal.
    """
    c = (np.arange(size) + 0.5) / size - 0.5
    d = np.maximum(np.abs(c)[:, None], np.abs(c)[None, :]) * 2   # distancia de Chebyshev normalizada (0 centro, 1 borde)
    return {"interior": d <= inner, "borde": (d > inner) & (d <= face), "fuera": d > face}


def cam_region_stats(cams, masks=None):
    """Fracción de la masa del mapa Grad-CAM en cada región, para un lote de mapas N×H×W en [0, 1]."""
    cams = np.asarray(cams, dtype=np.float64)
    masks = masks or region_masks(cams.shape[-1])
    tot = cams.sum(axis=(1, 2)) + 1e-12
    out = {f"masa_{k}": (cams * m).sum(axis=(1, 2)) / tot for k, m in masks.items()}
    out["area_50"] = np.array([_area_frac(c, 0.5) for c in cams])   # fracción del recorte que concentra el 50 % de la masa
    return pd.DataFrame(out)


def _area_frac(cam, frac):
    v = np.sort(cam.ravel())[::-1]
    c = np.cumsum(v)
    return float(np.searchsorted(c, frac * c[-1]) + 1) / v.size


__all__ = ["TEST_ORDER", "parse_run_name", "load_runs", "group_label", "metrics_table", "summary_by_gamma",
           "summary_by_group", "history_table", "video_preds", "paired_bootstrap_delta", "paired_bootstrap_groups", "select_cases", "region_masks", "cam_region_stats"]

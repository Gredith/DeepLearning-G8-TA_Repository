"""Fase A — Disección instrumentada de Xception (legacy_xception de timm).

Todas las funciones devuelven DataFrames o dicts para que el notebook
01_teardown_xception.ipynb los muestre y los guarde en results/teardown/.
"""
import copy
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

MODEL_NAME = "legacy_xception"  # Xception original (Chollet 2017), la usada en FaceForensics++

# Agrupación de los módulos de primer nivel en los "flows" del paper de Xception
FLOWS = {
    "stem": ["conv1", "bn1", "act1", "conv2", "bn2", "act2"],
    "entry": ["block1", "block2", "block3"],
    "middle": [f"block{i}" for i in range(4, 12)],
    "exit": ["block12", "conv3", "bn3", "act3", "conv4", "bn4", "act4"],
    "head": ["global_pool", "fc"],
}
BLOCK_ORDER = [n for flow in FLOWS.values() for n in flow
               if not n.startswith(("bn", "act", "global"))]


def build_model(pretrained=False, num_classes=1):
    return timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=num_classes)


def flow_of(name):
    for flow, names in FLOWS.items():
        if name in names:
            return flow
    return "?"


# ---------------------------------------------------------------- parámetros y FLOPs
def params_flops_per_block(model, input_size=299):
    """Parámetros, FLOPs y forma de salida de cada módulo de primer nivel.

    FLOPs = operaciones de punto flotante contadas por torch (1 MAC = 2 FLOPs).
    Solo cuenta conv/linear/matmul: BN y ReLU no se cuentan (son <1% del total).
    """
    model = model.eval()
    shapes = {}
    hooks = [m.register_forward_hook(lambda mod, i, o, n=n: shapes.__setitem__(n, tuple(o.shape[1:])))
             for n, m in model.named_children()]
    x = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    for h in hooks:
        h.remove()
    counts = fc.get_flop_counts()
    root = type(model).__name__
    rows = []
    for name, mod in model.named_children():
        if name.startswith(("bn", "act")):
            continue
        flops = sum(counts.get(f"{root}.{name}", {}).values())
        rows.append({
            "modulo": name,
            "flow": flow_of(name),
            "params": sum(p.numel() for p in mod.parameters()),
            "GFLOPs": flops / 1e9,
            "salida (C,H,W)": shapes.get(name),
        })
    df = pd.DataFrame(rows)
    df["% params"] = 100 * df["params"] / df["params"].sum()
    df["% FLOPs"] = 100 * df["GFLOPs"] / df["GFLOPs"].sum()
    return df


def per_flow(df):
    g = df.groupby("flow", sort=False)[["params", "GFLOPs", "% params", "% FLOPs"]].sum()
    return g.reindex([f for f in FLOWS if f in g.index])


# ---------------------------------------------------------------- separable vs estándar
def to_standard_conv(model):
    """Copia del modelo donde cada SeparableConv2d (depthwise 3x3 + pointwise 1x1)
    se reemplaza por una Conv2d estándar 3x3 con las mismas entradas/salidas."""
    m = copy.deepcopy(model)
    for parent in m.modules():
        for cname, child in list(parent.named_children()):
            if type(child).__name__ == "SeparableConv2d":
                dw, pw = child.conv1, child.pointwise
                setattr(parent, cname, nn.Conv2d(dw.in_channels, pw.out_channels, dw.kernel_size,
                                                 dw.stride, dw.padding, dw.dilation, bias=False))
    return m


def total_params_flops(model, input_size=299):
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model.eval()(torch.randn(1, 3, input_size, input_size))
    return sum(p.numel() for p in model.parameters()), fc.get_total_flops()


def separable_vs_standard(model, input_size=299):
    p_sep, f_sep = total_params_flops(model, input_size)
    std = to_standard_conv(model)
    p_std, f_std = total_params_flops(std, input_size)
    # verificación de que la red estándar sigue produciendo la misma forma de salida
    with torch.no_grad():
        assert std.eval()(torch.randn(1, 3, input_size, input_size)).shape == \
            model.eval()(torch.randn(1, 3, input_size, input_size)).shape
    return pd.DataFrame({
        "variante": ["Xception (separable)", "Misma red con conv estándar 3x3"],
        "params (M)": [p_sep / 1e6, p_std / 1e6],
        "GFLOPs": [f_sep / 1e9, f_std / 1e9],
    }).assign(**{"factor vs separable": lambda d: d["GFLOPs"] / d["GFLOPs"].iloc[0]})


# ---------------------------------------------------------------- campo receptivo teórico
def _kernel_stride_dil(m):
    k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
    s = m.stride if isinstance(m.stride, int) else m.stride[0]
    d = m.dilation if isinstance(m.dilation, int) else m.dilation[0]
    return k, s, d


def theoretical_receptive_field(model):
    """RF teórico por la rama principal (las ramas 'skip' 1x1 tienen RF menor
    y no lo amplían). r_out = r_in + (k-1)*d*j_in ; j_out = j_in*s."""
    r, j = 1, 1
    rows = []
    for top, mod in model.named_children():
        for name, leaf in mod.named_modules() if len(list(mod.children())) else [("", mod)]:
            if "skip" in name:
                continue
            if isinstance(leaf, (nn.Conv2d, nn.MaxPool2d)):
                k, s, d = _kernel_stride_dil(leaf)
                r = r + (k - 1) * d * j
                j = j * s
        if isinstance(mod, (nn.Conv2d, nn.Module)) and not top.startswith(("bn", "act", "global", "fc")):
            rows.append({"modulo": top, "flow": flow_of(top), "RF teórico (px)": r, "stride acumulado": j})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- campo receptivo efectivo
def effective_receptive_field(model, layer_names, input_size=299, n_samples=8, seed=0, inputs=None):
    """ERF (Luo et al., 2016): gradiente de la unidad central de la salida de cada
    capa respecto a la entrada, promediado en |grad| sobre varias entradas.
    Por defecto usa ruido gaussiano; con `inputs` (tensor N×3×H×W, p. ej. recortes
    de caras ya normalizados) mide el ERF sobre datos reales.
    Devuelve {capa: mapa 2D normalizado} y una tabla con el lado del cuadrado
    centrado que concentra el 95% / 50% de la masa del gradiente."""
    torch.manual_seed(seed)
    model = model.eval()
    feats = {}
    hooks = [dict(model.named_modules())[n].register_forward_hook(
        lambda mod, i, o, n=n: feats.__setitem__(n, o)) for n in layer_names]
    if inputs is None:
        inputs = torch.randn(n_samples, 3, input_size, input_size)
    input_size = inputs.shape[-1]
    maps = {n: np.zeros((input_size, input_size)) for n in layer_names}
    for k in range(len(inputs)):
        x = inputs[k:k + 1].clone().requires_grad_(True)
        model(x)
        for n in layer_names:
            f = feats[n]
            h, w = f.shape[-2:]
            g, = torch.autograd.grad(f[0, :, h // 2, w // 2].sum(), x, retain_graph=True)
            maps[n] += g.abs().sum(1)[0].numpy()
    for hk in hooks:
        hk.remove()
    rows = []
    for n, m in maps.items():
        m = m / m.sum()
        maps[n] = m
        rows.append({"capa": n, "lado ERF 95% (px)": _energy_side(m, 0.95),
                     "lado ERF 50% (px)": _energy_side(m, 0.50)})
    return maps, pd.DataFrame(rows)


def _energy_side(m, frac):
    """Lado del menor cuadrado centrado que contiene `frac` de la masa del mapa."""
    c = m.shape[0] // 2
    for half in range(0, c + 1):
        if m[c - half:c + half + 1, c - half:c + half + 1].sum() >= frac:
            return 2 * half + 1
    return m.shape[0]


# ---------------------------------------------------------------- latencia y memoria
def activation_memory_mb(model, input_size=299, batch=1):
    """Suma de los tensores de salida de cada capa hoja (fp32): aproxima lo que el
    entrenamiento guarda para el backward. Es una cota, no la medición del allocator."""
    total = [0]
    hooks = [m.register_forward_hook(lambda mod, i, o: total.__setitem__(0, total[0] + o.numel() * 4))
             for m in model.modules() if len(list(m.children())) == 0]
    with torch.no_grad():
        model.eval()(torch.randn(batch, 3, input_size, input_size))
    for h in hooks:
        h.remove()
    return total[0] / 2 ** 20


def latency_ms(model, input_size=299, batch=1, device="cpu", warmup=5, runs=30):
    model = model.eval().to(device)
    x = torch.randn(batch, 3, input_size, input_size, device=device)
    times = []
    with torch.no_grad():
        for i in range(warmup + runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - t) * 1000)
    return {"device": device, "batch": batch, "input": input_size,
            "mediana_ms": float(np.median(times)), "p90_ms": float(np.percentile(times, 90)),
            "ms_por_imagen": float(np.median(times)) / batch}


def gpu_peak_memory_mb(model, input_size=299, batch=32, train=True):
    """Pico de memoria GPU real (solo en Kaggle) para un paso forward(+backward)."""
    assert torch.cuda.is_available()
    model = model.cuda().train(train)
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(batch, 3, input_size, input_size, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(x)
    if train:
        out.float().mean().backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2 ** 20


# ---------------------------------------------------------------- activaciones intermedias
def intermediate_activations(model, x, layer_names):
    """Mapas de activación (media sobre canales) de varias capas para un batch x."""
    feats = {}
    hooks = [dict(model.named_modules())[n].register_forward_hook(
        lambda mod, i, o, n=n: feats.__setitem__(n, o.detach().mean(1).cpu())) for n in layer_names]
    with torch.no_grad():
        model.eval()(x)
    for h in hooks:
        h.remove()
    return feats

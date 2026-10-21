"""Grad-CAM (Selvaraju et al., 2017) para el logit 'falso' de Xception.

Capa objetivo por defecto: `block12` (salida 1024×10×10, la última con estructura
espacial antes de las separables finales). Se puede cambiar a 'conv4'.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def gradcam(model, x, layer="block12"):
    """x: N×3×H×W normalizado. Devuelve (mapas N×H×W en [0,1], prob_falso N)."""
    model = model.eval()
    store = {}
    h = dict(model.named_modules())[layer].register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    x = x.clone().requires_grad_(False)
    with torch.enable_grad():
        logits = model(x).view(-1)
        acts = store["a"]
        grads, = torch.autograd.grad(logits.sum(), acts)
    h.remove()
    w = grads.mean(dim=(2, 3), keepdim=True)                 # importancia por canal
    cam = F.relu((w * acts).sum(1, keepdim=True))            # N×1×h×w
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam - cam.flatten(1).min(1)[0][:, None, None]
    cam = cam / (cam.flatten(1).max(1)[0][:, None, None] + 1e-8)
    return cam.detach().cpu().numpy(), torch.sigmoid(logits).detach().cpu().numpy()


def denorm(x):
    return (x.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)


def grid(model, imgs: dict, layer="block12", title=""):
    """imgs: {etiqueta: tensor N×3×H×W}. Una fila por etiqueta con imagen + Grad-CAM superpuesto."""
    n = max(len(v) for v in imgs.values())
    fig, axes = plt.subplots(len(imgs), n, figsize=(2.6 * n, 2.8 * len(imgs)), squeeze=False)
    for r, (lab, x) in enumerate(imgs.items()):
        cams, probs = gradcam(model, x, layer)
        for c in range(n):
            ax = axes[r, c]; ax.axis("off")
            if c >= len(x):
                continue
            ax.imshow(denorm(x[c])); ax.imshow(cams[c], cmap="jet", alpha=0.4)
            ax.set_title(f"{lab} · p(falso)={probs[c]:.2f}", fontsize=8, loc="left")
    fig.suptitle(f"Grad-CAM ({layer}) {title}", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig

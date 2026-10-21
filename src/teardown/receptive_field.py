"""Analytical maximum receptive field through timm's legacy Xception blocks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import yaml


def pair(value) -> tuple[int, int]:
    return value if isinstance(value, tuple) else (value, value)


def advance_receptive_field(rf: int, jump: int, kernel, stride=1, padding=0, dilation=1) -> tuple[int, int]:
    """Apply one spatial operator to the receptive-field size and input jump."""
    kernel_h, _ = pair(kernel)
    stride_h, _ = pair(stride)
    dilation_h, _ = pair(dilation)
    return rf + (kernel_h - 1) * dilation_h * jump, jump * stride_h


def output_spatial_size(size: int, kernel, stride=1, padding=0, dilation=1) -> int:
    kernel_h, _ = pair(kernel)
    stride_h, _ = pair(stride)
    padding_h, _ = pair(padding)
    dilation_h, _ = pair(dilation)
    return math.floor((size + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1)


def analyze_module(module, rf: int, jump: int, spatial_size: int) -> tuple[int, int, int]:
    """Walk a serial module (the main branch) and update receptive field/size."""
    import torch.nn as nn

    for layer in module.modules():
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
            rf, jump = advance_receptive_field(
                rf,
                jump,
                layer.kernel_size,
                layer.stride if layer.stride is not None else layer.kernel_size,
                layer.padding,
                getattr(layer, "dilation", 1),
            )
            spatial_size = output_spatial_size(
                spatial_size,
                layer.kernel_size,
                layer.stride if layer.stride is not None else layer.kernel_size,
                layer.padding,
                getattr(layer, "dilation", 1),
            )
    return rf, jump, spatial_size


def compute_table(model, input_size: int) -> list[dict]:
    """Trace Xception stages and combine each residual block's maximum RF path."""
    import torch

    named_modules = dict(model.named_modules())
    stage_names = ["conv1", "conv2", *(f"block{i}" for i in range(1, 13)), "conv3", "conv4"]
    missing = [name for name in stage_names if name not in named_modules]
    if missing:
        raise ValueError(f"Model is not the expected Xception layout; missing {missing}")
    actual_shapes: dict[str, tuple[int, int]] = {}
    handles = []
    for name in stage_names:
        module = named_modules[name]
        handles.append(module.register_forward_hook(
            lambda _module, _inputs, output, stage=name: actual_shapes.__setitem__(stage, tuple(output.shape[-2:]))
        ))
    model.eval()
    with torch.inference_mode():
        model(torch.zeros(1, 3, input_size, input_size))
    for handle in handles:
        handle.remove()

    rows: list[dict] = [{
        "block": "input",
        "cumulative_stride": 1,
        "jump": 1,
        "receptive_field": 1,
        "output_resolution": f"{input_size}x{input_size}",
        "empirical_resolution": f"{input_size}x{input_size}",
    }]
    rf, jump, size = 1, 1, input_size
    for name in stage_names:
        module = named_modules[name]
        if name.startswith("block"):
            # timm Block has a serial ``rep`` main path and a 1x1/identity skip.
            rf_main, jump_main, size_main = analyze_module(module.rep, rf, jump, size)
            skip_rf = rf  # 1x1 or identity shortcut does not grow the field.
            rf, jump, size = max(rf_main, skip_rf), jump_main, size_main
        else:
            rf, jump, size = analyze_module(module, rf, jump, size)
        actual = actual_shapes.get(name, (size, size))
        rows.append({
            "block": name,
            "cumulative_stride": jump,
            "jump": jump,
            "receptive_field": rf,
            "output_resolution": f"{size}x{size}",
            "empirical_resolution": f"{actual[0]}x{actual[1]}",
        })
    return rows


def analyze(config: dict, output_path: Path) -> list[dict]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Receptive-field analysis requires torch and timm") from exc
    try:
        from ..models.xception import build_xception
    except ImportError:  # pragma: no cover - direct CLI invocation
        from models.xception import build_xception
    model_cfg = config.get("model", {})
    model = build_xception(pretrained=False, name=model_cfg.get("name", "legacy_xception"))
    input_size = int(config.get("data", {}).get("image_size", 299))
    rows = compute_table(model, input_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata_path = output_path.with_name("receptive_field_metadata.json")
    metadata_path.write_text(json.dumps({
        "model": model_cfg.get("name", "legacy_xception"),
        "input_size": input_size,
        "method": "maximum main-path receptive field per residual block; 1x1/identity shortcut keeps a shorter route",
        "resolution_check_passed": all(row["output_resolution"] == row["empirical_resolution"] for row in rows),
    }, indent=2) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", default="results/metrics/receptive_field.csv")
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    rows = analyze(config, Path(args.output))
    print(f"Wrote receptive-field table: {args.output} ({len(rows)} stages)")


if __name__ == "__main__":
    main()

"""Per-flow parameters, FLOPs, output shapes, and separable-convolution savings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

ENTRY_MODULES = ("conv1", "conv2", "block1", "block2", "block3")
MIDDLE_MODULES = tuple(f"block{i}" for i in range(4, 12))
EXIT_MODULES = ("block12", "conv3", "conv4")


def flow_for_module(name: str) -> str | None:
    top_level = name.split(".", 1)[0]
    if top_level in ENTRY_MODULES:
        return "entry_flow"
    if top_level in MIDDLE_MODULES:
        return "middle_flow"
    if top_level in EXIT_MODULES:
        return "exit_flow"
    return None


def analyze(config: dict, output_root: Path) -> tuple[Path, Path]:
    try:
        import torch
        from torch import nn
        from fvcore.nn import FlopCountAnalysis
    except ImportError as exc:
        raise RuntimeError("T4 requires torch, timm, and fvcore; install environment/requirements.txt") from exc

    try:
        from ..models.xception import build_xception
    except ImportError:  # pragma: no cover - direct CLI invocation
        from models.xception import build_xception

    model_cfg = config.get("model", {})
    model = build_xception(pretrained=False, name=model_cfg.get("name", "legacy_xception"))
    model.eval()
    image_size = int(config.get("data", {}).get("image_size", 299))
    dummy = torch.zeros(1, 3, image_size, image_size)
    shapes: dict[str, list[int]] = {}
    separable_shapes: dict[str, list[int]] = {}
    handles = []

    named_modules = dict(model.named_modules())
    required = set(ENTRY_MODULES + MIDDLE_MODULES + EXIT_MODULES)
    missing = sorted(required - named_modules.keys())
    if missing:
        raise ValueError(f"Xception layout differs from the expected legacy Xception; missing modules: {missing}")

    for name in required:
        module = named_modules[name]
        handles.append(module.register_forward_hook(
            lambda _module, _inputs, output, module_name=name: shapes.__setitem__(
                module_name, list(output.shape) if isinstance(output, torch.Tensor) else []
            )
        ))
    separable_modules: list[tuple[str, nn.Module, nn.Conv2d, nn.Conv2d]] = []
    for name, module in model.named_modules():
        convs = [(child_name, child) for child_name, child in module.named_modules() if child_name and isinstance(child, nn.Conv2d)]
        depthwise = [(child_name, child) for child_name, child in convs if child.groups == child.in_channels and child.in_channels > 1 and child.kernel_size != (1, 1)]
        pointwise = [(child_name, child) for child_name, child in convs if child.kernel_size == (1, 1) and child.groups == 1]
        if not depthwise or not pointwise or module.__class__.__name__.lower().find("separable") < 0:
            continue
        depthwise_conv = depthwise[0][1]
        pointwise_conv = pointwise[-1][1]
        separable_modules.append((name, module, depthwise_conv, pointwise_conv))
        handles.append(module.register_forward_hook(
            lambda _module, _inputs, output, module_name=name: separable_shapes.__setitem__(
                module_name, list(output.shape) if isinstance(output, torch.Tensor) else []
            )
        ))

    with torch.inference_mode():
        model(dummy)
    analysis = FlopCountAnalysis(model, dummy)
    flops_by_module = analysis.by_module()
    for handle in handles:
        handle.remove()

    group_rows = []
    total_params = sum(parameter.numel() for parameter in model.parameters())
    total_flops = int(flops_by_module.get("", 0))
    for group, modules in (("entry_flow", ENTRY_MODULES), ("middle_flow", MIDDLE_MODULES), ("exit_flow", EXIT_MODULES)):
        parameter_count = sum(sum(parameter.numel() for parameter in named_modules[name].parameters()) for name in modules)
        flop_count = sum(int(flops_by_module.get(name, 0)) for name in modules)
        shape = shapes.get(modules[-1], [])
        group_rows.append({
            "block": group,
            "parameters": parameter_count,
            "parameter_percent": 100.0 * parameter_count / total_params if total_params else 0.0,
            "flops": flop_count,
            "flops_percent": 100.0 * flop_count / total_flops if total_flops else 0.0,
            "output_shape": json.dumps(shape),
        })

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "teardown_flops.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)

    savings_rows = []
    for name, module, depthwise_conv, pointwise_conv in separable_modules:
        shape = separable_shapes.get(name, [])
        if len(shape) != 4:
            continue
        height, width = shape[-2:]
        cin = depthwise_conv.in_channels
        cout = pointwise_conv.out_channels
        kernel = depthwise_conv.kernel_size[0]
        separable_params = sum(parameter.numel() for parameter in module.parameters())
        standard_params = kernel * kernel * cin * cout
        separable_flops = height * width * (kernel * kernel * cin + cin * cout)
        standard_flops = height * width * kernel * kernel * cin * cout
        savings_rows.append({
            "module": name,
            "flow": flow_for_module(name),
            "cin": cin,
            "cout": cout,
            "kernel": kernel,
            "separable_parameters": separable_params,
            "standard_parameters_no_bias": standard_params,
            "parameter_savings_factor": standard_params / separable_params if separable_params else 0.0,
            "separable_flops_approx": separable_flops,
            "standard_flops_approx": standard_flops,
            "flops_savings_factor": standard_flops / separable_flops if separable_flops else 0.0,
            "output_shape": json.dumps(shape),
        })
    savings_path = output_root / "teardown_separable_savings.csv"
    fields = list(savings_rows[0]) if savings_rows else ["module", "flow", "cin", "cout", "kernel"]
    with savings_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(savings_rows)

    unsupported = sorted(str(operator) for operator in analysis.unsupported_ops())
    (output_root / "teardown_flops_metadata.json").write_text(json.dumps({
        "model": model_cfg.get("name", "legacy_xception"),
        "input_shape": list(dummy.shape),
        "total_parameters": total_params,
        "total_flops_fvcore": total_flops,
        "fvcore_unsupported_ops": unsupported,
        "separable_modules_found": len(savings_rows),
    }, indent=2) + "\n", encoding="utf-8")

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar([row["block"] for row in group_rows], [row["parameters"] for row in group_rows])
    axes[0].set_title("Parameters by flow")
    axes[0].set_ylabel("Parameters")
    axes[1].bar([row["block"] for row in group_rows], [row["flops"] for row in group_rows])
    axes[1].set_title("FLOPs by flow")
    axes[1].set_ylabel("FLOPs (fvcore)")
    figure.tight_layout()
    figure_path = output_root.parent / "figures" / "teardown_flops_by_flow.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return summary_path, savings_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output-dir", default="results/metrics")
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    summary, savings = analyze(config, Path(args.output_dir))
    print(f"Wrote {summary} and {savings}")


if __name__ == "__main__":
    main()

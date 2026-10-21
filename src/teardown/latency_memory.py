"""Measure Xception inference latency and peak inference/training GPU memory."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import yaml


def analyze(config: dict, output_path: Path, warmup: int = 20, iterations: int = 100) -> list[dict]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Latency/memory analysis requires torch and timm") from exc
    try:
        from ..models.xception import build_xception
    except ImportError:  # pragma: no cover - direct CLI invocation
        from models.xception import build_xception

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config.get("model", {})
    model = build_xception(pretrained=False, name=model_cfg.get("name", "legacy_xception")).to(device)
    image_size = int(config.get("data", {}).get("image_size", 299))
    training_cfg = config.get("training", {})
    learning_rate = float(training_cfg.get("learning_rate", 1e-4))
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    latency_rows = []
    memory_rows = []

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for batch_size in (1, 32):
        inputs = torch.randn(batch_size, 3, image_size, image_size, device=device)
        model.eval()
        with torch.inference_mode():
            for _ in range(warmup):
                model(inputs)
            synchronize()
            durations = []
            for _ in range(iterations):
                synchronize()
                start = time.perf_counter()
                model(inputs)
                synchronize()
                durations.append((time.perf_counter() - start) * 1000)
            inference_memory = None
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                model(inputs)
                synchronize()
                inference_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        latency_rows.append({
            "mode": "inference",
            "batch_size": batch_size,
            "warmup_iterations": warmup,
            "measured_iterations": iterations,
            "mean_latency_ms": statistics.mean(durations),
            "std_latency_ms": statistics.stdev(durations) if len(durations) > 1 else 0.0,
            "peak_memory_mb": inference_memory,
            "device": gpu_name,
        })

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        labels = torch.randint(0, 2, (batch_size,), device=device).float()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        train_memory_error = None
        try:
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs).reshape(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                loss.backward()
                optimizer.step()
            synchronize()
            training_memory = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else None
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            training_memory = None
            train_memory_error = str(exc).splitlines()[0]
            if device.type == "cuda":
                torch.cuda.empty_cache()
        memory_rows.append({
            "mode": "training_peak_memory",
            "batch_size": batch_size,
            "warmup_iterations": 3,
            "measured_iterations": 1,
            "mean_latency_ms": None,
            "std_latency_ms": None,
            "peak_memory_mb": training_memory,
            "device": gpu_name,
            "error": train_memory_error,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(latency_rows[0]))
        writer.writeheader()
        writer.writerows(latency_rows)
    memory_path = output_path.with_name("memory.csv")
    with memory_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(memory_rows[0]))
        writer.writeheader()
        writer.writerows(memory_rows)
    return latency_rows + memory_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", default="results/metrics/latency.csv")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 2:
        parser.error("--warmup must be >= 1 and --iterations must be >= 2")
    with Path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    rows = analyze(config, Path(args.output), args.warmup, args.iterations)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

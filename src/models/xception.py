"""Xception model factory."""

import timm


def build_xception(pretrained: bool = True, name: str = "legacy_xception"):
    """Build a timm Xception variant with one binary-classification logit."""
    available = timm.list_models("*xception*")
    if name not in available:
        raise ValueError(f"Unknown timm model {name!r}. Available Xception models: {available}")
    return timm.create_model(name, pretrained=pretrained, num_classes=1)

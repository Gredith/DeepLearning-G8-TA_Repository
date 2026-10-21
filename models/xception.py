"""Xception model factory."""

import timm


def build_xception(pretrained: bool = True):
    """Build Xception with one binary-classification logit per image."""
    return timm.create_model("legacy_xception", pretrained=pretrained, num_classes=1)

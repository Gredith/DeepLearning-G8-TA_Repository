"""Acceptance checks for the focal loss used by the Kaggle E2 notebooks (src/kaggle_e2/losses.py)."""

import importlib.util
import unittest
from pathlib import Path

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
LOSSES_PATH = Path(__file__).resolve().parents[1] / "src" / "kaggle_e2" / "losses.py"


def _load_losses():
    # src/kaggle_e2 holds flat notebook modules, so load the file directly instead of as a package.
    spec = importlib.util.spec_from_file_location("kaggle_e2_losses", LOSSES_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is installed in the Kaggle/runtime environment")
class KaggleE2FocalLossTests(unittest.TestCase):
    def test_gamma_zero_matches_bce_loss_and_gradients(self):
        import torch
        from torch import nn

        losses = _load_losses()
        torch.manual_seed(7)
        logits = torch.randn(1024) * 4
        targets = torch.randint(0, 2, (1024,)).float()
        focal_input = logits.clone().requires_grad_(True)
        bce_input = logits.clone().requires_grad_(True)
        focal = losses.build_loss(gamma=0.0)(focal_input, targets)
        bce = nn.BCEWithLogitsLoss()(bce_input, targets)
        focal.backward()
        bce.backward()
        self.assertTrue(torch.allclose(focal, bce, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(focal_input.grad, bce_input.grad, atol=1e-8))

    def test_easy_example_is_downweighted(self):
        import torch

        losses = _load_losses()
        criterion = losses.BinaryFocalLoss(gamma=2.0, reduction="none")
        values = criterion(torch.tensor([5.0, 0.0]), torch.tensor([1.0, 1.0]))
        self.assertLess(float(values[0]), float(values[1]) * 0.01)

    def test_extreme_logits_have_finite_loss_and_gradients(self):
        import torch

        losses = _load_losses()
        for gamma in (0.0, 1.0, 2.0, 5.0):
            logits = torch.tensor([-50.0, 50.0, 50.0, -50.0], requires_grad=True)
            targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
            loss = losses.BinaryFocalLoss(gamma=gamma)(logits, targets)
            loss.backward()
            self.assertTrue(torch.isfinite(loss), gamma)
            self.assertTrue(torch.isfinite(logits.grad).all(), gamma)


if __name__ == "__main__":
    unittest.main()

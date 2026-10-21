"""Acceptance checks for BinaryFocalLoss (E2 T3)."""

import importlib.util
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is installed in the Kaggle/runtime environment")
class BinaryFocalLossTests(unittest.TestCase):
    def test_gamma_zero_matches_bce(self):
        import torch
        from torch import nn

        from src.models.losses import BinaryFocalLoss

        torch.manual_seed(7)
        logits = torch.randn(128)
        targets = torch.randint(0, 2, (128,)).float()
        focal = BinaryFocalLoss(gamma=0, alpha=None)(logits, targets)
        bce = nn.BCEWithLogitsLoss()(logits, targets)
        self.assertTrue(torch.allclose(focal, bce, atol=1e-6, rtol=1e-6))

    def test_easy_example_is_downweighted(self):
        import torch

        from src.models.losses import BinaryFocalLoss

        criterion = BinaryFocalLoss(gamma=2, reduction="none")
        logits = torch.tensor([5.0, 0.0])
        targets = torch.tensor([1.0, 1.0])
        losses = criterion(logits, targets)
        self.assertLess(float(losses[0]), float(losses[1]) * 0.01)

    def test_extreme_logits_have_finite_loss_and_gradients(self):
        import torch

        from src.models.losses import BinaryFocalLoss

        logits = torch.tensor([-50.0, 50.0], requires_grad=True)
        targets = torch.tensor([0.0, 1.0])
        loss = BinaryFocalLoss(gamma=2)(logits, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_alpha_and_reduction_validation(self):
        from src.models.losses import BinaryFocalLoss

        with self.assertRaises(ValueError):
            BinaryFocalLoss(gamma=-1)
        with self.assertRaises(ValueError):
            BinaryFocalLoss(alpha=1.1)
        with self.assertRaises(ValueError):
            BinaryFocalLoss(reduction="invalid")


if __name__ == "__main__":
    unittest.main()

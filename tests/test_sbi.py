"""Pruebas de la Fase D (D-14): SBI, aumentos de robustez, ítems de train y una corrida de humo con recortes sintéticos.

No usan FF++: las imágenes son caras dibujadas con PIL.
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_E2 = ROOT / "src" / "kaggle_e2"
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("timm") is not None
if str(KAGGLE_E2) not in sys.path:
    sys.path.insert(0, str(KAGGLE_E2))


def fake_face(size=299, seed=0):
    """Cara sintética: fondo, óvalo de piel, ojos y boca, con ruido (no es un recorte de FF++)."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (size, size), tuple(int(c) for c in rng.integers(40, 200, 3)))
    d = ImageDraw.Draw(img)
    s = size / 299
    d.ellipse((75 * s, 55 * s, 225 * s, 255 * s), fill=(220, 180, 150))
    d.ellipse((115 * s, 120 * s, 135 * s, 135 * s), fill=(40, 40, 40))
    d.ellipse((165 * s, 120 * s, 185 * s, 135 * s), fill=(40, 40, 40))
    d.rectangle((130 * s, 200 * s, 170 * s, 210 * s), fill=(150, 60, 60))
    a = np.asarray(img, dtype=np.int16) + rng.integers(-12, 13, (size, size, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


class SBITests(unittest.TestCase):
    def setUp(self):
        import sbi
        self.sbi = sbi
        self.img = fake_face()

    def test_same_size_and_type(self):
        out = self.sbi.self_blend(self.img, np.random.default_rng(0))
        self.assertEqual(out.size, self.img.size)
        self.assertEqual(out.mode, "RGB")

    def test_deterministic_with_seed(self):
        a = np.asarray(self.sbi.self_blend(self.img, np.random.default_rng(5)))
        b = np.asarray(self.sbi.self_blend(self.img, np.random.default_rng(5)))
        c = np.asarray(self.sbi.self_blend(self.img, np.random.default_rng(6)))
        self.assertTrue((a == b).all())
        self.assertFalse((a == c).all())

    def test_mask_is_face_shaped(self):
        for seed in range(20):
            m = self.sbi.face_mask(299, np.random.default_rng(seed))
            self.assertEqual(m.shape, (299, 299))
            self.assertGreaterEqual(m.min(), 0.0)
            self.assertLessEqual(m.max(), 1.0)
            self.assertGreater(m[149, 149], 0.9, "el centro de la cara siempre se mezcla")
            for y, x in [(0, 0), (0, 298), (298, 0), (298, 298)]:
                self.assertLess(m[y, x], 0.01, "las esquinas (fondo) no se mezclan")
            area = (m > 0.5).mean()
            self.assertTrue(0.15 < area < 0.6, area)

    def test_blend_changes_image_mostly_inside_mask(self):
        changed_in, changed_out = [], []
        for seed in range(10):
            out, m = self.sbi.self_blend(self.img, np.random.default_rng(seed), return_mask=True)
            d = np.abs(np.asarray(out, np.float32) - np.asarray(self.img, np.float32)).mean(-1)
            # dentro cambia la mitad «fuente» si la copia alterada va dentro; si no, cambia fuera.
            changed_in.append(d[m > 0.9 * m.max()].mean()); changed_out.append(d[m < 0.01].mean())
        self.assertTrue(any(v > 1.0 for v in changed_in + changed_out), "el SBI no puede ser idéntico al real")

    def test_no_black_frame_from_misalignment(self):
        """La desalineación no debe dejar un marco negro (sería un atajo para reconocer los SBI)."""
        bright = Image.new("RGB", (299, 299), (200, 200, 200))
        for seed in range(30):
            out = np.asarray(self.sbi.self_blend(bright, np.random.default_rng(seed)))
            self.assertGreater(out[[0, -1], :, :].min(), 60, seed)
            self.assertGreater(out[:, [0, -1], :].min(), 60, seed)

    def test_jpeg_and_robust_aug(self):
        q = self.sbi.jpeg(self.img, 30)
        self.assertEqual(q.size, self.img.size)
        self.assertGreater(np.abs(np.asarray(q, np.float32) - np.asarray(self.img, np.float32)).mean(), 0.5)
        out = self.sbi.robust_aug(self.img, np.random.default_rng(0), p_jpeg=1.0, p_blur=1.0)
        self.assertEqual(out.size, self.img.size)
        same = self.sbi.robust_aug(self.img, np.random.default_rng(0), p_jpeg=0.0, p_blur=0.0)
        self.assertTrue((np.asarray(same) == np.asarray(self.img)).all())


def _write_crops(root, pairs, cats_by_split, n_frames=3, size=96):
    import data as D
    k = 0
    for pairs_split, cats in zip(pairs, cats_by_split):
        reals, fakes = D.video_ids(pairs_split)
        groups = [(D.REAL, reals)] + [(c, fakes) for c in cats]
        for cat, ids in groups:
            for vid in ids:
                d = Path(root) / cat / vid
                d.mkdir(parents=True, exist_ok=True)
                for f in range(n_frames):
                    img = fake_face(size, seed=k)
                    if cat != D.REAL:   # «falsos» sintéticos: un parche más claro en la cara
                        a = np.asarray(img).copy(); a[35:60, 35:60] = np.clip(a[35:60, 35:60] + 40, 0, 255)
                        img = Image.fromarray(a)
                    img.save(d / f"{f:03d}.png"); k += 1


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch y timm están en el runtime de Kaggle")
class TrainItemsAndSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import data as D
        cls.D = D
        cls.tmp = tempfile.TemporaryDirectory()
        cls.crops = Path(cls.tmp.name) / "crops"
        splits = ROOT / "results" / "splits"
        cls.tr = D.subset_pairs(D.load_pairs(splits, "train"), 2, 42)
        cls.va = D.subset_pairs(D.load_pairs(splits, "val"), 2, 42)
        cls.te = D.subset_pairs(D.load_pairs(splits, "test"), 2, 42)
        _write_crops(cls.crops, [cls.tr, cls.va, cls.te], [["Deepfakes"], ["Deepfakes"], D.METHODS])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_deepfakes_mode_matches_protocol_b(self):
        """Con train_fakes='deepfakes' los ítems (ruta, etiqueta, clave) son los mismos y en el mismo orden que antes."""
        old, miss_old = self.D.build_items(self.crops, self.tr, ["Deepfakes"], max_frames=3)
        new, miss_new = self.D.build_train_items(self.crops, self.tr, "deepfakes", ("Deepfakes",), max_frames=3)
        self.assertEqual([it[:3] for it in new], old)
        self.assertEqual(miss_old, miss_new)
        self.assertTrue(all(it[3] == "img" for it in new))

    def test_sbi_and_mix_modes_keep_balance(self):
        reals = sum(1 for it in self.D.build_train_items(self.crops, self.tr, "deepfakes")[0] if it[1] == 0)
        for mode in ("sbi", "deepfakes+sbi"):
            items, _ = self.D.build_train_items(self.crops, self.tr, mode, max_frames=3)
            n_fake = sum(1 for it in items if it[1] == 1)
            self.assertEqual(n_fake, reals, f"{mode}: balance 1:1 como en D-06")
        val, _ = self.D.build_train_items(self.crops, self.va, "deepfakes+sbi", max_frames=3, val=True)
        kinds = {it[3] for it in val}
        self.assertEqual(kinds, {"img", "sbi"}, "val: reales + Deepfakes + SBI deterministas")

    def test_dataset_val_is_deterministic(self):
        items, _ = self.D.build_train_items(self.crops, self.va, "deepfakes+sbi", max_frames=3, val=True)
        tf = self.D.eval_transform(96)
        a = self.D.FrameDataset(items, tf, fixed_seed=1234)
        b = self.D.FrameDataset(items, tf, fixed_seed=1234)
        i = next(k for k, it in enumerate(items) if it[3] == "sbi")
        self.assertTrue((a[i][0] == b[i][0]).all())

    def test_mix_items_are_sometimes_sbi(self):
        import torch
        items, _ = self.D.build_train_items(self.crops, self.tr, "deepfakes+sbi", max_frames=3)
        tf = self.D.eval_transform(96)
        ds = self.D.FrameDataset(items, tf, sbi_prob=1.0)
        i = next(k for k, it in enumerate(items) if it[3] == "mix")
        torch.manual_seed(0)
        x_sbi = ds[i][0]
        x_df = tf(Image.open(items[i][0]).convert("RGB"))
        self.assertFalse(torch.allclose(x_sbi, x_df), "con sbi_prob=1 el Deepfakes se reemplaza por un SBI")
        ds0 = self.D.FrameDataset(items, tf, sbi_prob=0.0)
        self.assertTrue(torch.allclose(ds0[i][0], x_df), "con sbi_prob=0 queda el Deepfakes")

    def test_smoke_run_all_variants(self):
        """Corrida completa (train → val → test + estrés JPEG) en CPU para cada variante del notebook 07."""
        import train
        orig = train.build_model
        train.build_model = lambda pretrained=True, num_classes=1: orig(pretrained=False, num_classes=num_classes)
        try:
            with tempfile.TemporaryDirectory() as out:
                variants = [("R0", "deepfakes", 0.0, False), ("R1", "deepfakes+sbi", 0.0, False),
                            ("R2", "deepfakes+sbi", 2.0, False), ("R3", "deepfakes+sbi", 2.0, True),
                            ("SBI", "sbi", 0.0, False)]
                for var, fakes, g, aug in variants:
                    cfg = train.Config(run_name=f"{var}_s42", variant=var, train_fakes=fakes, gamma=g, robust_aug=aug,
                                       stress_jpeg=(50,), crops_dir=str(self.crops), out_dir=out, ckpt_dir=out,
                                       n_train_pairs=2, n_val_pairs=2, n_test_pairs=2, frames_per_video=3,
                                       epochs=2, warmup_epochs=1, batch_size=8, img_size=96, num_workers=0, amp=False)
                    r = train.run(cfg, device="cpu", verbose=False)
                    self.assertIn("FaceSwap", r["test"])
                    self.assertIn("FaceSwap@jpeg50", r["test"])
                    self.assertIn("auc_video", r["test"]["FaceSwap"])
                    self.assertEqual(r["data"]["train"]["videos_faltantes"], 0)
                    saved = json.loads((Path(out) / f"{var}_s42.json").read_text())
                    self.assertEqual(saved["config"]["variant"], var)
                import analysis as A
                runs = A.load_runs(out)
                tab = A.metrics_table(runs)
                self.assertEqual(set(tab.grupo), {"R0", "R1", "R2", "R3", "SBI"})
                res = A.paired_bootstrap_groups(runs, "R1", "R0", test="FaceSwap", n_boot=50)
                self.assertIsNotNone(res)
                self.assertEqual(res["semillas"], 1)
        finally:
            train.build_model = orig


if __name__ == "__main__":
    unittest.main()

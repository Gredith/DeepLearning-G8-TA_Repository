"""Pure arithmetic checks for the receptive-field recurrence."""

import unittest

from src.teardown.receptive_field import advance_receptive_field, output_spatial_size


class ReceptiveFieldMathTests(unittest.TestCase):
    def test_valid_stem_convolutions_from_299_pixels(self):
        size = 299
        rf, jump = 1, 1
        rf, jump = advance_receptive_field(rf, jump, kernel=3, stride=2, padding=0)
        size = output_spatial_size(size, kernel=3, stride=2, padding=0)
        self.assertEqual((rf, jump, size), (3, 2, 149))
        rf, jump = advance_receptive_field(rf, jump, kernel=3, stride=1, padding=0)
        size = output_spatial_size(size, kernel=3, stride=1, padding=0)
        self.assertEqual((rf, jump, size), (7, 2, 147))

    def test_stride_and_dilation_scale_receptive_field(self):
        self.assertEqual(advance_receptive_field(7, 2, kernel=(3, 3), stride=(2, 2), dilation=(2, 2)), (15, 4))

    def test_legacy_xception_maximum_field_recurrence(self):
        rf, jump = 1, 1
        rf, jump = advance_receptive_field(rf, jump, 3, 2, 0)  # conv1
        rf, jump = advance_receptive_field(rf, jump, 3, 1, 0)  # conv2
        for reps, stride in ((2, 2), (2, 2), (2, 2), *((3, 1) for _ in range(8)), (2, 2)):
            for _ in range(reps):  # depthwise 3x3; pointwise 1x1 leaves RF unchanged
                rf, jump = advance_receptive_field(rf, jump, 3, 1, 1)
            if stride > 1:
                rf, jump = advance_receptive_field(rf, jump, 3, stride, 1)  # block max-pool
        for _ in range(2):  # exit-flow separable convolutions
            rf, jump = advance_receptive_field(rf, jump, 3, 1, 1)
        self.assertEqual((rf, jump), (1083, 32))


if __name__ == "__main__":
    unittest.main()

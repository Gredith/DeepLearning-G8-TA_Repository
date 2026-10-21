"""Dependency-light grouping checks for Xception teardown outputs."""

import unittest

from src.teardown.flops_params import flow_for_module


class FlowGroupingTests(unittest.TestCase):
    def test_maps_stem_and_blocks_to_expected_flow(self):
        self.assertEqual(flow_for_module("conv1"), "entry_flow")
        self.assertEqual(flow_for_module("block3.rep.1"), "entry_flow")
        self.assertEqual(flow_for_module("block4.rep.1"), "middle_flow")
        self.assertEqual(flow_for_module("block11"), "middle_flow")
        self.assertEqual(flow_for_module("block12.rep.1"), "exit_flow")
        self.assertEqual(flow_for_module("conv4"), "exit_flow")
        self.assertIsNone(flow_for_module("fc"))


if __name__ == "__main__":
    unittest.main()

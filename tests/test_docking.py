import unittest

import numpy as np

from src.docking import validate


class DockingTests(unittest.TestCase):
    def setUp(self):
        self.truth = dict(center_world_m=[2., 0., .25], yaw_deg=0.)

    def command(self, point=(2., 0., .25), yaw=0., **settings):
        return validate(self.truth, dict(midpoint_world=point, yaw_deg=yaw), **settings)

    def test_centered(self):
        row = self.command()
        self.assertTrue(row["success"])
        self.assertAlmostEqual(row["achieved_depth_m"], .34)

    def test_side_contact_including_exact_touch(self):
        for offset in (.08, .085):
            result = self.command((2.,offset,.25))
            self.assertEqual(result["status"], "collision")
            self.assertAlmostEqual(result["contact_fraction"], .22/.56)

    def test_yaw_clip(self):
        result = self.command(yaw=20.)
        self.assertEqual(result["status"], "collision")
        self.assertGreater(result["contact_fraction"], .5)
        self.assertLess(result["contact_fraction"], 1.)

    def test_command_too_short(self):
        self.assertEqual(self.command(insertion_m=.12)["status"], "insufficient_depth")

    def test_wrong_height_and_missed_target(self):
        self.assertEqual(self.command((2.,0.,.34))["status"], "collision")
        self.assertEqual(self.command((2.,2.,.25))["status"], "missed_opening")

    def test_thin_blocker(self):
        # A tiny solid occupying the left hole must still be detected.
        box = ([2.2,.25,.25], np.eye(3), [-.00005,-.03,-.03], [.00005,.03,.03], (0,0,0))
        row = self.command(obstacles=[box])
        self.assertEqual(row["status"], "collision")
        self.assertAlmostEqual(row["achieved_depth_m"], .19995)

    def test_rotated_world(self):
        truth = dict(center_world_m=[1.3,-.9,.25], yaw_deg=-37.)
        self.assertTrue(validate(truth,truth)["success"])

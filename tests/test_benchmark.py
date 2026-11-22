"""Regression checks for selective control and the denominators in the report."""
import unittest

import numpy as np

from benchmark import decide, design, evaluate, summarize
from src.scene import ground_truth, solid_boxes


class BenchmarkPolicyTests(unittest.TestCase):
    def test_design_and_seed_separation(self):
        dev, test = design("dev"), design("test")
        self.assertTrue({t.seed for t in dev}.isdisjoint({t.seed for t in test}))
        single = [t for t in test if t.kind == "single"]
        cells = {(t.layout, t.distance, t.yaw, t.occlusion) for t in single}
        self.assertEqual(len(cells), 180)
        self.assertEqual(len(test), 188)

    def test_ambiguous_detections_never_release(self):
        accepted = {"status": "ACCEPT", "center_world_m": [2., 0., .25], "yaw_deg": 0.}
        occluded = {"status": "REOBSERVE", "reason": "missing_opening"}
        self.assertEqual(decide([accepted, occluded]), (None, "AMBIGUOUS"))
        self.assertEqual(decide([occluded, occluded]), (None, "FRAGMENTED_OR_OCCLUDED"))
        self.assertEqual(decide([occluded]), (None, "missing_opening"))

    def test_wrong_release_counts_a_real_collision(self):
        trial = design("test")[0]
        truth = ground_truth(trial.pallets[0])
        solids = solid_boxes(trial.pallets)
        wrong = {"midpoint_world": np.asarray(truth["center_world_m"]) + [0., .18, 0.],
                 "yaw_deg": truth["yaw_deg"]}
        row = evaluate(trial, wrong, "ACCEPT", truth, solids, True, [], "test")
        self.assertEqual(row["dock_status"], "collision")
        refusal = evaluate(trial, None, "REOBSERVE", truth, solids, True, [], "test")
        aggregate = summarize([row, refusal])
        self.assertEqual(aggregate["wrong_release_rate"], 1.)
        self.assertEqual(aggregate["coverage"], .5)
        self.assertEqual(aggregate["feasible_rejections"], 1)
        self.assertEqual(aggregate["feasible_success_rate"], 0.)
        self.assertIsNone(summarize([refusal])["wrong_release_rate"])

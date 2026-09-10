"""CPU tests for physically relevant waypoint command validation."""

import copy
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

from scripts.co_rl.stair_oracle.waypoint_profile import load_profile, sample, validate_profile
from scripts.co_rl.stair_oracle.replay_guard import acquire_replay_lock


class WaypointTests(unittest.TestCase):
    def setUp(self):
        self.p = load_profile(Path(__file__).with_name("waypoints") / "example.json")

    def test_keyframes_and_final_hold(self):
        for w in self.p["waypoints"]:
            result = sample(self.p, w["t"])
            for actual, expected in zip(result["shoulder_rad"], w["shoulder_rad"]):
                self.assertAlmostEqual(actual, expected)
            self.assertEqual(result["shoulder_rad_s"], [0., 0.])
        self.assertEqual(sample(self.p, 10.)["shoulder_rad"], self.p["waypoints"][-1]["shoulder_rad"])

    def test_derivative_matches_position_curve(self):
        t, eps = .95, 1e-6
        for i in range(2):
            finite_difference = (sample(self.p, t+eps)["shoulder_rad"][i] - sample(self.p, t-eps)["shoulder_rad"][i])/(2*eps)
            self.assertAlmostEqual(finite_difference, sample(self.p,t)["shoulder_rad_s"][i], places=6)

    def test_reject_unachievable_speed_between_keyframes(self):
        self.p["waypoints"][3]["t"] = .851
        with self.assertRaisesRegex(ValueError, "interpolation"):
            validate_profile(self.p)

    def test_reject_nonfinite_and_motor_limit_violations(self):
        for key, value in (("wheel_rad_s", 40.01), ("feedforward_nm", 17.01), ("shoulder_rad", float('nan'))):
            with self.subTest(key=key):
                p = copy.deepcopy(self.p)
                p["waypoints"][0][key][0] = value
                with self.assertRaises(ValueError):
                    validate_profile(p)

    def test_reject_duplicate_times_and_out_of_scope_angle(self):
        p = copy.deepcopy(self.p)
        p["waypoints"][1]["t"] = 0
        with self.assertRaises(ValueError):
            validate_profile(p)
        self.p["scene"]["yaw_deg"] = 15
        with self.assertRaises(ValueError):
            validate_profile(self.p)

    def test_replay_lock_rejects_duplicate_and_releases(self):
        with tempfile.TemporaryDirectory() as d, patch('scripts.co_rl.stair_oracle.replay_guard.available_memory_kib', return_value=8*1024*1024):
            p = Path(d)/'lock'
            first = acquire_replay_lock(p)
            try:
                with self.assertRaisesRegex(RuntimeError, 'Another waypoint'):
                    acquire_replay_lock(p)
            finally:
                first.close()
            acquire_replay_lock(p).close()

    def test_low_memory_rejected_before_simulator_start(self):
        with tempfile.TemporaryDirectory() as d, patch('scripts.co_rl.stair_oracle.replay_guard.available_memory_kib', return_value=1024):
            with self.assertRaisesRegex(RuntimeError, 'RAM'):
                acquire_replay_lock(Path(d)/'lock')


if __name__ == '__main__':
    unittest.main()

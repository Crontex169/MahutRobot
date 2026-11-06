import unittest
import numpy as np
import sys
import os

# Add src to path
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from mahut_kol.utils.teleop_ps4 import MahutPS4Controller


class TestMahutPS4Controller(unittest.TestCase):
    def setUp(self):
        self.controller = MahutPS4Controller(deadzone=0.1, arm_scale=0.5, gripper_scale=0.2)

    def test_deadzone_zeroes_small_inputs(self):
        # All inputs within deadzone (< 0.1)
        action = self.controller.compute_action_from_inputs(
            lx=0.05, ly=-0.08, rx=0.02, ry=-0.05, l2=0.0, r2=0.0, droll=0.04
        )
        self.assertEqual(len(action), 7)
        np.testing.assert_allclose(action, np.zeros(7, dtype=np.float32), atol=1e-5)

    def test_left_stick_controls_joint0_and_joint1(self):
        # Stick X = 1.0, Y = 0.0 -> joint0 (taban yaw) should be active, joint1 0.0
        action = self.controller.compute_action_from_inputs(
            lx=1.0, ly=0.0, rx=0.0, ry=0.0, l2=0.0, r2=0.0
        )
        self.assertNotEqual(action[0], 0.0)
        self.assertEqual(action[1], 0.0)
        self.assertEqual(action[2], 0.0)
        self.assertEqual(action[3], 0.0)
        self.assertEqual(action[4], 0.0)

        # Stick X = 0.0, Y = -1.0 (pushed forward) -> joint1 (omuz pitch) should be active, joint0 0.0
        action_y = self.controller.compute_action_from_inputs(
            lx=0.0, ly=-1.0, rx=0.0, ry=0.0, l2=0.0, r2=0.0
        )
        self.assertEqual(action_y[0], 0.0)
        self.assertNotEqual(action_y[1], 0.0)

    def test_right_stick_controls_joint2_and_joint3(self):
        # Right stick Y = -1.0 -> joint2 (dirsek pitch) active
        action = self.controller.compute_action_from_inputs(
            lx=0.0, ly=0.0, rx=0.0, ry=-1.0, l2=0.0, r2=0.0
        )
        self.assertNotEqual(action[2], 0.0)
        self.assertEqual(action[0], 0.0)
        self.assertEqual(action[1], 0.0)
        self.assertEqual(action[3], 0.0)
        self.assertEqual(action[4], 0.0)

        # Right stick X = 1.0 -> joint3 (bilek pitch) active
        action_x = self.controller.compute_action_from_inputs(
            lx=0.0, ly=0.0, rx=1.0, ry=0.0, l2=0.0, r2=0.0
        )
        self.assertNotEqual(action_x[3], 0.0)
        self.assertEqual(action_x[2], 0.0)

    def test_droll_controls_joint4(self):
        # D-Pad / droll = 1.0 -> joint4 (bilek roll) active
        action = self.controller.compute_action_from_inputs(
            lx=0.0, ly=0.0, rx=0.0, ry=0.0, l2=0.0, r2=0.0, droll=1.0
        )
        self.assertNotEqual(action[4], 0.0)
        self.assertEqual(action[0], 0.0)
        self.assertEqual(action[1], 0.0)
        self.assertEqual(action[2], 0.0)
        self.assertEqual(action[3], 0.0)

    def test_triggers_control_gripper(self):
        # L2 triggers opening: negative delta towards joint lower limit (0.0)
        action_open = self.controller.compute_action_from_inputs(
            lx=0.0, ly=0.0, rx=0.0, ry=0.0, l2=1.0, r2=0.0
        )
        self.assertLess(action_open[5], 0.0)
        self.assertLess(action_open[6], 0.0)
        self.assertEqual(action_open[5], action_open[6])

        # R2 triggers closing: positive delta towards joint upper limit (0.042)
        action_close = self.controller.compute_action_from_inputs(
            lx=0.0, ly=0.0, rx=0.0, ry=0.0, l2=0.0, r2=1.0
        )
        self.assertGreater(action_close[5], 0.0)
        self.assertGreater(action_close[6], 0.0)
        self.assertEqual(action_close[5], action_close[6])

    def test_reset_check(self):
        # PS4 Triangle is button 2, Options is button 9
        self.assertTrue(self.controller.is_reset_requested(pressed_buttons=[2], pressed_keys=[]))
        self.assertTrue(self.controller.is_reset_requested(pressed_buttons=[9], pressed_keys=[]))
        self.assertTrue(self.controller.is_reset_requested(pressed_buttons=[], pressed_keys=['r']))
        self.assertFalse(self.controller.is_reset_requested(pressed_buttons=[0, 1], pressed_keys=[]))


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for Milestone 3: MultiShapeRelationalTask Environment.

Verifies:
1. Environment registration under both aliases
2. Relational target pose math for all 5 relations (numpy and torch)
3. Object pool integrity: exactly 9 manipulation objects loaded
4. Underground inactive object parking (z = -10.0 m)
5. Tabletop active object placement (z = 0.0175 m)
6. Pairwise non-overlapping distance between active objects (> 0.045 m)
7. Option overrides for manipulated, reference, and distractor selection
8. Observation extraction and evaluation metrics
"""

import math
import os
import sys
import unittest
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mahut_kol.envs
from mahut_kol.envs.multi_shape_task import (
    DEFAULT_OBJECT_HEIGHT,
    DEFAULT_RELATIONAL_DELTA,
    RELATIONS,
    TABLETOP_HALF_HEIGHT,
    MultiShapeRelationalTask,
    compute_all_relational_target_positions,
    compute_relational_target_position,
)


class TestRelationalTargetMath(unittest.TestCase):
    """Test standalone relational coordinate calculations."""

    def test_numpy_single_vector(self):
        ref = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        delta = 0.045
        height = 0.035

        front = compute_relational_target_position(ref, "FRONT", delta, height)
        np.testing.assert_allclose(front, [0.955, 2.0, 3.0], atol=1e-5)

        back = compute_relational_target_position(ref, "BACK", delta, height)
        np.testing.assert_allclose(back, [1.045, 2.0, 3.0], atol=1e-5)

        left = compute_relational_target_position(ref, "LEFT", delta, height)
        np.testing.assert_allclose(left, [1.0, 1.955, 3.0], atol=1e-5)

        right = compute_relational_target_position(ref, "RIGHT", delta, height)
        np.testing.assert_allclose(right, [1.0, 2.045, 3.0], atol=1e-5)

        top = compute_relational_target_position(ref, "TOP", delta, height)
        np.testing.assert_allclose(top, [1.0, 2.0, 3.035], atol=1e-5)

    def test_torch_batched_tensor(self):
        ref = torch.tensor([[0.20, 0.00, 0.0175], [0.15, -0.05, 0.0175]], dtype=torch.float32)
        delta = 0.045
        height = 0.035

        all_targets = compute_all_relational_target_positions(ref, delta, height)
        self.assertEqual(len(all_targets), 5)
        self.assertTrue(torch.allclose(all_targets["FRONT"][:, 0], torch.tensor([0.155, 0.105])))
        self.assertTrue(torch.allclose(all_targets["BACK"][:, 0], torch.tensor([0.245, 0.195])))
        self.assertTrue(torch.allclose(all_targets["LEFT"][:, 1], torch.tensor([-0.045, -0.095])))
        self.assertTrue(torch.allclose(all_targets["RIGHT"][:, 1], torch.tensor([0.045, -0.005])))
        self.assertTrue(torch.allclose(all_targets["TOP"][:, 2], torch.tensor([0.0525, 0.0525])))

    def test_invalid_relation(self):
        with self.assertRaises(ValueError):
            compute_relational_target_position([0, 0, 0], "DIAGONAL")


class TestMultiShapeRelationalTaskEnv(unittest.TestCase):
    """Integration tests for MultiShapeRelationalTask environment."""

    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("MahutKol-MultiShapeRelational-v1")

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_env_registration(self):
        """Verify environment registration under primary and compatibility aliases."""
        env2 = gym.make("MultiShapeRelationalTask-v0")
        self.assertIsNotNone(env2)
        env2.close()

    def test_pool_size(self):
        """Verify exactly 9 manipulation objects are initialized in scene."""
        unwrapped = self.env.unwrapped
        self.assertEqual(len(unwrapped.manipulation_objects), 9)

    def test_reset_and_spawning(self):
        """Verify active vs inactive objects placement on reset."""
        obs, info = self.env.reset(options={"num_distractors": 2})
        unwrapped = self.env.unwrapped

        self.assertIsNotNone(unwrapped.manipulated_object_id)
        self.assertIsNotNone(unwrapped.reference_object_id)
        self.assertEqual(len(unwrapped.distractor_object_ids), 2)

        active_keys = [unwrapped.manipulated_object_id, unwrapped.reference_object_id] + unwrapped.distractor_object_ids
        self.assertEqual(len(active_keys), 4)

        # Inactive count = 9 - 4 = 5
        inactive_keys = [k for k in unwrapped.manipulation_objects if k not in active_keys]
        self.assertEqual(len(inactive_keys), 5)

        # Check inactive objects are parked underground at z <= -5.0
        for inact in inactive_keys:
            pose = unwrapped.manipulation_objects[inact].pose.p.squeeze().tolist()
            self.assertLessEqual(pose[2], -5.0, f"Inactive object {inact} at z={pose[2]}")

        # Check active objects are on table (z close to 0.0175) and within workspace
        active_positions = []
        for act in active_keys:
            pose = unwrapped.manipulation_objects[act].pose.p.squeeze().tolist()
            self.assertAlmostEqual(pose[2], TABLETOP_HALF_HEIGHT, places=2)
            self.assertGreaterEqual(pose[0], unwrapped.workspace_x_bounds[0] - 0.01)
            self.assertLessEqual(pose[0], unwrapped.workspace_x_bounds[1] + 0.01)
            self.assertGreaterEqual(pose[1], unwrapped.workspace_y_bounds[0] - 0.01)
            self.assertLessEqual(pose[1], unwrapped.workspace_y_bounds[1] + 0.01)
            active_positions.append(np.array(pose[:2]))

        # Check pairwise distances between all active objects
        for i in range(len(active_positions)):
            for j in range(i + 1, len(active_positions)):
                dist = np.linalg.norm(active_positions[i] - active_positions[j])
                self.assertGreaterEqual(
                    dist, 0.045,
                    f"Active objects {active_keys[i]} and {active_keys[j]} collide! dist={dist}"
                )

    def test_explicit_option_overrides(self):
        """Verify passing explicit manipulated, reference, and relation options."""
        # Not: Bu test eskiden referans 'red_triangular_prism' + TOP kullanıyordu. Aşama 1'de
        # prizmanın üstü geçersiz ilan edildi (ValueError); bu yüzden TOP için silindir
        # referans kullanılıyor, prizma+TOP için ayrı bir hata testi var.
        options = {
            "manipulated_object": "blue_cylinder",
            "reference_object": "red_cylinder",
            "distractor_objects": ["green_cube"],
            "relation": "TOP",
        }
        obs, info = self.env.reset(options=options)
        unwrapped = self.env.unwrapped
        self.assertEqual(unwrapped.manipulated_object_id, "blue_cylinder")
        self.assertEqual(unwrapped.reference_object_id, "red_cylinder")
        self.assertEqual(unwrapped.distractor_object_ids, ["green_cube"])
        self.assertEqual(unwrapped.current_relation, "TOP")

        # Verify evaluation
        eval_metrics = unwrapped.evaluate()
        self.assertIn("success", eval_metrics)
        self.assertIn("distance_to_target", eval_metrics)


if __name__ == "__main__":
    unittest.main()

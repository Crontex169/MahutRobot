import unittest
import os
import sys
import torch
import numpy as np

# Add src to path
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import gymnasium as gym
import mahut_kol.envs
from mani_skill.utils.structs import Pose


class TestMahutKolTasks(unittest.TestCase):
    def test_minimal_task_initialization_and_step(self):
        """Mevcut MahutKol-Minimal-v0 ortamının 7 eklemli yeni modelle çalıştığını doğrula."""
        env = gym.make("MahutKol-Minimal-v0", obs_mode="state")
        obs, info = env.reset()
        # 7 qpos + 7 qvel = 14
        self.assertEqual(obs.shape, (1, 14))
        self.assertEqual(env.action_space.shape, (7,))
        action = np.zeros(7, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        env.close()

    def test_custom_push_cube_task_initialization(self):
        """MahutKol-PushCube-v0 ortamının başlatılması ve gözlem boyutları."""
        env = gym.make("MahutKol-PushCube-v0", obs_mode="state")
        obs, info = env.reset()
        # 14 (qpos+qvel) + 7 (tcp_pose) + 7 (cube_pose) + 3 (goal_pos) = 31
        self.assertEqual(obs.shape, (1, 31))
        self.assertEqual(env.action_space.shape, (7,))
        env.close()

    def test_custom_push_cube_task_step_and_reward(self):
        """Aksiyon adımı ve ödül hesaplama doğrulaması."""
        env = gym.make("MahutKol-PushCube-v0", obs_mode="state")
        obs, info = env.reset()
        action = np.zeros(7, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertIn("success", info)
        self.assertIn("dist_to_goal", info)
        self.assertTrue(isinstance(reward, torch.Tensor) or isinstance(reward, np.ndarray) or isinstance(reward, float))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        env.close()

    def test_custom_task_success_condition(self):
        """Küp hedef halkasına yerleştirildiğinde başarı sinyali üretildiğini doğrula."""
        env = gym.make("MahutKol-PushCube-v0", obs_mode="state")
        env.reset()
        unwrapped = env.unwrapped
        # Küpü doğrudan hedefin üzerine yerleştir
        goal_pos = unwrapped.goal_region.pose.p
        unwrapped.cube.set_pose(
            Pose.create_from_pq(
                p=torch.tensor([[goal_pos[0, 0], goal_pos[0, 1], unwrapped.cube_half_size]], device=unwrapped.device)
            )
        )
        eval_dict = unwrapped.evaluate()
        self.assertTrue(eval_dict["success"].item())
        env.close()

    def test_pick_cube_task_initialization(self):
        """Yeni entegre edilen MahutKol-PickCube-v1 ortamının başlatılması."""
        env = gym.make("MahutKol-PickCube-v1", obs_mode="state")
        obs, info = env.reset()
        # 14 (qpos+qvel) + 7 (cube_pose) = 21
        self.assertEqual(obs.shape, (1, 21))
        self.assertEqual(env.action_space.shape, (7,))
        action = np.zeros(7, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertIn("success", info)
        env.close()


if __name__ == "__main__":
    unittest.main()

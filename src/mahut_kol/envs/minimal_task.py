"""
MahutKol En Sade ManiSkill 3 Ortamı (Minimal Task).

Bu ortam, MahutKol robotunu sade bir masaüstü sahnede düzgün şekilde konumlandırır
ve eklem hareketlerini test etmek için hazır bir zemin sunar.
"""

from typing import Dict, Any, List
import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mahut_kol.robots.mahut_robot import MahutRobot


@register_env("MahutKol-Minimal-v0", max_episode_steps=500)
class MahutKolMinimalEnv(BaseEnv):
    """
    Sade Masaüstü Görev Ortamı.
    Robot masanın üzerine [0, 0, 0] konumunda yerleştirilir.
    """

    SUPPORTED_ROBOTS = ["mahut_kol"]
    agent: MahutRobot

    def __init__(self, *args, robot_uids: str = "mahut_kol", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self) -> List[CameraConfig]:
        """Robotu ve sahneyi izleyen ön/yan kamera."""
        pose = sapien_utils.look_at(eye=[0.45, -0.45, 0.40], target=[0.05, 0.0, 0.12])
        return [CameraConfig("cam_side", pose, 512, 512, np.deg2rad(55), 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        """GUI penceresi (human render) için 3. şahıs kamera."""
        pose = sapien_utils.look_at(eye=[0.50, -0.45, 0.45], target=[0.05, 0.0, 0.12])
        return CameraConfig("render_camera", pose, 512, 512, np.deg2rad(55), 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0.0558]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            # Robotu başlangıç pozisyonuna getir
            # [joint0, joint1, joint2, joint3, joint4, finger_left, finger_right]
            init_qpos = torch.tensor(
                [0.0, 0.4, 0.0, 0.0, 0.0, 0.02, 0.02],
                device=self.device,
                dtype=torch.float32,
            ).repeat(b, 1)
            self.agent.robot.set_qpos(init_qpos)

    def evaluate(self) -> Dict[str, torch.Tensor]:
        return {
            "success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)

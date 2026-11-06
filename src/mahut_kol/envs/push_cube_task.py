"""
MahutKol Örnek Özel Görev Ortamı (Custom Task: PushCube).

Bu ortam, ManiSkill 3 mimarisi üzerinde MahutKol için özel bir görev
nasıl tanımlanır, nesneler sahneye nasıl eklenir, gözlemler ve ödüller
nasıl yapılandırılır adımlarını somut olarak gösterir.

Görev Tanımı:
Robot masanın üzerindeki kırmızı hedef küpü, yeşil hedef bölgesine iter.
"""

from typing import Dict, Any, List
import numpy as np
import sapien
import torch

from transforms3d.euler import euler2quat

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mahut_kol.robots.mahut_robot import MahutRobot


@register_env("MahutKol-PushCube-v0", max_episode_steps=200)
class MahutKolPushCubeEnv(BaseEnv):
    """
    MahutKol Küp İtme (Push Cube) Görev Ortamı.

    Özellikler:
    - Dinamik hedef küp (kırmızı)
    - Kinematik hedef bölgesi (yeşil şeffaf hedef alanı)
    - Çoklu kamera desteği (cam_side, render_camera)
    - Başarı kontrolü (evaluate)
    - Şekillendirilmiş ödül fonksiyonu (dense_reward)
    """

    SUPPORTED_ROBOTS = ["mahut_kol"]
    agent: MahutRobot

    cube_half_size: float = 0.015  # 3cm küp (yarım kenar 1.5cm)
    goal_radius: float = 0.04     # 4cm hedef tolerans yarıçapı

    def __init__(self, *args, robot_uids: str = "mahut_kol", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self) -> List[CameraConfig]:
        """Eğitim (obs_mode='rgb'/'rgbd') ve algılama için kameralar."""
        side_pose = sapien_utils.look_at(eye=[0.45, -0.40, 0.35], target=[0.14, 0.0, 0.05])
        return [
            CameraConfig("cam_side", side_pose, 256, 256, np.deg2rad(55), 0.01, 100),
        ]

    @property
    def _default_human_render_camera_configs(self):
        """GUI penceresi (render_mode='human') veya video kaydı kamerası."""
        pose = sapien_utils.look_at(eye=[0.48, -0.42, 0.40], target=[0.12, 0.0, 0.05])
        return CameraConfig("render_camera", pose, 512, 512, np.deg2rad(55), 0.01, 100)

    def _load_agent(self, options: dict):
        """Robotu sahneye (masanın referans noktasına) yerleştir."""
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0.0558]))

    def _load_scene(self, options: dict):
        """Sahne elemanlarını (masa, küp, hedef alanı) oluştur."""
        # 1. Masa ve zemin
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        # 2. Dinamik Kırmızı Küp (İtilecek nesne)
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[0.9, 0.15, 0.15, 1.0],
            name="cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0.15, 0.0, self.cube_half_size]),
        )
        mat = sapien.physx.PhysxMaterial(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        for obj in self.cube._objs:
            for shape in obj.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).collision_shapes:
                shape.set_physical_material(mat)

        # 3. Görsel Hedef Bölgesi (Kinematik kırmızı-beyaz hedef halkası)
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0.14, 0.10, 1e-3]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Her epizot başlangıcında pozisyonları sıfırla / rastgeleleştir."""
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Robot başlangıç eklem açıları (hafif öne eğik hazır duruş)
            # [joint0, joint1, joint2, joint3, joint4, finger_left, finger_right]
            init_qpos = torch.tensor(
                [0.0, 0.4, 0.0, 0.0, 0.0, 0.02, 0.02],
                device=self.device,
                dtype=torch.float32,
            ).repeat(b, 1)
            self.agent.robot.set_qpos(init_qpos)

            # Hedef küpün masa üzerindeki başlangıç pozisyonu (hafif rastgelelik)
            cube_xyz = torch.zeros((b, 3), device=self.device)
            cube_xyz[:, 0] = 0.15 + (torch.rand((b,), device=self.device) - 0.5) * 0.02
            cube_xyz[:, 1] = 0.00 + (torch.rand((b,), device=self.device) - 0.5) * 0.04
            cube_xyz[:, 2] = self.cube_half_size
            self.cube.set_pose(Pose.create_from_pq(p=cube_xyz))

            # Hedef bölgesi pozisyonu (masa yüzeyinde yatay hedef halkası)
            goal_xyz = torch.zeros((b, 3), device=self.device)
            goal_xyz[:, 0] = 0.14
            goal_xyz[:, 1] = 0.10
            goal_xyz[:, 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=goal_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

    def _get_obs_extra(self, info: dict) -> Dict[str, torch.Tensor]:
        """Politika ve ajan için ek durum (state) bilgileri."""
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                cube_pose=self.cube.pose.raw_pose,
                goal_pos=self.goal_region.pose.p,
            )
        return obs

    def evaluate(self) -> Dict[str, torch.Tensor]:
        """Görev başarı kriteri."""
        cube_pos = self.cube.pose.p
        goal_pos = self.goal_region.pose.p

        # 2D düzlemde küp ile hedef arasındaki Öklid mesafesi
        dist_to_goal = torch.linalg.norm(cube_pos[:, :2] - goal_pos[:, :2], dim=1)

        # Küp hedef dairesi içinde mi ve masadan düşmemiş mi?
        is_in_goal = dist_to_goal <= self.goal_radius
        is_on_table = cube_pos[:, 2] <= (self.cube_half_size + 0.03)
        success = torch.logical_and(is_in_goal, is_on_table)

        return {
            "success": success,
            "dist_to_goal": dist_to_goal,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
        """Pekiştirmeli öğrenme için şekillendirilmiş (dense) ödül."""
        tcp_pos = self.agent.tcp.pose.p
        cube_pos = self.cube.pose.p

        # 1. Aşama: Robottan küpe yaklaşma ödülü
        tcp_to_cube_dist = torch.linalg.norm(tcp_pos - cube_pos, dim=1)
        reaching_reward = 1.0 - torch.tanh(5.0 * tcp_to_cube_dist)

        # 2. Aşama: Küpü hedefe yaklaştırma ödülü
        cube_to_goal_dist = info["dist_to_goal"]
        push_reward = 1.0 - torch.tanh(4.0 * cube_to_goal_dist)

        # Toplam ödül (başarı bonusu ile)
        reward = reaching_reward + 2.0 * push_reward
        reward += info["success"].float() * 3.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6.0

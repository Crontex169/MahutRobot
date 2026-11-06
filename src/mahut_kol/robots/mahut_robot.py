"""
MahutRobot: MahutKol için Sade ve Temiz ManiSkill 3 BaseAgent Robot Tanımı.

Özellikler:
- 5-DOF Kol: joint0 (taban yaw), joint1 (omuz pitch), joint2 (dirsek pitch), joint3 (bilek pitch), joint4 (bilek roll)
- 2-Parmak Paralel Tutucu: finger_left, finger_right
"""

import copy
import os
from typing import Dict, List, Optional
import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import (
    PDJointPosControllerConfig,
    deepcopy_dict,
)
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose


_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_URDF_PATH = os.path.abspath(
    os.path.join(_CURRENT_DIR, "../../../assets/urdf/mahut_kol.urdf")
)
_DEFAULT_MJCF_PATH = os.path.abspath(
    os.path.join(_CURRENT_DIR, "../../../assets/mahut_kol_maniskill.xml")
)


@register_agent()
class MahutRobot(BaseAgent):
    """
    MahutKol Robot Sınıfı (ManiSkill 3 uyumlu, 5-DOF Kol + 2-Parmak Tutucu).
    """

    uid = "mahut_kol"
    urdf_path = _DEFAULT_URDF_PATH
    disable_self_collisions: bool = True

    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            finger_left_link=dict(material="gripper", patch_radius=0.05, min_patch_radius=0.02),
            finger_right_link=dict(material="gripper", patch_radius=0.05, min_patch_radius=0.02),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([0.0, 0.4, 0.0, 0.0, 0.0, 0.02, 0.02], dtype=np.float32),
            pose=sapien.Pose(p=[0, 0, 0.0558]),
        ),
        zero=Keyframe(
            qpos=np.zeros(7, dtype=np.float32),
            pose=sapien.Pose(p=[0, 0, 0.0558]),
        ),
    )

    arm_joint_names: List[str] = [
        "joint0",
        "joint1",
        "joint2",
        "joint3",
        "joint4",
    ]
    gripper_joint_names: List[str] = [
        "finger_left",
        "finger_right",
    ]
    ee_link_name: str = "link_gripper_center"

    @property
    def _controller_configs(self) -> Dict:
        """
        Robot kontrolcü yapılandırması:
        - pd_joint_pos: Mutlak eklem pozisyonu kontrolü
        - pd_joint_delta_pos: Artımlı (delta) pozisyon kontrolü
        """
        active_joint_names = [j.name for j in self.robot.active_joints]

        # 5 kol eklemi için Feetech FT5330M stall torku (3.432 Nm)
        # 2 tutucu parmak için gerçekçi kavrama kuvveti (15 N) ve esneklik sönümlemesi
        stiffness = [80.0, 80.0, 80.0, 80.0, 80.0, 150.0, 150.0]
        damping = [10.0, 10.0, 10.0, 10.0, 10.0, 15.0, 15.0]
        force_limit = [3.432, 3.432, 3.432, 3.432, 3.432, 15.0, 15.0]

        pd_joint_pos = PDJointPosControllerConfig(
            active_joint_names,
            lower=None,
            upper=None,
            stiffness=stiffness,
            damping=damping,
            force_limit=force_limit,
            normalize_action=False,
        )

        delta_lower = [-0.05, -0.05, -0.05, -0.05, -0.05, -0.005, -0.005]
        delta_upper = [0.05, 0.05, 0.05, 0.05, 0.05, 0.005, 0.005]

        pd_joint_delta_pos = PDJointPosControllerConfig(
            active_joint_names,
            lower=delta_lower,
            upper=delta_upper,
            stiffness=stiffness,
            damping=damping,
            force_limit=force_limit,
            use_delta=True,
            use_target=False,
            normalize_action=True,
        )

        pd_joint_target_delta_pos = copy.deepcopy(pd_joint_delta_pos)
        pd_joint_target_delta_pos.use_target = True

        controller_configs = dict(
            pd_joint_delta_pos=pd_joint_delta_pos,
            pd_joint_pos=pd_joint_pos,
            pd_joint_target_delta_pos=pd_joint_target_delta_pos,
        )
        return deepcopy_dict(controller_configs)

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        self.finger1_link = self.robot.links_map["finger_left_link"]
        self.finger2_link = self.robot.links_map["finger_right_link"]
        self.tcp = self.robot.links_map[self.ee_link_name]

    @property
    def tcp_pos(self):
        """Uç organın global 3D pozisyonu."""
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        """Uç organın global pozu."""
        return self.tcp.pose

    def is_grasping(self, object: Actor, min_force: float = 0.5) -> torch.Tensor:
        """Kavrama kuvveti denetimi."""
        l_forces = self.scene.get_pairwise_contact_forces(self.finger1_link, object)
        r_forces = self.scene.get_pairwise_contact_forces(self.finger2_link, object)
        l_norm = torch.linalg.norm(l_forces, axis=1)
        r_norm = torch.linalg.norm(r_forces, axis=1)
        return torch.logical_and(l_norm >= min_force, r_norm >= min_force)

    def is_static(self, threshold: float = 0.2) -> torch.Tensor:
        """Robotun durağanlık denetimi."""
        qvel = self.robot.get_qvel()[:, :5]
        return torch.max(torch.abs(qvel), dim=1)[0] <= threshold

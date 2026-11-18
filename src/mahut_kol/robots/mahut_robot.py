"""
MahutRobot: yeni MahutKol (5-DOF + tek servolu paralel kıskaç) için ManiSkill 3 BaseAgent.

- Kol: joint0 (taban yaw), joint1-3 (paralel pitch), joint4 (bilek roll = yaklaşma ekseni).
  Hepsi Feetech FT5330M, ±90°, 3.4 Nm.
- Kıskaç: tek servo (gerçekte j6, 0 = açık, π = kapalı) iki prismatik parmağı
  0..0.018 m sürer (parmak = açı · GRIPPER_M_PER_RAD). URDF'te servo eklemi yok; eylemdeki
  servo açısı kontrolcüde iki parmak hedefine çevrilir. Temas anında parmak durur, servo da
  durur; bu yüzden kıskaç state'i parmak konumundan geri hesaplanır.
- Eylem (pd_joint_pos, 6): [joint0..4 (rad), kıskaç servo açısı (rad)].
  Robot qpos (7): [joint0..4, finger_left, finger_right (m)]. Dönüşüm: robot_state().
"""

import os
from typing import Dict, List

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import (
    PDJointPosController,
    PDJointPosControllerConfig,
    deepcopy_dict,
)
from mani_skill.agents.registration import register_agent
from mani_skill.utils.structs.actor import Actor
from gymnasium import spaces

_DEFAULT_URDF_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets/urdf/mahut_kol.urdf")
)

ARM_DOF = 5
FINGER_TRAVEL = 0.018                     # m, parmak başına (URDF limiti)
GRIPPER_OPEN = 0.0                        # rad
GRIPPER_CLOSED = float(np.pi)             # rad
GRIPPER_M_PER_RAD = FINGER_TRAVEL / np.pi  # MJCF equality polycoef 0.00572958
ARM_LIMIT = 1.5708                        # FT5330M ±90°
ACTION_DIM = ARM_DOF + 1
STATE_NAMES = ["joint0", "joint1", "joint2", "joint3", "joint4", "gripper"]


def robot_state(qpos):
    """Robot qpos (..., 7) -> kayıt/eylem şeması (..., 6): 5 kol + kıskaç servo açısı (rad)."""
    if isinstance(qpos, torch.Tensor):
        return torch.cat([qpos[..., :ARM_DOF], qpos[..., ARM_DOF:ARM_DOF + 2].mean(-1, keepdim=True) / GRIPPER_M_PER_RAD], -1)
    q = np.asarray(qpos)
    return np.concatenate([q[..., :ARM_DOF], q[..., ARM_DOF:ARM_DOF + 2].mean(-1, keepdims=True) / GRIPPER_M_PER_RAD], -1)


class GripperServoController(PDJointPosController):
    """Tek eylem (servo açısı, rad) -> iki parmağın ortak konum hedefi (m)."""

    def _initialize_action_space(self):
        self.single_action_space = spaces.Box(
            np.array([GRIPPER_OPEN], np.float32), np.array([GRIPPER_CLOSED], np.float32), dtype=np.float32
        )

    def _preprocess_action(self, action):
        return super()._preprocess_action(action) * GRIPPER_M_PER_RAD


class GripperServoControllerConfig(PDJointPosControllerConfig):
    controller_cls = GripperServoController


@register_agent()
class MahutRobot(BaseAgent):
    uid = "mahut_kol"
    urdf_path = _DEFAULT_URDF_PATH
    disable_self_collisions: bool = True

    urdf_config = dict(
        _materials=dict(gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)),
        link=dict(
            finger_left_link=dict(material="gripper", patch_radius=0.05, min_patch_radius=0.02),
            finger_right_link=dict(material="gripper", patch_radius=0.05, min_patch_radius=0.02),
        ),
    )

    keyframes = dict(
        zero=Keyframe(qpos=np.zeros(7, dtype=np.float32), pose=sapien.Pose(p=[0, 0, 0])),
    )

    arm_joint_names: List[str] = ["joint0", "joint1", "joint2", "joint3", "joint4"]
    gripper_joint_names: List[str] = ["finger_left", "finger_right"]
    ee_link_name: str = "link_gripper_center"

    # Kol: FT5330M 3.4 Nm tork sınırı; yüksek kazanç gerçek servonun sert konum döngüsünü taklit
    # eder (yerçekimi altında ~0.3° kalıcı hata ≈ servo ölü bandı). Parmak: servo 0.1146 Nm /
    # GRIPPER_M_PER_RAD ≈ 20 N, iki parmağa ~10 N.
    ARM_STIFFNESS = 400.0
    ARM_DAMPING = 20.0
    ARM_FORCE = 3.4
    FINGER_STIFFNESS = 1000.0
    FINGER_DAMPING = 10.0
    FINGER_FORCE = 10.0

    @property
    def _controller_configs(self) -> Dict:
        arm = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.ARM_STIFFNESS,
            damping=self.ARM_DAMPING,
            force_limit=self.ARM_FORCE,
            normalize_action=False,
        )
        gripper = GripperServoControllerConfig(
            self.gripper_joint_names,
            lower=None,
            upper=None,
            stiffness=self.FINGER_STIFFNESS,
            damping=self.FINGER_DAMPING,
            force_limit=self.FINGER_FORCE,
            normalize_action=False,
        )
        return deepcopy_dict(dict(pd_joint_pos=dict(arm=arm, gripper=gripper)))

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        self.finger1_link = self.robot.links_map["finger_left_link"]
        self.finger2_link = self.robot.links_map["finger_right_link"]
        self.tcp = self.robot.links_map[self.ee_link_name]

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose

    def is_grasping(self, object: Actor, min_force: float = 0.5) -> torch.Tensor:
        l_forces = self.scene.get_pairwise_contact_forces(self.finger1_link, object)
        r_forces = self.scene.get_pairwise_contact_forces(self.finger2_link, object)
        return torch.logical_and(
            torch.linalg.norm(l_forces, axis=1) >= min_force, torch.linalg.norm(r_forces, axis=1) >= min_force
        )

    def is_static(self, threshold: float = 0.2) -> torch.Tensor:
        qvel = self.robot.get_qvel()[:, :ARM_DOF]
        return torch.max(torch.abs(qvel), dim=1)[0] <= threshold
